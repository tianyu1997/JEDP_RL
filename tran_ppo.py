#PPO-LSTM
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import time
import numpy as np
import panda_gym
import gymnasium as gym
import wandb  # 导入wandb
from collections import deque

# Hyperparameters
learning_rate  = 0.0003
gamma           = 0.9
lmbda           = 0.9
eps_clip        = 0.2
K_epoch         = 10
rollout_len    = 3
buffer_size    = 10
minibatch_size = 32
save_interval  = 1000

# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPO(nn.Module):
    def __init__(self, input_dim=3, output_dim=1, action_scale = [0.1, 0.1]):
        super(PPO, self).__init__()
        self.data = []
        self.action_scale = action_scale
        self.fc1   = nn.Linear(input_dim,128)
        self.fc_mu = nn.Linear(128,output_dim)
        self.fc_std  = nn.Linear(128,output_dim)
        self.fc_v = nn.Linear(128, 1)
        self.fc_a = nn.Linear(output_dim, 128)
        self.fc_t = nn.Linear(256, 3)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.optimization_step = 0
        self.rollout = []
        self.rollout_len = 3

        # Move model to device
        self.to(device)

    def pi(self, x):
        x = F.relu(self.fc1(x))
        mu = self.action_scale[0] * torch.tanh(self.fc_mu(x))
        std = self.action_scale[1] * F.softplus(self.fc_std(x))
        return mu, std
    
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

    def tran_predictor(self, x, a):
        s = F.relu(self.fc1(x))
        a = F.relu(self.fc_a(a))
        x = torch.cat([s,a],dim=-1)
        x = self.fc_t(x)
        return x
        
      
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_batch, a_batch, r_batch, s_prime_batch, prob_a_batch, done_batch = [], [], [], [], [], []
        data = []

        for j in range(buffer_size):
            for i in range(minibatch_size):
                rollout = self.data.pop()
                s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []

                for transition in rollout:
                    s, a, r, s_prime, prob_a, done = transition
                
                    s_lst.append(s)
                    a_lst.append(a)
                    r_lst.append([r])
                    s_prime_lst.append(s_prime)
                    prob_a_lst.append(prob_a.detach().cpu().numpy())
                    done_mask = 0 if done else 1
                    done_lst.append([done_mask])

                s_batch.append(s_lst)
                a_batch.append(a_lst)
                r_batch.append(r_lst)
                s_prime_batch.append(s_prime_lst)
                prob_a_batch.append(prob_a_lst)
                done_batch.append(done_lst)
                    
            mini_batch = torch.tensor(np.array(s_batch), dtype=torch.float).to(device), torch.tensor(np.array(a_batch), dtype=torch.float).to(device), \
                          torch.tensor(np.array(r_batch), dtype=torch.float).to(device), torch.tensor(np.array(s_prime_batch), dtype=torch.float).to(device), \
                          torch.tensor(np.array(done_batch), dtype=torch.float).to(device), torch.tensor(np.array(prob_a_batch), dtype=torch.float).to(device)
            data.append(mini_batch)

        return data

    def calc_advantage(self, data):
        data_with_adv = []
        for mini_batch in data:
            s, a, r, s_prime, done_mask, old_log_prob = mini_batch
            with torch.no_grad():
                td_target = r + gamma * self.v(s_prime) * done_mask
                delta = td_target - self.v(s)
            delta = delta.cpu().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(np.array(advantage_lst), dtype=torch.float).to(device)
            data_with_adv.append((s, a, r, s_prime, done_mask, old_log_prob, td_target, advantage))

        return data_with_adv

        
    def train_net(self, logname):
        if len(self.data) == minibatch_size * buffer_size:
            data = self.make_batch()
            data = self.calc_advantage(data)

            for i in range(K_epoch):
                for mini_batch in data:
                    s, a, r, s_prime, done_mask, old_log_prob, td_target, advantage = mini_batch

                    mu, std = self.pi(s)
                    dist = Normal(mu, std)
                    log_prob = dist.log_prob(a)
                    ratio = torch.exp(log_prob - old_log_prob)  # a/b == exp(log(a)-log(b))

                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
                    loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , td_target)

                    self.optimizer.zero_grad()
                    loss.mean().backward()
                    nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimization_step += 1

                    # 记录损失到wandb
                    
            wandb.log({f"{logname}_loss": loss.mean().item(), f"{logname}_optimization_step": self.optimization_step})

        
def main():
    wandb.init(project="JEDP_RL", name='es')  # 初始化wandb项目
    env = gym.make('PandaReach-v3', control_type="Joints",  reward_type="dense")
    l = 5
    len_deque = l * env.observation_space['desired_goal'].shape[0] + (l-1) * env.action_space.shape[0]
    es_model = PPO(len_deque, env.action_space.shape[0], action_scale=[0.03, 0.03])
    model = PPO(128, env.action_space.shape[0], action_scale=[0.1, 0.1])
    model.load_state_dict('')
    score = 0.1
    score_count = 0
    e_score = 0
    e_score_count = 0
    print_interval = 20
    rollout = []
    e_rollout = []
    e_flag = 1
    e_loss_threshold = 0.00005
    
    
    for n_epi in range(10000):
        loss_queue = deque(maxlen=l)
        input_queue = deque(maxlen=len_deque)
        s, _ = env.reset()
        s = s['desired_goal']-s['achieved_goal']
        for x in s:
            input_queue.append(x)
        done = False
        while len(input_queue) < len_deque:
            a = 0.05 * env.action_space.sample()
            s_prime, r, done, truncated, info = env.step(a)
            s_prime = s_prime['desired_goal']-s_prime['achieved_goal']
            for x in a:
                input_queue.append(x)
            for x in s_prime:
                input_queue.append(x)

        count = 0
        while count < 300 and not done:
            while True:
                if e_flag:
                    
                    old_input = input_queue.copy()
                    mu, std = es_model.pi(torch.tensor(input_queue,dtype=torch.float).to(device))
                    std += 0.0000001
                    dist = Normal(mu, std)
                    a = dist.sample()
                    log_prob = dist.log_prob(a)

                    s_pre = es_model.tran_predictor(torch.tensor(input_queue,dtype=torch.float).to(device), a)
                    a = a.detach().cpu().numpy()
                    s_prime, r, done, truncated, info = env.step(a)
                    s_prime = s_prime['desired_goal']-s_prime['achieved_goal']
                    for x in a:
                        input_queue.append(x)
                    for x in s_prime:
                        input_queue.append(x)
                    
                    loss = F.smooth_l1_loss(s_pre, torch.tensor(s_prime,dtype=torch.float).to(device))
                    loss_queue.append(loss.item())
                    if np.mean(loss_queue) < e_loss_threshold:
                        e_flag = 0
                    else:
                        e_flag = 1
                    es_model.optimizer.zero_grad()
                    loss.mean().backward()
                    nn.utils.clip_grad_norm_(es_model.parameters(), 1.0)
                    es_model.optimizer.step()
                    r = -100*loss.item()

                    e_rollout.append((old_input, a, r, input_queue, log_prob, done))
                    e_score += r
                    e_score_count += 1
                    count += 1
                    if len(e_rollout) == rollout_len:
                        es_model.put_data(e_rollout)
                        # print(len(es_model.data))
                        e_rollout = []
                        break
                    
                    if done:
                        break
                
                else:
                    old_input = input_queue.copy()
                    state = es_model.fc1(torch.tensor(input_queue,dtype=torch.float).to(device))
                    mu, std = model.pi(state)
                    std += 0.0000001
                    dist = Normal(mu, std)
                    a = dist.sample()
                    log_prob = dist.log_prob(a)
                    s_pre = es_model.tran_predictor(torch.tensor(input_queue,dtype=torch.float).to(device), a)
                    a = a.detach().cpu().numpy()
                    s_prime, r, done, truncated, info = env.step(a)
                    # print(r)
                    s_prime = s_prime['desired_goal']-s_prime['achieved_goal']
                    for x in a:
                        input_queue.append(x)
                    for x in s_prime:
                        input_queue.append(x)
                    loss = F.smooth_l1_loss(s_pre, torch.tensor(s_prime,dtype=torch.float).to(device))
                    loss_queue.append(loss.item())
                    if np.mean(loss_queue) < e_loss_threshold:
                        e_flag = 0
                    else:
                        e_flag = 1
                    es_model.optimizer.zero_grad()
                    loss.mean().backward()
                    nn.utils.clip_grad_norm_(es_model.parameters(), 1.0)
                    es_model.optimizer.step()
                    state_prime = es_model.fc1(torch.tensor(input_queue,dtype=torch.float).to(device))
                    rollout.append((state.detach().cpu().numpy(), a, r, state_prime.detach().cpu().numpy(), log_prob, done))
                    score += r
                    score_count += 1
                    count += 1
                    if len(rollout) == rollout_len:
                        model.put_data(rollout)
                        rollout = []
                        break
                    
                    if done:
                        break


            model.train_net('actor')
            es_model.train_net('es')

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.5f}, optmization step: {}".format(n_epi, score/score_count, model.optimization_step))
            print("                  avg e_score : {:.5f}, e_optmization step: {}".format(e_score/e_score_count, es_model.optimization_step))
            wandb.log({"episode": n_epi, "avg_score": score/score_count, "avg_e_score": e_score/e_score_count, "optmization step": model.optimization_step})  # 记录平均得分到wandb
            score = 0.0
            e_score = 0.0
            score_count = 0
            e_score_count = 0

            # 保存模型checkpoint
        if n_epi % save_interval == 0 and n_epi != 0:
            torch.save(es_model.state_dict(), f"checkpoints/es_checkpoint_{n_epi}.pth")
            torch.save(model.state_dict(), f"checkpoints/checkpoint_{n_epi}.pth")

    env.close()

if __name__ == '__main__':
    main()