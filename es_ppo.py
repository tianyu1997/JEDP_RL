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
from config import *
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR


# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPO(nn.Module):
    def __init__(self, obs_dim=3, action_dim=1, time_length=5, action_scale = [0.1, 0.1], exploration_action_scale = [0.03, 0.03]):
        super(PPO, self).__init__()
        self.data = dict({'actor':[], 'explorer':[]})
        self.action_scale = action_scale
        self.exploration_action_scale = exploration_action_scale
        input_dim = time_length * obs_dim + (time_length-1) * action_dim
        self.fc1   = nn.Sequential(
            nn.Linear(input_dim,128),
            nn.Linear(128,256),
            nn.Linear(256,128)
        )
        self.fc_mu = nn.Linear(128,action_dim)
        self.es_mu = nn.Linear(128,action_dim)
        self.fc_std  = nn.Linear(128,action_dim)
        self.es_std = nn.Linear(128,action_dim)
        self.fc_v = nn.Linear(128, 1)
        self.fc_es_v = nn.Linear(128, 1)
        self.fc_a = nn.Linear(action_dim, 128)
        self.fc_t = nn.Linear(256, obs_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.scheduler = LinearLR(self.optimizer, T_max=100)
        self.optimization_step = 0
        self.explorer_optimization_step = 0

        self._init_weights()
        # Move model to device
        self.to(device)
    
    def _init_weights(self):
        """Initialize model parameters."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def es_pi(self, x):
        x = F.relu(self.fc1(x))
        mu = self.exploration_action_scale[0] * torch.tanh(self.es_mu(x))
        std = self.exploration_action_scale[1] * F.softplus(self.es_std(x))
        return mu, std
    
    def es_v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_es_v(x)
        return v

    def pi(self, x):
        x = F.relu(self.fc1(x))
        mu = self.action_scale[0] * torch.tanh(self.fc_mu(x))
        std = self.action_scale[1] * F.softplus(self.fc_std(x))
        return mu, std
    
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

    def state_predictor(self, x, a):
        s = F.relu(self.fc1(x))
        a = F.relu(self.fc_a(a))
        x = torch.cat([s,a],dim=-1)
        x = self.fc_t(x)
        return x
        
    def put_data(self, transition, name='actor'):
        self.data[name].append(transition)
        
    def make_batch(self, name='actor'):
        s_batch, a_batch, r_batch, s_prime_batch, prob_a_batch, done_batch = [], [], [], [], [], []
        data = []

        for j in range(buffer_size):
            for i in range(minibatch_size):
                rollout = self.data[name].pop()
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

    def calc_advantage(self, data, name='actor'):
        data_with_adv = []
        for mini_batch in data:
            s, a, r, s_prime, done_mask, old_log_prob = mini_batch
            with torch.no_grad():
                if name == 'actor':
                    td_target = r + gamma * self.v(s_prime) * done_mask
                    delta = td_target - self.v(s)
                else:
                    td_target = r + gamma * self.es_v(s_prime) * done_mask
                    delta = td_target - self.es_v(s)
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

        
    def train_net(self, name='actor'):
        if len(self.data[name]) == minibatch_size * buffer_size:
            data = self.make_batch(name)
            data = self.calc_advantage(data, name)

            for i in range(K_epoch):
                for mini_batch in data:
                    s, a, r, s_prime, done_mask, old_log_prob, td_target, advantage = mini_batch
                    if name == 'actor':
                        mu, std = self.pi(s)
                        critic_loss = F.smooth_l1_loss(self.v(s) , td_target)
                    else:
                        mu, std = self.es_pi(s)
                        critic_loss = F.smooth_l1_loss(self.es_v(s) , td_target)
                    dist = Normal(mu, std)
                    entropy = dist.entropy()
                    log_prob = dist.log_prob(a)
                    ratio = torch.exp(log_prob - old_log_prob)  # a/b == exp(log(a)-log(b))

                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
                    actor_loss = -torch.min(surr1, surr2)
                    loss =  actor_loss + 0.5 * critic_loss + 0.01 * entropy.mean() 

                    self.optimizer.zero_grad()
                    loss.mean().backward()
                    nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                    self.optimizer.step()
                    
                    if name == 'actor':
                        self.optimization_step += 1
                    else:
                        self.explorer_optimization_step += 1

                    # 记录损失到wandb
            
            self.scheduler.step()
            if name == 'actor':
                wandb.log({f"{name}_loss": loss.mean().item(), f"{name}_optimization_step": self.optimization_step})
            else:
                wandb.log({f"{name}_loss": loss.mean().item(), f"{name}_optimization_step": self.explorer_optimization_step})

        
def main():
    set_seed(seed)  # 设置随机种子
    name = f'es_3_{minibatch_size}_{seed}_linearLR'
    wandb.init(project="JEDP_RL", name=name)  # 初始化wandb项目
    env = gym.make('PandaReach-v3', control_type="Joints",  reward_type="dense")
    
    len_deque = time_length * env_obs_dim + (time_length-1) * env.action_space.shape[0]
    model = PPO(env_obs_dim, env.action_space.shape[0], time_length=time_length, action_scale=[0.1, 0.1], exploration_action_scale=[0.03, 0.03])
    
    # model.load_state_dict('checkpoints/checkpoint_2000.pth')
    score = 0.0
    score_count = 0
    e_score = 0
    e_score_count = 0
    rollout = []
    e_rollout = []
    e_flag = 1
    done_queue = deque(maxlen=50)
    len_queue = deque(maxlen=50)
    
    
    for n_epi in range(epoisodes):
        loss_queue = deque(maxlen=time_length)
        input_queue = deque(maxlen=len_deque)
        s, _ = env.reset()
        s = get_input(s)
        for x in s:
            input_queue.append(x)
        done = False
        while len(input_queue) < len_deque:
            a = 0.05 * env.action_space.sample()
            s_prime, r, done, truncated, info = env.step(a)
            s_prime = get_input(s_prime)
            for x in a:
                input_queue.append(x)
            for x in s_prime:
                input_queue.append(x)

        count = 0
        while count < 200 and not done:
            while True:
                old_input = input_queue.copy()
                if e_flag:
                    mu, std = model.es_pi(torch.tensor(input_queue,dtype=torch.float).to(device))
                    std += 0.0000001
                    dist = Normal(mu, std)
                    a = dist.sample()
                    log_prob = dist.log_prob(a)

                    s_pre = model.state_predictor(torch.tensor(input_queue,dtype=torch.float).to(device), a)
                    a = a.detach().cpu().numpy()
                    s_prime, r, done, truncated, info = env.step(a)
                    s_prime = get_input(s_prime)
                    for x in a:
                        input_queue.append(x)
                    for x in s_prime:
                        input_queue.append(x)
                    
                    loss = F.smooth_l1_loss(s_pre, torch.tensor(s_prime,dtype=torch.float).to(device))
                    loss_queue.append(loss.item())
                    
                    if np.mean(loss_queue) < loss_threshold:
                        e_flag = 0
                    else:
                        e_flag = 1

                    model.optimizer.zero_grad()
                    loss.mean().backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    model.optimizer.step()
                    r = -loss.item()

                    e_rollout.append((old_input, a, r, input_queue, log_prob, done))
                    e_score += r
                    e_score_count += 1
                    count += 1
                    if len(e_rollout) == rollout_len:
                        model.put_data(e_rollout, 'explorer')
                        # print(len(es_model.data))
                        e_rollout = []
                        break
                    
                    if done:
                        break
                
                else:
                    mu, std = model.pi(torch.tensor(input_queue,dtype=torch.float).to(device))
                    std += 0.0000001
                    dist = Normal(mu, std)
                    a = dist.sample()
                    log_prob = dist.log_prob(a)
                    s_pre = model.state_predictor(torch.tensor(input_queue,dtype=torch.float).to(device), a)
                    a = a.detach().cpu().numpy()
                    s_prime, r, done, truncated, info = env.step(a)
                    # print(r)
                    s_prime = get_input(s_prime)
                    for x in a:
                        input_queue.append(x)
                    for x in s_prime:
                        input_queue.append(x)
                    loss = F.smooth_l1_loss(s_pre, torch.tensor(s_prime,dtype=torch.float).to(device))
                    loss_queue.append(loss.mean().item())
                    if np.mean(loss_queue) < loss_threshold:
                        e_flag = 0
                    else:
                        e_flag = 1
                    model.optimizer.zero_grad()
                    loss.mean().backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    model.optimizer.step()

                    rollout.append((old_input, a, r, input_queue, log_prob, done))
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
            model.train_net('explorer')

        done_queue.append(int(done))
        len_queue.append(count)
        if n_epi % print_interval == 0 and n_epi != 0:
            if score_count == 0:
                score_count += 1
            if e_score_count == 0:
                e_score_count += 1
            print("# of episode :{}, SR: {:.2f}, avg score : {:.5f}, ave_len:{:.2f}, optmization step: {}".format(n_epi, np.mean(done_queue), score/score_count, np.mean(len_queue), model.optimization_step))
            print("                  avg e_score : {:.5f}, e_optmization step: {}".format(e_score/e_score_count, model.explorer_optimization_step))
            wandb.log({"episode": n_epi, "success_rate:":np.mean(done_queue), "ave_len":np.mean(len_queue), "avg_score": score/score_count, "avg_e_score": e_score/e_score_count, "optmization step": model.optimization_step})  # 记录平均得分到wandb
            score = 0.0
            e_score = 0.0
            score_count = 0
            e_score_count = 0

            # 保存模型checkpoint
        if n_epi % save_interval == 0 and n_epi != 0:
            torch.save(model.state_dict(), f"checkpoints/{name}_checkpoint_{n_epi}.pth")
            
    torch.save(model.state_dict(), f"checkpoints/{name}_checkpoint_{n_epi}.pth")
    env.close()

if __name__ == '__main__':
    main()