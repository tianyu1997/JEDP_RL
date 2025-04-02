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
learning_rate  = 0.0001
gamma           = 0.9
lmbda           = 0.9
eps_clip        = 0.15
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
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.optimization_step = 0
        self.rollout = []
        self.rollout_len = 3

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

    def pi(self, x):
        x = F.relu(self.fc1(x))
        mu = self.action_scale[0] * torch.tanh(self.fc_mu(x))
        std = self.action_scale[1] * F.softplus(self.fc_std(x))
        return mu, std
    
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v
        
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

def get_input(s):
    s_d = s['desired_goal']-s['achieved_goal']
    input = np.concatenate([s['achieved_goal'], s_d])
    return input
      
def main():
    wandb.init(project="JEDP_RL", name='ppo')  # 初始化wandb项目
    env = gym.make('PandaReach-v3', control_type="Joints",  reward_type="dense")
    l = 5
    len_deque = l * env.observation_space['desired_goal'].shape[0] + (l-1) * env.action_space.shape[0]
    model = PPO(len_deque, env.action_space.shape[0], action_scale=[0.1, 0.1])
    
    score = 0.0
    score_count = 0
    print_interval = 20
    rollout = []
   
    
    for n_epi in range(10000):
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
            for t in range(rollout_len):
                old_input = input_queue.copy()
                mu, std = model.pi(torch.tensor(input_queue,dtype=torch.float).to(device))
                dist = Normal(mu, std)
                a = dist.sample()
                log_prob = dist.log_prob(a)

                a = a.detach().cpu().numpy()
                s_prime, r, done, truncated, info = env.step(a)
                # print(r*100)
                s_prime = get_input(s_prime)
                for x in a:
                    input_queue.append(x)
                for x in s_prime:
                    input_queue.append(x)
            
                rollout.append((old_input, a, r, input_queue, log_prob, done))
                if len(rollout) == rollout_len:
                    model.put_data(rollout)
                    rollout = []
                score += r
                score_count += 1
                count += 1
                if done:
                    break

            model.train_net('actor')

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.5f}, optmization step: {}".format(n_epi, score/score_count, model.optimization_step))
            wandb.log({"episode": n_epi, "avg_score": score/score_count, "optmization step": model.optimization_step})  # 记录平均得分到wandb
            score = 0.0
            score_count = 0

            # 保存模型checkpoint
        if n_epi % save_interval == 0 and n_epi != 0:
            torch.save(model.state_dict(), f"checkpoints/ppo_checkpoint_{n_epi}.pth")

    env.close()

if __name__ == '__main__':
    main()