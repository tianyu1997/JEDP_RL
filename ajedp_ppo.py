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

class PPO_LSTM(nn.Module):
    def __init__(self, input_dim=3, output_dim=1):
        super(PPO, self).__init__()
        self.data = []
        
        self.fc1   = nn.Linear(input_dim,128)
        self.lstm = nn.LSTM(128,128)
        self.fc_mu = nn.Linear(128,output_dim)
        self.fc_std  = nn.Linear(128,output_dim)
        self.fc_v = nn.Linear(128,1)
        self.fc_s = nn.Linear(128,128)
        self.fc_cf = nn.Linear(128,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.optimization_step = 0

        # Move model to device
        self.to(device)

    def s(self, x, hidden):
        x = F.relu(self.fc1(x))
        x, hidden = self.lstm(x, hidden)
        self.state = self.fc_s(x)
        self.cf = self.fc_cf(x)
        return self.state, hidden

    def pi(self, x):
        action_scale = [0.1, 0.1]
        x, hidden = self.s(x, hidden)
        mu = action_scale[0] * torch.tanh(self.fc_mu(x))
        std = action_scale[1] * F.softplus(self.fc_std(x))
        return mu, std, hidden
    
    def v(self, x):
        x, hidden = self.s(x, hidden)
        v = self.fc_v(x)
        return v, hidden
      
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_batch, a_batch, r_batch, s_prime_batch, prob_a_batch, done_batch, h_in_batch, h_out_batch = [], [], [], [], [], [], [], []
        data = []

        for j in range(buffer_size):
            for i in range(minibatch_size):
                rollout = self.data.pop()
                s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst, h_in_lst, h_out_lst = [], [], [], [], [], [], [], []

                for transition in rollout:
                    s, a, r, s_prime, prob_a, done, h_in, h_out = transition
                
                    s_lst.append(s.detach().cpu().numpy())
                    a_lst.append(a.detach().cpu().numpy())
                    r_lst.append([r])
                    s_prime_lst.append(s_prime.detach().cpu().numpy())
                    prob_a_lst.append(prob_a.detach().cpu().numpy())
                    done_mask = 0 if done else 1
                    done_lst.append([done_mask])
                    h_in_lst.append(h_in.detach().cpu().numpy())
                    h_out_lst.append(h_out.detach().cpu().numpy())

                s_batch.append(s_lst)
                a_batch.append(a_lst)
                r_batch.append(r_lst)
                s_prime_batch.append(s_prime_lst)
                prob_a_batch.append(prob_a_lst)
                done_batch.append(done_lst)
                h_in_batch.append(h_in_lst)
                h_out_batch.append(h_out_lst)

                    
            mini_batch = torch.tensor(np.array(s_batch), dtype=torch.float).to(device), torch.tensor(np.array(a_batch), dtype=torch.float).to(device), \
                          torch.tensor(np.array(r_batch), dtype=torch.float).to(device), torch.tensor(np.array(s_prime_batch), dtype=torch.float).to(device), \
                          torch.tensor(np.array(done_batch), dtype=torch.float).to(device), torch.tensor(np.array(prob_a_batch), dtype=torch.float).to(device), \
                          torch.tensor(np.array(h_in_batch), dtype=torch.float).to(device), torch.tensor(np.array(h_out_batch), dtype=torch.float).to(device)
            data.append(mini_batch)

        return data

    def calc_advantage(self, data):
        data_with_adv = []
        for mini_batch in data:
            s, a, r, s_prime, done_mask, old_log_prob, h_in, h_out = mini_batch
            with torch.no_grad():
                vs = self.v(s_prime, h_out)
                td_target = r + gamma * vs * done_mask
                delta = td_target - vs
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

        
    def train_net(self):
        if len(self.data) == minibatch_size * buffer_size:
            data = self.make_batch()
            data = self.calc_advantage(data)

            for i in range(K_epoch):
                for mini_batch in data:
                    s, a, r, s_prime, done_mask, old_log_prob, td_target, advantage, h_in, h_out = mini_batch

                    mu, std = self.pi(s, h_in)
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
                    wandb.log({"loss": loss.mean().item(), "optimization_step": self.optimization_step})


class PPO(nn.Module):
    def __init__(self, input_dim=3, output_dim=1):
        super(PPO, self).__init__()
        self.data = []
        
        self.fc1   = nn.Linear(input_dim,128)
        self.fc_mu = nn.Linear(128,output_dim)
        self.fc_std  = nn.Linear(128,output_dim)
        self.fc_v = nn.Linear(128,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.optimization_step = 0

        # Move model to device
        self.to(device)

    def pi(self, x):
        action_scale = [0.1, 0.1]
        x = F.relu(self.fc1(x))
        mu = action_scale[0] * torch.tanh(self.fc_mu(x))
        std = action_scale[1] * F.softplus(self.fc_std(x))
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
                
                    s_lst.append(s.detach().cpu().numpy())
                    a_lst.append(a.detach().cpu().numpy())
                    r_lst.append([r])
                    s_prime_lst.append(s_prime.detach().cpu().numpy())
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

        
    def train_net(self):
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
                    wandb.log({"loss": loss.mean().item(), "optimization_step": self.optimization_step})

        
def main():
    wandb.init(project="JEDP_RL", name='ajedp')  # 初始化wandb项目
    env = gym.make('PandaReach-v3', control_type="Joints",  reward_type="dense")
    stater = PPO_LSTM(env.observation_space['desired_goal'].shape[0], env.action_space.shape[0])
    model = PPO(128, env.action_space.shape[0])
    score = 0.0
    print_interval = 20
    rollout = []
    
    for n_epi in range(10000):
        h_out = (torch.zeros([1, 1, 128], dtype=torch.float), torch.zeros([1, 1, 128], dtype=torch.float))
        obs, _ = env.reset()
        obs = obs['desired_goal']-obs['achieved_goal']
        obs = torch.tensor(obs, dtype=torch.float).to(device)
        done = False
        
        count = 0
        while count < 200 and not done:
            for t in range(rollout_len):
                for _ in range(5):
                    h_in = h_out
                    ea, h_out = stater.pi(obs, h_in)
                    if stater.cf > 0.6:
                        s = stater.state
                        break
                    obs, r, done, truncated, info = env.step(a.detach().cpu().numpy())
                    
                mu, std = model.pi(s)
                dist = Normal(mu, std)
                a = dist.sample()
                log_prob = dist.log_prob(a)
                obs, r, done, truncated, info = env.step(a.detach().cpu().numpy())
                r *= 10
                # print(r*100)
                s_prime = s_prime['desired_goal']-s_prime['achieved_goal']
                s_prime = torch.tensor(s_prime, dtype=torch.float).to(device)

                rollout.append((s, a, r, s_prime, log_prob, done))
                if len(rollout) == rollout_len:
                    model.put_data(rollout)
                    rollout = []
                
                s = s_prime
                score += r
                count += 1
                if done:
                    break

        
            model.train_net()
        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}, optmization step: {}".format(n_epi, score/print_interval, model.optimization_step))
            wandb.log({"episode": n_epi, "avg_score": score/print_interval})  # 记录平均得分到wandb
            score = 0.0

            # 保存模型checkpoint
        if n_epi % save_interval == 0 and n_epi != 0:
            torch.save(model.state_dict(), f"checkpoints/ajedp_checkpoint_{n_epi}.pth")

    env.close()

if __name__ == '__main__':
    main()