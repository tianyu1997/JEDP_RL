from panda_gym.envs.robots.panda import Panda
from panda_gym.pybullet import PyBullet
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random  # Add random for seed initialization
import wandb  # Add wandb for logging
from config import *  # Import configuration settings




def initialize_weights(module):
    """
    Initialize the weights of the model.
    Args:
        module (nn.Module): A module in the neural network.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)  # Xavier initialization for linear layers
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LSTM):
        for name, param in module.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)


class JacobianPredictor(nn.Module):
    def __init__(self, input_dim=3, output_dim=21, device=None, actor=None):
        """
        Initialize the Jacobian predictor model.
        Args:
            input_dim (int): Dimension of the input (end-effector position).
            output_dim (int): Dimension of the output (joint angles).
            device (str): Device to run the model on ('cpu' or 'cuda').
        """
        if device is None: 
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        super(JacobianPredictor, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.hidden_size),
            nn.ReLU(),
        )
        
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True)  # Ensure LSTM supports batch processing
        self.fc2 = self.confidece = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )
        self.confidece = nn.Sequential(
            nn.Linear(self.hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
    
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience, verbose=True
        )  # Replace StepLR with ReduceLROnPlateau
        self.clip_value = clip_value # Add gradient clipping value
        self.reset()
        self.apply(initialize_weights)  # Apply weight initialization
        self.actor_path = actor
        self.actor = None
        self.state = None
        if self.actor_path is not None:
            self.actor.load(self.actor_path)  # Load the pre-trained model if available
        self.to(device)

    def reset(self, batch_size=1):
        """
        Reset the hidden state of the LSTM.
        Args:
            batch_size (int): Number of parallel sequences (batch size).
        """
        self.hidden = (
            torch.zeros([1, batch_size, self.hidden_size], dtype=torch.float).to(self.device),
            torch.zeros([1, batch_size, self.hidden_size], dtype=torch.float).to(self.device)
        )

    def forward(self, input):
        """
        Forward pass of the model.
        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        x = self.fc1(input)
        x = x.view(x.size(0), 1, self.hidden_size)  # Adjust for batch size
        s, hidden = self.lstm(x, self.hidden)
        self.hidden = (hidden[0].detach(), hidden[1].detach())
        # if not self.training:
        self.state = torch.cat([s.squeeze(), self.hidden[0].squeeze(), self.hidden[1].squeeze()], dim=-1)
        self.state = self.state.detach().cpu().numpy()
        output = self.fc2(s)
        c = self.confidece(s)
        return output, c
    
    
    
    def collect_data(self, envs):
        """
        Collect data from multiple envs in parallel.
        Args:
            envs (list[Panda]): List of env instances.
            input_queues (list[deque]): List of input queues for each env.
            random_range (float): Range for random actions.
            device (torch.device): Device to move tensors to.
        Returns:
            torch.Tensor: Batched input tensor.
            torch.Tensor: Batched target tensor (Jacobian).
        """
        inputs, targets = [], []
        for env in envs:
            # Generate random action and update input queue
            old_obs = env.get_observation()
            action = get_random_action()
            env.set_action(action)
            env.sim.step()
            new_obs = env.get_observation()
            
            # Prepare input tensor
            input_tensor = torch.tensor(np.concatenate([old_obs, action, new_obs]), dtype=torch.float32).to(self.device).flatten()
            jacobian = env.get_jacobian()
            target_tensor = torch.flatten(torch.tensor(np.array(jacobian), dtype=torch.float32)).to(self.device)

           
            inputs.append(input_tensor)
            targets.append(target_tensor)

        # Stack inputs and targets for batch processing
        inputs = torch.stack(inputs)
        targets = torch.stack(targets)

        return inputs, targets
    
    def train_model(self, envs: list, epochs=epochs, batch_size=batch_size):
        """
        Train the Jacobian predictor model using multiple envs.
        Args:
            envs (list[Panda]): List of env instances to interact with.
            epochs (int): Number of training epochs.
            batch_size (int): Number of envs to use for parallel data collection.
        """
        assert len(envs) == batch_size, "Number of envs must match batch size."

        self.reset(batch_size)  # Reset LSTM hidden state for batch size
        sum_loss = 0
        step = 0

        for epoch in range(epochs):
            # Collect data from all envs
            inputs, targets = self.collect_data(envs)

            # Forward pass
            outputs, confidence = self.forward(inputs)

            # Compute loss with confidence
            self.optimizer.zero_grad()
            
            # mse_loss = F.mse_loss(outputs.squeeze(), targets)
            # print(mse_loss.item())
            # Adjust confidence penalty based on step
            confidence_weight = min(1, step / confidence_k + confidence_bias)  # Gradually increase confidence importance
            confidence_loss = torch.mean((1 - confidence) **2 ) * confidence_weight

            # Penalize large errors with high confidence
            prediction_loss = torch.mean(confidence.squeeze(-1) * (outputs.squeeze() - targets) ** 2)

            # Combine losses
            loss = prediction_loss + confidence_loss_coef * confidence_loss
            loss.backward()
            
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_value)

            # Update model parameters
            self.optimizer.step()
            sum_loss += loss.item()
            step += 1

            # Log metrics to wandb
            wandb.log({
                "epoch": epoch + 1,
                "step": step,
                "loss": loss.item(),
                "confidence": confidence.mean().item(),
                "confidence_penalty": confidence_loss.item(),
                "prediction_penalty": prediction_loss.item(),
                "confidence_weight": confidence_weight,
                "learning_rate": self.optimizer.param_groups[0]['lr']
            })

            # Log progress every 100 epochs
            if (epoch + 1) % log_interval == 0:
                avg_loss = sum_loss / log_interval
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.6f}')
                wandb.log({"avg_loss": avg_loss})
                sum_loss = 0  # Reset loss accumulator
                self.reset(batch_size)
                step = 0  # Reset step counter
                # Reset envs and input queues
                for env in envs:
                    env.reset()
                    env.sim.step()
            
            if (epoch + 1) % save_interval == 0:
                # Save model every 1000 epochs
                self.save_model( f'checkpoints/jacobian_predictor_epoch_{epoch + 1}.pth')
                # wandb.save(checkpoint_path)

        print("Training complete.")
        # Save the model
        self.save_model(predictior_checkpoint_path)
        # wandb.save(checkpoint_path)

    def predict(self, x): 
        pass

    def save_model(self, path="checkpoints/jacobian_predictor.pth"):
        """
        Save the model's state dictionary to the specified path.
        Args:
            path (str): Path to save the model.
        """
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path="checkpoints/jacobian_predictor.pth"):
        """
        Load the model's state dictionary from the specified path.
        Args:
            path (str): Path to load the model from.
        """
        self.load_state_dict(torch.load(path, map_location=self.device))
        # self.actor.load(self.actor_path)
        self.to(self.device)
        print(f"Model loaded from {path}")


def train_with_sweep(config=None):
    """
    Train the model using hyperparameter configurations from wandb sweep.
    Args:
        config (dict): Hyperparameter configuration provided by wandb.
    """
    with wandb.init(config=config):
        config = wandb.config

        # Set random seed for reproducibility
        set_seed(config.seed)

        # Initialize the PyBullet simulator and envs
        torch.autograd.set_detect_anomaly(True)
        envs = [
            get_env
            for _ in range(config.batch_size)
        ]
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        jp = JacobianPredictor(input_dim=input_dim, output_dim=output_dim, device=device)

        # Watch the model with wandb
        wandb.watch(jp, log="all")

        # Train the model
        jp.train_model(envs, epochs=config.epochs, batch_size=config.batch_size)

        # Finish wandb run
        wandb.finish()


if __name__ == "__main__":
    sweep = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if sweep:
    # Define sweep configuration
        sweep_config = {
            "method": "grid",  # Can be "random", "grid", or "bayes"
            "metric": {"name": "loss", "goal": "minimize"},
            "parameters": {
                "learning_rate": {"values": [0.001, 0.0005, 0.0001]},
                "batch_size": {"values": [8, 16, 32]},
                "epochs": {"value": 20000},
                "clip_value": {"values": [0.5, 1.0]},
                "seed": {"value": 42},
            },
        }

        # Initialize sweep
        sweep_id = wandb.sweep(sweep_config, project="jacobian-predictor")

        # Start sweep agent
        wandb.agent(sweep_id, function=train_with_sweep)
    else:
        wandb.init(project="jacobian-predictor", name="test")
        set_seed(seed)  # Set random seed for reproducibility
        batch_size = batch_size
        torch.autograd.set_detect_anomaly(True)
        envs = [
            get_env()
            for _ in range(batch_size)
        ]
        # Example: Load model for evaluation or resume training
        # Uncomment the following lines to load a saved model
        index = 1
        jp = JacobianPredictor(input_dim=input_dim, output_dim=output_dim, device=device, actor=None)
        # jp.load_model(f"checkpoints/jacobian_predictor_epoch_100000.pth")
        jp.train_model(envs, epochs=epochs, batch_size=batch_size)  # Resume training