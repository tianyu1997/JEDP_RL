from panda_gym.envs.robots.panda import Panda
from panda_gym.pybullet import PyBullet
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import numpy as np
import random  # Add random for seed initialization
import wandb  # Add wandb for logging

def set_seed(seed):
    """
    Set the random seed for reproducibility.
    Args:
        seed (int): The seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
    def __init__(self, input_dim=3, output_dim=21, device='cpu'):
        """
        Initialize the Jacobian predictor model.
        Args:
            input_dim (int): Dimension of the input (end-effector position).
            output_dim (int): Dimension of the output (joint angles).
            device (str): Device to run the model on ('cpu' or 'cuda').
        """
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        super(JacobianPredictor, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.hidden_size = 128
        self.lstm = nn.LSTM(128, self.hidden_size, batch_first=True)  # Ensure LSTM supports batch processing
        self.fc2 = nn.Linear(128, output_dim)
        self.confidece = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #     self.optimizer, mode='min', factor=0.1, patience=10, verbose=True
        # )  # Replace StepLR with ReduceLROnPlateau
        self.clip_value = 0.5  # Add gradient clipping value
        self.reset()
        self.apply(initialize_weights)  # Apply weight initialization
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

    def forward(self, input, hidden=None):
        """
        Forward pass of the model.
        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        x = self.fc1(input)
        x = x.view(x.size(0), 1, 128)  # Adjust for batch size
        s, hidden = self.lstm(x, self.hidden)
        self.hidden = (hidden[0].detach(), hidden[1].detach())
        output = self.fc2(s)
        c = self.sigmoid(self.confidece(s))
        return output, c
    
    def train_model(self, robots: list[Panda], epochs=10000, batch_size=8):
        """
        Train the Jacobian predictor model using multiple robots.
        Args:
            robots (list[Panda]): List of robot instances to interact with.
            epochs (int): Number of training epochs.
            batch_size (int): Number of robots to use for parallel data collection.
        """
        assert len(robots) == batch_size, "Number of robots must match batch size."

        random_range = 1
        input_queues = [deque(maxlen=13) for _ in range(batch_size)]
        for robot, queue in zip(robots, input_queues):
            robot.reset()
            queue.extend(robot.get_ee_position())
        self.reset(batch_size)  # Reset LSTM hidden state for batch size

        sum_loss = 0
        step = 0

        for epoch in range(epochs):
            inputs, targets, confidences = [], [], []

            # Collect data from all robots
            for robot, queue in zip(robots, input_queues):
                action = np.random.uniform(-random_range, random_range, 7)
                queue.extend(action)
                queue.extend(robot.get_ee_position())

                input_tensor = torch.tensor(queue.copy(), dtype=torch.float32).to(self.device)
                jacobian = robot.get_jacobian()
                target_tensor = torch.flatten(torch.tensor(np.array(jacobian), dtype=torch.float32)).to(self.device)

                inputs.append(input_tensor)
                targets.append(target_tensor)

            # Stack inputs and targets for batch processing
            inputs = torch.stack(inputs)
            targets = torch.stack(targets)

            # Forward pass
            outputs, confidence = self.forward(inputs)
            confidences.append(confidence)

            # Compute loss with confidence
            self.optimizer.zero_grad()

            # Adjust confidence penalty based on step
            confidence_weight = min(1, step / 20 + 0.1)  # Gradually increase confidence importance
            confidence_loss = torch.mean((1 - confidence) ** 2) * confidence_weight

            # Penalize large errors with high confidence
            prediction_loss = torch.mean(confidence.squeeze(-1) * (outputs.squeeze() - targets)**2)

            # Combine losses
            loss = prediction_loss + 0.01 * confidence_loss
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
            if (epoch + 1) % 100 == 0:
                avg_loss = sum_loss / 100
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}')
                wandb.log({"avg_loss": avg_loss})
                sum_loss = 0  # Reset loss accumulator
                self.reset(batch_size)
                step = 0  # Reset step counter
                # Reset robots and input queues
                for robot, queue in zip(robots, input_queues):
                    robot.reset()
                    queue.clear()
                    queue.extend(robot.get_ee_position())

        print("Training complete.")
        # Save the model
        checkpoint_path = 'checkpoints/jacobian_predictor.pth'
        torch.save(self.state_dict(), checkpoint_path)
        print(f"Model saved to {checkpoint_path}")
        wandb.save(checkpoint_path)

    def predict(self, x): 
        pass


if __name__ == "__main__":
    # Initialize wandb
    bacth_size = 32
    wandb.init(project="jacobian-predictor", config={
        "learning_rate": 0.001,
        "epochs": 100000,
        "batch_size": bacth_size,
        "scheduler": "ReduceLROnPlateau",
        "clip_value": 1.0,
        "seed": 42,
    })

    # Set random seed for reproducibility
    set_seed(42)

    # Initialize the PyBullet simulator and robots
    sim = PyBullet(render_mode="rgb_array", renderer="Tiny")
    torch.autograd.set_detect_anomaly(True)
    robots = [
        Panda(
            sim=sim,
            block_gripper=False,
            base_position=None,
            control_type="joints",
        )
        for _ in range(bacth_size)  # Create 8 robots for parallel training
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    jp = JacobianPredictor(input_dim=13, output_dim=21, device=device)

    # Watch the model with wandb
    wandb.watch(jp, log="all")

    jp.train_model(robots, epochs=100000, batch_size=bacth_size)

    # Finish wandb run
    wandb.finish()