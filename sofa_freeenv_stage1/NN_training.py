import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random  # Add random for seed initialization
import wandb  # Add wandb for logging
from config_sofa import *  # Import configuration settings
from collections import deque


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
    def __init__(self, input_dim=13, output_dim=21, device=None, actor=None):
        """
        Initialize the Jacobian predictor model.
        Args:
            input_dim (int): Dimension of the input (obs + action + obs).
            output_dim (int): Dimension of the output (Jacobian matrix flattened).
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
        
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True)
        
        self.fc2 = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )
        
        self.confidence = nn.Sequential(
            nn.Linear(self.hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
    
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience, verbose=True
        )
        self.clip_value = clip_value
        self.reset()
        self.apply(initialize_weights)
        self.actor_path = actor
        self.actor = None
        self.state = None
        if self.actor_path is not None:
            self.actor.load(self.actor_path)
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
        x = x.view(x.size(0), 1, self.hidden_size)
        s, hidden = self.lstm(x, self.hidden)
        self.hidden = (hidden[0].detach(), hidden[1].detach())
        
        self.state = torch.cat([s.squeeze(), self.hidden[0].squeeze(), self.hidden[1].squeeze()], dim=-1)
        self.state = self.state.detach().cpu().numpy()
        
        output = self.fc2(s)
        c = self.confidence(s)
        return output, c
    
    def collect_episode_data(self, env, max_steps=1000):
        """
        Collect data from one complete episode until truncated=True.
        Args:
            env (Panda): Single environment instance.
            max_steps (int): Maximum steps per episode to prevent infinite loops.
        Returns:
            list: List of (input_tensor, target_tensor) tuples collected during the episode.
        """
        episode_data = []
        
        # Reset environment and get first observation
        obs, info = env.reset()
        old_obs = env.get_state()
        
        step_count = 0
        truncated = False
        
        while not truncated and step_count < max_steps:
            # Generate random action
            action = get_random_action()
            
            # Take step in environment
            obs, reward, done, truncated, info = env.step(action)
            # new_obs = env.get_observation()
            new_obs = obs
            
            # Prepare input tensor: [old_obs, action, new_obs]
            input_tensor = torch.tensor(
                np.concatenate([old_obs, action, new_obs]), 
                dtype=torch.float32
            ).to(self.device).flatten()
            
            # Get target Jacobian
            jacobian = env.jacobian(action)
            target_tensor = torch.flatten(
                torch.tensor(np.array(jacobian), dtype=torch.float32)
            ).to(self.device)
            
            # Store the sample
            episode_data.append((input_tensor, target_tensor))
            
            # Update for next iteration
            old_obs = new_obs
            step_count += 1
            
            # Check if episode should end (done or truncated)
            if done or truncated:
                break
        
        print(f"Episode completed with {len(episode_data)} samples, truncated: {truncated}, done: {done}")
        return episode_data
    
    def train_on_episode_data(self, episode_data, minibatch_size=32):
        """
        Train the neural network on collected episode data using minibatches.
        Args:
            episode_data (list): List of (input_tensor, target_tensor) tuples.
            minibatch_size (int): Size of minibatches for training.
        Returns:
            float: Average loss for this episode.
        """
        if len(episode_data) == 0:
            return 0.0
        
        # Convert episode data to tensors
        inputs = torch.stack([data[0] for data in episode_data])  # [episode_length, 13]
        targets = torch.stack([data[1] for data in episode_data])  # [episode_length, 21]
        
        total_loss = 0.0
        num_batches = 0
        
        # Create minibatches
        for i in range(0, len(episode_data), minibatch_size):
            end_idx = min(i + minibatch_size, len(episode_data))
            batch_inputs = inputs[i:end_idx]  # [batch_size, 13]
            batch_targets = targets[i:end_idx]  # [batch_size, 21]
            
            # Reset LSTM hidden state for each minibatch
            self.reset(batch_size=batch_inputs.size(0))
            
            # Forward pass
            outputs, confidence = self.forward(batch_inputs)
            
            # Compute loss with confidence
            self.optimizer.zero_grad()
            
            # Confidence loss
            confidence_loss = torch.mean((1 - confidence) ** 2)
            
            # Prediction loss weighted by confidence
            prediction_loss = torch.mean(
                confidence.squeeze(-1) * (outputs.squeeze() - batch_targets) ** 2
            )
            
            # Combine losses
            loss = prediction_loss + confidence_loss_coef * confidence_loss
            
            # Backward pass
            loss.backward()
            
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_value)
            
            # Update model parameters
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def train_model_episodic(self, env, epochs=epochs, minibatch_size=32):
        """
        Train the Jacobian predictor model using episodic data collection.
        Args:
            env (Panda): Single environment instance to interact with.
            epochs (int): Number of training epochs (episodes).
            minibatch_size (int): Size of minibatches for neural network updates.
        """
        total_episodes = 0
        total_samples = 0
        episode_losses = []
        
        for epoch in range(epochs):
            print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
            
            # Collect data from one complete episode
            episode_data = self.collect_episode_data(env)
            
            if len(episode_data) > 0:
                # Train on the collected episode data
                avg_loss = self.train_on_episode_data(episode_data, minibatch_size)
                episode_losses.append(avg_loss)
                
                total_episodes += 1
                total_samples += len(episode_data)
                
                # Calculate confidence statistics
                with torch.no_grad():
                    inputs = torch.stack([data[0] for data in episode_data])
                    self.reset(batch_size=inputs.size(0))
                    _, confidence = self.forward(inputs)
                    avg_confidence = confidence.mean().item()
                
                # Log metrics to wandb
                wandb.log({
                    "epoch": epoch + 1,
                    "episode": total_episodes,
                    "episode_length": len(episode_data),
                    "episode_loss": avg_loss,
                    "total_samples": total_samples,
                    "avg_confidence": avg_confidence,
                    "learning_rate": self.optimizer.param_groups[0]['lr']
                })
                
                print(f"Episode {total_episodes}: {len(episode_data)} samples, Loss: {avg_loss:.6f}, Confidence: {avg_confidence:.4f}")
                
                # Update learning rate scheduler every few episodes
                if total_episodes % 10 == 0:
                    recent_loss = np.mean(episode_losses[-10:]) if len(episode_losses) >= 10 else avg_loss
                    self.scheduler.step(recent_loss)
                    
                    # Log average loss over recent episodes
                    wandb.log({"avg_loss_10_episodes": recent_loss})
                    print(f"Average loss over last 10 episodes: {recent_loss:.6f}")
            
            # Save model periodically
            if (epoch + 1) % save_interval == 0:
                self.save_model(f'checkpoints/jacobian_predictor_epoch_{epoch + 1}.pth')
                print(f"Model saved at epoch {epoch + 1}")

        print(f"\nTraining complete!")
        print(f"Total episodes: {total_episodes}")
        print(f"Total samples collected: {total_samples}")
        print(f"Average samples per episode: {total_samples / max(1, total_episodes):.1f}")
        
        # Save the final model
        self.save_model(predictior_checkpoint_path)

    def predict(self, x): 
        """
        Make a prediction for a given input.
        Args:
            x: Input tensor
        Returns:
            Prediction and confidence
        """
        self.eval()
        with torch.no_grad():
            output, confidence = self.forward(x)
        self.train()
        return output, confidence

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
        self.to(self.device)
        print(f"Model loaded from {path}")


def train_with_sweep_episodic(config=None):
    """
    Train the model using hyperparameter configurations from wandb sweep with episodic training.
    Args:
        config (dict): Hyperparameter configuration provided by wandb.
    """
    with wandb.init(config=config):
        config = wandb.config

        # Set random seed for reproducibility
        set_seed(config.seed)

        # Initialize single environment
        torch.autograd.set_detect_anomaly(True)
        env = get_env(name="endo")  # Single environment
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        jp = JacobianPredictor(input_dim=input_dim, output_dim=output_dim, device=device)

        # Watch the model with wandb
        wandb.watch(jp, log="all")

        # Train the model with episodic data collection
        jp.train_model_episodic(env, epochs=config.epochs, minibatch_size=config.get('minibatch_size', 32))

        # Finish wandb run
        wandb.finish()


if __name__ == "__main__":
    sweep = 0
    use_cpu = False  # Set to True to force CPU usage
    
    # Device selection
    # if use_cpu:
    #     device = torch.device("cpu")
    #     print("Using CPU for training")
    # else:
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     print(f"Using device: {device}")
    device = torch.device("cpu")
    if sweep:
        # Define sweep configuration (modified for episodic training)
        sweep_config = {
            "method": "grid",
            "metric": {"name": "episode_loss", "goal": "minimize"},
            "parameters": {
                "learning_rate": {"values": [0.001, 0.0005, 0.0001]},
                "epochs": {"value": 1000},  # Reduced since each epoch is now an episode
                "clip_value": {"values": [0.5, 1.0]},
                "minibatch_size": {"values": [16, 32, 64]},
                "seed": {"value": 42},
            },
        }

        # Initialize sweep
        sweep_id = wandb.sweep(sweep_config, project="jacobian-predictor-episodic")

        # Start sweep agent
        wandb.agent(sweep_id, function=train_with_sweep_episodic)
    else:
        wandb.init(project="jacobian-predictor-episodic", name="episodic_training_test")
        set_seed(seed)
        torch.autograd.set_detect_anomaly(True)
        
        # Create single environment
        env = get_env()
        
        # Create model
        jp = JacobianPredictor(input_dim=input_dim, output_dim=output_dim, device=device, actor=None)
        
        # Option to load pre-trained model
        # jp.load_model(f"checkpoints/jacobian_predictor_epoch_100000.pth")
        
        # Train the model with episodic data collection
        jp.train_model_episodic(env, epochs=1000, minibatch_size=32)  # Reduced epochs since each is an episode
        
        wandb.finish()