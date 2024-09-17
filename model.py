import torch
import torch.nn as nn
import torch.optim as optim
import random

# Inference model: A simple neural network to process data points
class InferenceModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(InferenceModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Critic model: Predicts future reward (value) for each data point
class CriticModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CriticModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)  # Output the predicted future reward

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Actor: Chooses whether to learn from the data point based on the Critic's prediction
class Actor:
    def __init__(self, epsilon):
        self.epsilon = epsilon  # Exploration vs exploitation

    def choose_action(self, expected_reward):
        # Epsilon-greedy decision: Explore or exploit based on the critic's prediction
        if random.uniform(0, 1) < self.epsilon:
            return random.choice([0, 1])  # Random action (0: skip, 1: learn)
        else:
            return 1 if expected_reward > 0 else 0  # Exploit (1: learn, 0: skip)

# Main RL process: Actor-Critic training loop
def train_actor_critic(data, input_size, hidden_size, output_size, num_epochs=1000):
    # Initialize models
    inference_model = InferenceModel(input_size, hidden_size, output_size)
    critic_model = CriticModel(input_size, hidden_size)
    actor = Actor(epsilon=0.1)

    # Optimizers
    inference_optimizer = optim.Adam(inference_model.parameters(), lr=0.001)
    critic_optimizer = optim.Adam(critic_model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):
        for data_point, label in data:
            data_point = torch.tensor(data_point, dtype=torch.float32)
            label = torch.tensor(label, dtype=torch.float32)

            # Step 1: Critic predicts future reward for the data point
            expected_reward = critic_model(data_point)

            # Step 2: Actor decides whether to learn from this data point
            action = actor.choose_action(expected_reward.item())

            if action == 1:  # Actor decides to learn from the data point
                # Step 3: Inference model processes the data and computes the loss
                inference_optimizer.zero_grad()
                prediction = inference_model(data_point)
                loss = loss_fn(prediction, label)
                loss.backward()
                inference_optimizer.step()

                # Step 4: The loss is used as a negative reward (lower loss = higher reward)
                reward = -loss.item()

                # Step 5: Update the Critic using the loss (reward)
                critic_optimizer.zero_grad()
                target = torch.tensor(reward).unsqueeze(0)
                critic_loss = loss_fn(expected_reward, target)
                critic_loss.backward()
                critic_optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Reward: {reward:.4f}")

    return inference_model, critic_model

# Example usage:
# Assume we have a dataset of 100 data points, each with 10 features and a single label.
data = [(torch.rand(10), torch.rand(1)) for _ in range(100)]
trained_inference_model, trained_critic_model = train_actor_critic(data, input_size=10, hidden_size=20, output_size=1)
