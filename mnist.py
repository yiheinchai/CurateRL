import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import random

# Inference model: A simple CNN to classify MNIST images
class InferenceModel(nn.Module):
    def __init__(self):
        super(InferenceModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(torch.relu(self.conv2(x)), 2)
        x = x.view(-1, 32 * 7 * 7)  # Flatten
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Critic model: Predicts future reward (value) for each MNIST image
class CriticModel(nn.Module):
    def __init__(self):
        super(CriticModel, self).__init__()
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 1)  # Output predicted future reward

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Actor: Chooses whether to learn from the image based on the Critic's prediction
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
def train_actor_critic(num_epochs=5):
    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

    # Initialize models
    inference_model = InferenceModel()
    critic_model = CriticModel()
    actor = Actor(epsilon=0.1)

    # Optimizers
    inference_optimizer = optim.Adam(inference_model.parameters(), lr=0.001)
    critic_optimizer = optim.Adam(critic_model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data, target

            # Step 1: Critic predicts future reward for the image
            conv_features = torch.relu(inference_model.conv1(data))  # Shared feature extraction part
            conv_features = torch.max_pool2d(torch.relu(inference_model.conv2(conv_features)), 2)
            conv_features_flat = conv_features.view(-1, 32 * 7 * 7)
            expected_reward = critic_model(conv_features_flat)

            # Step 2: Actor decides whether to learn from this image
            action = actor.choose_action(expected_reward.mean().item())

            if action == 1:  # Actor decides to learn from the data point
                # Step 3: Inference model processes the data and computes the loss
                inference_optimizer.zero_grad()
                prediction = inference_model(data)
                loss = loss_fn(prediction, target)
                loss.backward()
                inference_optimizer.step()

                # Step 4: The loss is used as a negative reward (lower loss = higher reward)
                reward = -loss.item()

                # Step 5: Update the Critic using the loss (reward)
                critic_optimizer.zero_grad()
                target_value = torch.tensor([reward], dtype=torch.float32).expand_as(expected_reward)
                critic_loss = nn.MSELoss()(expected_reward, target_value)
                critic_loss.backward()
                critic_optimizer.step()

            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}, Reward: {reward:.4f}")

    return inference_model, critic_model

# Train the model
trained_inference_model, trained_critic_model = train_actor_critic(num_epochs=5)
