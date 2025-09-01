import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

# Load Data
df = pd.read_csv("sleep_data.csv")
X_real = df.drop("target", axis=1).values

# Define Generator
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# Train Generator
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = 10  # Adjust based on dataset
output_dim = X_real.shape[1]

generator = Generator(input_dim, output_dim).to(device)
optimizer = optim.Adam(generator.parameters(), lr=0.001)
criterion = nn.MSELoss()

for epoch in range(1000):  # Train for 1000 epochs
    noise = torch.randn((len(X_real), input_dim)).to(device)
    fake_data = generator(noise)
    loss = criterion(fake_data, torch.tensor(X_real, dtype=torch.float32).to(device))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

torch.save(generator.state_dict(), "generator_model.pth")
