import torch
import torch.nn as nn
import torch.optim as optim

# Generate some dummy data (features and labels)
num_samples = 1000
num_features = 5
X = torch.randn(num_samples, num_features)
# Assuming the last column represents the number of cars passing through stop lights
y = torch.randint(low=0, high=10, size=(num_samples, 1)).float()

# Define a simple neural network model
class TrafficFlowModel(nn.Module):
    def __init__(self, input_size):
        super(TrafficFlowModel, self).__init__()
        self.fc = nn.Linear(input_size, 1)  # Assuming a single output for car count

    def forward(self, x):
        return self.fc(x)

model = TrafficFlowModel(num_features)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

# Once the model is trained,  use it to predict the car count for new data