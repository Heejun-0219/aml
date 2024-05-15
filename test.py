import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# Define a simple neural network architecture
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(784, 128)  # From 784 inputs to 128 neurons
        self.fc2 = nn.Linear(128, 64)   # Second layer
        self.fc3 = nn.Linear(64, 10)    # Output layer for 10 classes

    def forward(self, x):
        x = torch.flatten(x, 1)  # Flatten the image
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x

# Load a trained model (assuming you have a saved model)
# This would typically be loaded with torch.load() for a .pt or .pth file
# net = torch.load('path_to_trained_model.pth')

# Instead, for illustration, we initialize a new network (not trained)
net = Network()

# MNIST data loading
transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

# Simplified prediction function using PyTorch
def predict(model, n):
    model.eval()
    with torch.no_grad():
        data, target = test_data[n]
        data = data.unsqueeze(0)  # Add batch dimension
        output = model(data)
        prediction = output.argmax(dim=1, keepdim=True)

        print('Network output: ')
        print(output)
        print('Network prediction: ')
        print(prediction.item())
        print('Actual image: ')

        plt.imshow(data.squeeze().numpy(), cmap='Greys')
        plt.show()

predict(net, 0)
