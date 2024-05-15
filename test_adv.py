import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# Define a simple CNN architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Initialize network
net = CNN()
net.eval()  # Set the model to evaluation mode

# Load MNIST data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

# Function to generate basic adversarial examples
def generate_adversarial(net, n, steps, eta):
    net.eval()
    data, target = test_data[n]
    original = data.clone()
    data = data.unsqueeze(0)  # Add batch dimension
    data.requires_grad = True

    for step in range(steps):
        output = net(data)
        loss = F.nll_loss(output, torch.LongTensor([target]))
        net.zero_grad()
        loss.backward()
        # Increase the effect of each step
        data.data += eta * torch.sign(data.grad.data)  # Use sign of gradient
        data.data.clamp_(0, 1)

    return original, data.squeeze()

# Function to generate targeted adversarial examples
def generate_targeted_adversarial(net, image, target, steps, eta):
    net.eval()
    original = image.clone()
    image = image.unsqueeze(0)  # Add batch dimension
    image.requires_grad = True
    target_tensor = torch.tensor([target], dtype=torch.long)  # Target class tensor

    for step in range(steps):
        output = net(image)
        loss = F.nll_loss(output, target_tensor)
        net.zero_grad()
        loss.backward()
        # Adjust image to maximize the loss (move towards the target)
        image.data -= eta * torch.sign(image.grad.data)  # Move against the gradient
        image.data.clamp_(0, 1)

    return original, image.squeeze()

# Function to predict using the network and display images with softmax probabilities
def predict_and_display(model, original, adversarial):
    model.eval()
    with torch.no_grad():
        original_output = model(original.unsqueeze(0))
        original_prediction = original_output.argmax(dim=1, keepdim=True).item()
        original_prob = F.softmax(original_output, dim=1).squeeze()

        adversarial_output = model(adversarial.unsqueeze(0))
        adversarial_prediction = adversarial_output.argmax(dim=1, keepdim=True).item()
        adversarial_prob = F.softmax(adversarial_output, dim=1).squeeze()

        difference = torch.sum(torch.abs(original - adversarial))

        plt.figure(figsize=(14, 7))
        plt.subplot(2, 3, 1)
        plt.title(f'Original Image (Predicted: {original_prediction})')
        plt.imshow(original.squeeze().numpy(), cmap='Greys')
        plt.subplot(2, 3, 2)
        plt.title('Softmax Probabilities: Original')
        plt.bar(np.arange(10), original_prob.numpy())
        plt.ylim([0, 1])

        plt.subplot(2, 3, 4)
        plt.title(f'Adversarial Example (Predicted: {adversarial_prediction})')
        plt.imshow(adversarial.squeeze().numpy(), cmap='Greys')
        plt.subplot(2, 3, 5)
        plt.title('Softmax Probabilities: Adversarial')
        plt.bar(np.arange(10), adversarial_prob.numpy())
        plt.ylim([0, 1])

        plt.show()

        print(f"Difference in pixel values sum: {difference.item()}")

# Example usage of the functions
original, adversarial_example = generate_adversarial(net, 1, 50, 0.01)
predict_and_display(net, original, adversarial_example)

# Select an original image and target class different from the original class
original_image, _ = test_data[1]  # Assuming the original class is not 3
target_class = 3  # Target class to mislead the model
original, adversarial_example = generate_targeted_adversarial(net, original_image, target_class, 50, 0.01)
predict_and_display(net, original, adversarial_example)
