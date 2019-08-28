# https://medium.com/ai-society/gans-from-scratch-1-a-deep-introduction-with-code-in-pytorch-and-tensorflow-cb03cdcdba0f

import torch
from torch import nn as nn
from torch import optim as optim
from torch.autograd.variable import Variable
import torchvision
from torchvision import transforms as transforms
from torchvision import datasets as datasets
from utils import Logger

# Normalizing and loading the MNIST data:

def mnist_data():
    compose = transforms.Compose([transforms.ToTensor(), transforms.Normalize((.5, ), (.5, ))]) # Transform that normalizes to mean and std of 0.5
    out_dir = './dataset'
    return datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)

data = mnist_data()

data_loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)
# Num batches

num_batches = len(data_loader)

# Creating the discriminator with 3 hidden layers:

class DiscriminatorNet(torch.nn.Module):
    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        n_features = 784 # Each image is 28 * 28 pixels.
        n_out = 1
        self.hidden0 = nn.Sequential( nn.Linear(n_features, 1024),nn.LeakyReLU(0.2),nn.Dropout(0.3)) # First Layer: 1024 nodes. The nodes have a Leaky ReLu with a 0.3 dropout.
        self.hidden1 = nn.Sequential( nn.Linear(1024, 512),nn.LeakyReLU(0.2),nn.Dropout(0.3)) # Second layer: 512 nodes. The nodes have a Leaky ReLu with a 0.3 dropout.
        self.hidden2 = nn.Sequential( nn.Linear(512, 256),nn.LeakyReLU(0.2),nn.Dropout(0.3)) # Third layer: 256 nodes. The nodes have a Leaky ReLu with a 0.3 dropout.
        self.out = nn.Sequential(torch.nn.Linear(256, n_out), torch.nn.Sigmoid()) # Last layer. This node has a sigmoid function to modify the output.

    def forward(self, x):
        x = self.hidden0(x) # Runs through forward prop.
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x

discriminator = DiscriminatorNet()

# Converting images to vectors and vice versa:

def images_to_vectors(images): # Changes images to colour based 784*1 vector.
    return images.view(images.size(0), 784)

def vectors_to_images(vectors):  # Changes colour based 784*1 vectors to images.
    return vectors.view(vectors.size(0), 1, 28, 28)

# Creating the generator net:

class GeneratorNet(torch.nn.Module):
    def __init__(self):
        super(GeneratorNet, self).__init__()
        n_features = 100 #The initial number of numbers used to produce the artifical images in the generator.
        n_out = 784 # The size of the output.
        self.hidden0 = nn.Sequential(nn.Linear(n_features, 256), nn.LeakyReLU(0.2)) # First Layer: 256 nodes. The nodes have a Leaky ReLu.
        self.hidden1 = nn.Sequential(nn.Linear(256, 512), nn.LeakyReLU(0.2)) # Second Layer: 512 nodes. The nodes have a Leaky ReLu.
        self.hidden2 = nn.Sequential(nn.Linear(512, 1024), nn.LeakyReLU(0.2)) # Third Layer: 1024 nodes. The nodes have a Leaky ReLu.
        self.out = nn.Sequential(nn.Linear(1024, n_out), nn.Tanh()) # The output is created through the activation function of tanh from the previous layer.

    def forward(self, x):
        x = self.hidden0(x) # Forward prop.
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x

generator = GeneratorNet()

# Noise generator:

def noise(size):
    n = Variable(torch.randn(size, 100))
    return n

# Optimization:

d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002) # Implements Adam alg for SGD.
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002) # Implements Adam alg for SGD.

loss = nn.BCELoss()

def ones_target(size):
    data = Variable(torch.ones(size, 1))
    return data

def zeros_target(size):
    data = Variable(torch.zeros(size, 1))
    return data

# Training the discriminator and generator networks:

def train_discriminator(optimizer, real_data, fake_data):
    N = real_data.size(0)
    optimizer.zero_grad() # Reset gradients to zero

    # 1.1 Train on Real Data
    prediction_real = discriminator(real_data)
    error_real = loss(prediction_real, ones_target(N)) # Calculate error
    error_real.backward() # backpropagation

    # 1.2 Train on Fake Data
    prediction_fake = discriminator(fake_data)
    error_fake = loss(prediction_fake, zeros_target(N)) # Calculate error
    error_fake.backward() # backpropagation

    # 1.3 Update weights with gradients
    optimizer.step()

    # Return error and predictions for real and fake inputs
    return error_real + error_fake, prediction_real, prediction_fake

def train_generator(optimizer, fake_data):
    N = fake_data.size(0)
    optimizer.zero_grad() # Reset gradients to zero
    prediction = discriminator(fake_data) # Sample noise and generate fake data
    error = loss(prediction, ones_target(N)) # Calculate error
    error.backward() # backpropagate
    optimizer.step() # Update weights with gradients using GD equation.
    return error # Return error

num_test_samples = 16
test_noise = noise(num_test_samples)

# Create logger instance
logger = Logger(model_name='VGAN', data_name='MNIST')
# Total number of epochs to train
num_epochs = 200

for epoch in range(num_epochs):
    for n_batch, (real_batch,_) in enumerate(data_loader):
        N = real_batch.size(0) # The size of the real batch as a 1-d value.
        # 1. Train Discriminator
        real_data = Variable(images_to_vectors(real_batch))
        fake_data = generator(noise(N)).detach()  # Generate fake data and detach (so gradients are not calculated for generator).
        d_error, d_pred_real, d_pred_fake = \train_discriminator(d_optimizer, real_data, fake_data)  # Train Discriminator

        # 2. Train Generator
        # Generate fake data
        fake_data = generator(noise(N))
        # Train G
        g_error = train_generator(g_optimizer, fake_data)
        # Log batch error
        logger.log(d_error, g_error, epoch, n_batch, num_batches)
        # Display Progress every few batches
        if (n_batch) % 100 == 0:
            test_images = vectors_to_images(generator(test_noise))
            test_images = test_images.data
            logger.log_images(
                test_images, num_test_samples,
                epoch, n_batch, num_batches
            );
            # Display status Logs
            logger.display_status(
                epoch, num_epochs, n_batch, num_batches,
                d_error, g_error, d_pred_real, d_pred_fake
            )
