####### utils.py module comes from https://medium.com/ai-society/gans-from-scratch-1-a-deep-introduction-with-code-in-pytorch-and-tensorflow-cb03cdcdba0f

## I modified the code to make myself understand it.

import torch as th
import torchvision as tvn
from torch import nn as nn
from torch import optim as optim
from torch.autograd.variable import Variable
import torchvision
from torchvision import transforms as transforms
from torchvision import datasets as datasets
from utils import Logger

# Normalizing and loading the MNIST data from the training set:

data_loader = th.utils.data.DataLoader(tvn.datasets.MNIST('/files/', train = True, download = True,
                                                        transform = tvn.transforms.Compose([tvn.transforms.ToTensor(),
                                                        tvn.transforms.Normalize((0.1307,), (0.3081,))])),batch_size=100, shuffle=True)

# Num batches takes the number of batches based on the 100 batch size.

num_batches = len(data_loader)

# Creating the discriminator with 3 hidden layers:

class DiscriminatorNet(nn.Module):

    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        n_features = 784 # Each image is 28 * 28 pixels.
        n_out = 1
        self.hidden0 = nn.Sequential(nn.Linear(n_features, 1024),nn.LeakyReLU(0.2),nn.Dropout(0.3)) # First Layer: 1024 nodes. The nodes have a Leaky ReLu with a 0.3 dropout.
        self.hidden1 = nn.Sequential(nn.Linear(1024, 512),nn.LeakyReLU(0.2),nn.Dropout(0.3)) # Second layer: 512 nodes. The nodes have a Leaky ReLu with a 0.3 dropout.
        self.hidden2 = nn.Sequential(nn.Linear(512, 256),nn.LeakyReLU(0.2),nn.Dropout(0.3)) # Third layer: 256 nodes. The nodes have a Leaky ReLu with a 0.3 dropout.
        self.out = nn.Sequential(nn.Linear(256, n_out), nn.Sigmoid()) # Last layer. This node has a sigmoid function to modify the output.

    def forward(self, x):
        x = self.hidden0(x) # Runs through forward prop.
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x

discriminator = DiscriminatorNet()

# Creating the generator net:

class GeneratorNet(nn.Module):

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

# Converting images to vectors and vice versa for fitting into the NNs:

def images_to_vectors(images): # Changes images to colour based 784*1 vector.
    return images.view(images.size(0), 784)

def vectors_to_images(vectors):  # Changes colour based 784*1 vectors to images.
    return vectors.view(vectors.size(0), 1, 28, 28)

# This is to create the fake data with noise.

def noise(size):
    n = th.randn(size, 100)
    return n

num_test_samples = 16

# Noise generator to create the fake samples. The noise will be added to 

def noise(size):
    n = th.randn(size, 100)
    return n

# Optimization:

d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002) # Implements Adam alg for SGD and sets default parameters.
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002) # Implements Adam alg for SGD.

loss = nn.BCELoss() # Loss function as binary cross entropy loss.

def ones_target(size):
    data = th.ones(size, 1)
    return data

def zeros_target(size):
    data = th.zeros(size, 1)
    return data

# Training the discriminator and generator networks:

def train_discriminator(real_data, fake_data):
    N = real_data.size(0)
    d_optimizer.zero_grad() # Reset gradients to zero
    
    output_real = discriminator(real_data)
    loss_real = loss(output_real, ones_target(N)) # Calculate error
    loss_real.backward() # backpropagation
    
    output_fake = discriminator(fake_data)
    loss_fake = loss(output_fake, zeros_target(N)) # Calculate error
    loss_fake.backward() # backpropagation
    
    d_optimizer.step() # Update weights with gradients
    return loss_real + loss_fake, output_real, output_fake

def train_generator(fake_data):

    N = fake_data.size(0)
    g_optimizer.zero_grad() # Reset gradients to zero
    
    output = discriminator(fake_data) # Sample noise and generate fake data
    final_loss = loss(output, ones_target(N)) # Calculate error
    final_loss.backward() # backpropagate
    
    g_optimizer.step() # Update weights with gradients using GD equation.
    return final_loss # Return error

# Create logger instance to record the data in the computer hard drive. 

logger = Logger(model_name='VGAN', data_name='MNIST')

# Total number of epochs to train

num_epochs = 200

for epoch in range(num_epochs):
    for n_batch, (real_batch,_) in enumerate(data_loader):
        N = real_batch.size(0) # The size of the real batch as in the number of images per batch.
        
        # Train Discriminator
        real_data = images_to_vectors(real_batch) # turns the 100 images per batch and reshapes them.
        fake_data = generator(noise(N)).detach()  # Generate fake data from noise and detach (so gradients are not calculated for generator).
        d_loss, d_output_real, d_output_fake = train_discriminator(real_data, fake_data)  
        
        # Train Generator
        fake_data = generator(noise(N)) # Generate fake data
        g_loss = train_generator(fake_data)
        
        # Log batch error
        logger.log(d_loss, g_loss, epoch, n_batch, num_batches)
        
        # Display Progress every few batches
        if (n_batch) % 100 == 0:
            test_images = vectors_to_images(generator(noise(num_test_samples))).data # Get 16 images for the test set and tries them out.
            logger.log_images(test_images, num_test_samples, epoch, n_batch, num_batches) # Saves the images on the hard drive.
            logger.display_status(epoch, num_epochs, n_batch, num_batches, d_loss, g_loss, d_output_real, d_output_fake) # Display status Logs
