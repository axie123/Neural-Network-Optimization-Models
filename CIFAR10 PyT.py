#defining the network

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


# Loading the data =========================================================================================================================================


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # This function transforms the image into a tensor, and normalizes it to (0.5 * 3,) mean and the same std.

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_load = torch.utils.data.DataLoader(train_set, batch_size = 80, shuffle = True, num_workers = 2)

test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_load = torch.utils.data.DataLoader(test_set, batch_size = 1000, shuffle = False, num_workers = 2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Displaying the image =====================================================================================================================================


dataiter = enumerate(test_load)
idx, (data, truth) = next(dataiter)

fig = plt.figure()
for i in range(0,6):
    img = torchvision.utils.make_grid(data[i])
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.subplot(2,3, i+1)
    plt.tight_layout()
    plt.imshow(np.transpose(npimg)) 
    plt.title("Ground Truth: {}".format(classes[labels[j]]))
plt.show()


# The neural network =======================================================================================================================================


class NN(nn.Module):

    def __init__(self):
        super(NN, self).__init__()
        # 3 input image channel, 6 output channels, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(3, 6, 5) # 1st convolution: 3 input channels for the images to come into, 6 filters each 5x5 pxls to produce outputs.
        self.conv2 = nn.Conv2d(6, 16, 5) # 2nd convolution: 6 input channels from the previous convolution layer, 16 filters each 5x5 pxls to produce outputs.
        self.pool = nn.MaxPool2d(2, 2) # Max pooling with a stride of 2 with a kernal size of 2.
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16*5*5, 120)  # Linear transformation with the 400 inputs per sample and 120 outputs per sample. This is a hidden layer with 120 nodes.
        self.fc2 = nn.Linear(120, 84) # 120 inputs per sample and 84 outputs per sample. This is the second hidden layer with 84 nodes.
        self.fc3 = nn.Linear(84, 10) # 84 inputs per sample and 10 outputs per sample. This the third hidden layer with 10 nodes.

    def forward(self, x):
        x_1 = self.pool(F.relu(self.conv1(x))) # ReLu function activates with the output from the first convolution to get max-pooled.
        x_2 = self.pool(F.relu(self.conv2(x_1))) # ReLu function activates with the output from the second convolution to get max-pooled.
        x_3 = x_2.view(-1, 16*5*5) # Reshapes the data to 450 columns.
        x_4 = F.relu(self.fc1(x_3)) # ReLu function activates each of the 120 nodes from the first hidden layer.
        x_5 = F.relu(self.fc2(x_4)) # ReLu function activates each of the 84 nodes from the second hidden layer.
        x_out = self.fc3(x) # 10 nodes give outputs from the third hidden layer.
        return [x_out,x_1,x_2,x_4,x_5]


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 50, 2)
        self.conv2 = nn.Conv2d(50, 40, 2)
        self.conv3 = nn.Conv2d(40, 30, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(3)
        self.bn2 = nn.BatchNorm2d(50)
        self.bn3 = nn.BatchNorm2d(40)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(30*6*6, 120)
        self.fc2 = nn.Linear(120, 90)
        self.fc3 = nn.Linear(90, 84)
        self.fc4 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.bn1(x)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.bn2(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.bn3(x)
        x = F.relu(self.conv3(x))
        x = x.view(-1, 30*6*6)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

net = NN()


# Training and Testing =====================================================================================================================================


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9) # Stochastic GD

def train(epoch):
    net.train()
    correct = 0
    total_training_loss = 0
    for batch_idx, (data, truth) in enumerate(train_load):
        optimizer.zero_grad()
        output = net(data)
        training_loss = criterion(output[0], truth)
        training_loss.backward()
        total_training_loss += training_loss.item()
        pred = output[0].data.max(1, keepdim=True)[1] 
        correct += pred.eq(truth.data.view_as(pred)).sum() 
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_load.dataset),
                                                                    100. * batch_idx / len(train_load), training_loss.item()))
    total_training_loss /= len(train_load.dataset)
    print('\nTraining set: Avg. loss: {:.4f}, Percentage: ({:.0f}%)\n'.format(total_training_loss, 100. * correct / len(train_load.dataset)))
    return [output, truth, 'training']
        
def test():
    net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, truths in test_load: 
            output = net(data) 
            test_loss += criterion(output[0], truth).item() 
            pred = output[0].data.max(1, keepdim=True)[1] 
            correct += pred.eq(truths.data.view_as(pred)).sum() 
    test_loss /= len(test_load.dataset) 
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_load.dataset),
            100. * correct / len(test_load.dataset)))
    return [output, truth, 'testing']


test()
for i in range(1,2):
    train(i)
    test()

'''
# Evaluating performance ===================================================================================================================================

dataiter = enumerate(test_load)
index, (images, labels) = next(dataiter)
fig = plt.figure()
for i in range(0,6):
    img = torchvision.utils.make_grid(images[i])
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.subplot(2,3, i+1)
    plt.tight_layout()
    plt.imshow(np.transpose(npimg,(1,2,0))) 
    plt.title("Ground Truth: {}".format(classes[labels[i]]))
plt.show()

outputs = net(images)
_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(6)))

# Predicts of NN on whole dataset ==========================================================================================================================

correct = 0
total = 0
with torch.no_grad(): # Sets all data to not require a gradient.
    for data in test_load:
        images, labels = data
        outputs = net(images) # Testing on the neural network with the testing set.
        _, predicted = torch.max(outputs.data, 1) # Receives the predictions by taking the option with the max predicted probability.
        total += labels.size(0) # Sum total.
        correct += (predicted == labels).sum().item() # Counts all the correct predicted images from the test set.

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

# Looking at the individual classes ========================================================================================================================


class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad(): # Sets all data to not require a gradient.
    for data in test_load:
        images, labels = data # Loads all the different types of images.
        outputs = net(images) # Runs through the trained neural net.
        _, predicted = torch.max(outputs, 1) # Makes predictions.
        c = (predicted == labels).squeeze() # This identifies all in the batch of 4 that match their right labels and removes all tensors with dim(1).
        for i in range(4): # 4 in a batch
            label = labels[i] # Get the label from the one of the four images.
            class_correct[label] += c[i].item() # Adds a 1 if it's correctly predicted and a 0 if not. (Images from the c batch)
            class_total[label] += 1 # Sets the total to 100.


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
'''

def layer_visuals(layer_num,data_results,ith):
    if data_results[2] == 'training':
        chosen_layer = data_results[0][layer_num]
        print(chosen_layer.shape)
        base = input('Base: \n')
        height = input('Height: \n')
        fig = plt.figure(figsize=(10,2))
        for i in range(ith,ith+5):
            image = chosen_layer[i].view(int(base),int(height))
            for j in range(5):
                plt.subplot(1,5,j + 1)
                plt.imshow(image.detach(), cmap = 'gray', interpolation = 'none')
                plt.xticks([])
                plt.yticks([])
        plt.show()
    elif data_results[2] == 'testing':
        chosen_layer = data_results[0][layer_num]
        print(chosen_layer.shape)
        base = input('Base: \n')
        height = input('Height: \n')
        fig = plt.figure(figsize=(10,2))
        for i in range(ith,ith+5):
            image = chosen_layer[i].view(int(base),int(height))
            for j in range(5):
                plt.subplot(1,5,j + 1)
                plt.imshow(image.detach(), cmap = 'gray', interpolation = 'none')
                plt.xticks([])
                plt.yticks([])
        plt.show()

def layer_loss_analysis(layer_num,data_results):
    if data_results[2] == 'training':
        chosen_layer = data_results[0][layer_num]
        layer = criterion(chosen_layer, dim=1)
        loss = criterion(layer, data_results[1])
        print('Layer Number: {}, Loss: {:.6f}\n'.format(layer_num, loss)) 
    if data_results[2] == 'testing':
        chosen_layer = data_results[0][layer_num]
        layer = criterion(chosen_layer, dim=1)
        loss = criterion(layer, data_results[1])
        print('Layer Number: {}, Loss: {:.6f}\n'.format(layer_num, loss))

def accuracy_individual_classes(classes,test_set):
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))
    for data in test_set:
        images, labels = data
        outputs = network(images)
        _, predicted = torch.max(outputs[0], 1)
        correct = (predicted == labels).squeeze()
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1
    for i in range(len(classes)):
        print('Accuracy of %s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

def saving_textfile(file,pandas_true):
    filename = input('Enter filename: \n')
    directory = input('Enter a directory: \n')
    if pandas_true == False:
        f = open(str(directory)+'\\'+str(filename),'w+')
        for line in file:
            f.writelines(str(list(line.numpy())))
        f.close()
    else:
        file.to_csv(str(directory)+'\\'+str(filename),'w+')

def weights_biases():
    parameters = {}
    for i in network.named_parameters():
        parameters[i[0]] = i[1] 
    specific_parameters = parameters.keys()
    while(True):
        print('The weights and biases of these layers have been identified: \n')
        for j in specific_parameters:
            print(j)
        print('\n')
        wanted_parameter = input('Please enter the wanted parameter or enter 0 to exit. Press E to export a specific parameter. \n')
        print('\n')
        if wanted_parameter == '0':
            break
        elif wanted_parameter == 'E' or wanted_parameter == 'e':
            wanted_parameter = input('Please enter the parameter to export: \n')
            data = parameters[str(wanted_parameter)].detach()
            saving_textfile(data,False)
            break
        else:
            ith_node = input('Enter the node number: \n')
            while(True):
                ith_weight_ith_node, end = input('Enter the input weights: \n').split()
                if end == 'x':
                    break
                else:
                    print('\n')
                    print(parameters[wanted_parameter][int(ith_node)][int(ith_weight_ith_node):int(end)].detach())
                    print('\n')
    print('Closed.')

