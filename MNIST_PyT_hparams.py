# Source: https://nextjournal.com/gkoehler/pytorch-mnist
## Changing variables when learning something can be risky!
import sys
import torch as th
import torchvision as tvn
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml
import sklearn.metrics as skm


def train(epoch):
    network.train() # training the network.
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad() # Set gradients to 0.
        #print("training data", data.shape)
        output = network(data) #runs the training data through the network.
        loss = th.nn.functional.nll_loss(output, target) # Computes the negative log likeihood loss function between the output and the target for the whole dataset.
        loss.backward() # Backpropagation: This calculates dJ/d[theta](Loss function derivative) for every parameter [theta] that needs a gradient. This will be added to the gradient of the parameter.
        optimizer.step() # Updates the value of the parameters with [theta] -= alpha*gradient of [theta]
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),
                                                                    100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append((batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
            th.save(network.state_dict(), '/home/xieruo1/Documents/hparam_optimization/model/model_opt.pth')
            th.save(optimizer.state_dict(), '/home/xieruo1/Documents/hparam_optimization/model/optimizer_opt.pth') #Unicode error: \U is a 8-char unicode escape. If followed by 's', it will be invalid.

def test():
    network.eval() # Tells that this is testing.
    test_loss = 0
    correct = 0
    with th.no_grad():
        for data, target in test_loader: # Loads the testset.
            #print("test data", data.shape)
            output = network(data) # Puts the testset into the trained network.
            test_loss += th.nn.functional.nll_loss(output, target, size_average=False).item() # Records the test loss function for the whole dataset.
            pred = output.data.max(1, keepdim=True)[1] # The prediction is the number with the highest probability (.max)
            correct += pred.eq(target.data.view_as(pred)).sum() # Sums up the number of correct predictions from the 'pred' data.
    test_loss /= len(test_loader.dataset) # Gives the avg test loss.
    test_losses.append(test_loss) # Upload the avg loss.
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))




if __name__ == "__main__":

    print("JOB STARTED")
    if len(sys.argv)==2:
        calcdir=sys.argv[1]
        print("directory: %s"%(calcdir))
        os.chdir(calcdir)
    else:
        calcdir=os.getcwd()
        print("directory: %s"%(calcdir))
        os.chdir(calcdir)


    if os.path.exists("template_settings.yml"):
        user_settings=yaml.load(open("template_settings.yml","r"))

    hparams = {'n_chanels_output_layer1': int(user_settings['model']['n_chanels_output_layer1']),
               'n_chanels_output_layer2': int(user_settings['model']['n_chanels_output_layer2']),
               'filtersize_layer1': int(user_settings['model']['filtersize_layer1']),
               'filtersize_layer2': int(user_settings['model']['filtersize_layer2']),
               'number_of_neurons_dense_layer1': int(user_settings['model']['number_of_neurons_dense_layer1']),
               'number_of_neurons_dense_layer2': int(user_settings['model']['number_of_neurons_dense_layer2']),
               'momentum': user_settings['training']['momentum'],
               'learning_rate': user_settings['training']['learning_rate'],
               'batch_size_train': int(user_settings['training']['batch_size_train'])
               }
    #print(hparams)

    n_epochs = 3 # 3 epochs to run on.
    batch_size_train = hparams['batch_size_train'] # The training batch has a size of 64 images.
    batch_size_test = 1000 # There will be 1000 images in the training set.
    learning_rate = hparams['learning_rate'] # This is the alpha value of the neural network's stochastic gradient descent/GD algo. Look in the notebook for an explaination.
    momentum =  hparams['momentum'] # This is the momentum used to get ourselves out of local minimas.
    log_interval = 10 # Makes training times count by a factor of 10.

    rdm_seed = 1 # The random seed serves as the basis for a random number generator to randomly set the initial weights of the neural network.

    th.backends.cudnn.enabled = False # This is used for a NIVDIA GPU.
    th.manual_seed(rdm_seed) # Sets the seed for the random number generator so that the weights are constant at the beginning.

    train_loader = th.utils.data.DataLoader(tvn.datasets.MNIST('/files/', train = True, download = True,
                                                            transform = tvn.transforms.Compose([tvn.transforms.ToTensor(),
                                                            tvn.transforms.Normalize((0.1307,), (0.3081,))])),batch_size=batch_size_train, shuffle=True)

    test_loader = th.utils.data.DataLoader(tvn.datasets.MNIST('/files/', train = False, download = True,
                                                            transform = tvn.transforms.Compose([tvn.transforms.ToTensor(),
                                                            tvn.transforms.Normalize((0.1307,), (0.3081,))])),batch_size=batch_size_test, shuffle=True)

    ex_batch = enumerate(test_loader) # Numbers all the images in the testing batch in chronological order.
    batch_idx, (ex_data, ex_targets) = next(ex_batch) # ex_data: the example pictures, ex_targets: the actual number, batch_idx: the batch number assigned to a group of images.
    ex_data.shape

    fig = plt.figure()
    for i in range(0,6):
        plt.subplot(2,3, i+1)
        plt.tight_layout()
        plt.imshow(ex_data[i][0], cmap = 'gray', interpolation = 'none')
        plt.title("Ground Truth: {}".format(ex_targets[i]))
        plt.xticks([])
        plt.yticks([])

    class NN(th.nn.Module):
        def __init__(self, hparams):
            super(NN, self).__init__()
            self.conv1 = th.nn.Conv2d(1, hparams['n_chanels_output_layer1'], kernel_size = hparams['filtersize_layer1']) # Sets 10 random filters 5*5 pxls.
            self.conv2 = th.nn.Conv2d(hparams['n_chanels_output_layer1'], hparams['n_chanels_output_layer2'], kernel_size = hparams['filtersize_layer2']) # Apply 10 more filters to 20 pxls of the same size.
            self.conv2_drop = th.nn.Dropout2d() # Spatial dropout. This completely shuts off certain filters.
            number_of_neurons = int(hparams['n_chanels_output_layer2'] * (int((31 - hparams['filtersize_layer1'] - 2*hparams['filtersize_layer2'])/4))**2)
            self.fc1 = th.nn.Linear(number_of_neurons, hparams['number_of_neurons_dense_layer1']) # Hidden layer with the 320 input channels and 50 nodes.
            self.fc2 = th.nn.Linear(hparams['number_of_neurons_dense_layer1'], hparams['number_of_neurons_dense_layer2']) # The hidden layer with 50 input channels and 10 nodes.

        def forward(self, x):
            x = th.nn.functional.relu(th.nn.functional.max_pool2d(self.conv1(x), 2)) # The input of the relu function is the max-pooled data with a kernel and stride of 2 at the first convolution.
            x = th.nn.functional.relu(th.nn.functional.max_pool2d(self.conv2_drop(self.conv2(x)),2)) # The input is the max-pooled data with a kernel and stride of 2 in the second convolution with some of filters switched off.
            shape = x.shape
            number_of_neurons= shape[1]*shape[2]*shape[3]
            x = x.view(-1,number_of_neurons) # Reshapes the data to 320 columns.
            x = th.nn.functional.relu(self.fc1(x)) # This is a hidden layer activates the 320 inputs with a ReLu function to create 50 outputs.
            x = th.nn.functional.dropout(x, training=self.training) # Drops some more filters.
            x = self.fc2(x) # This is another layer to get 10 outputs from the H.layer after the dropout.
            return th.nn.functional.log_softmax(x) # Applies the logarithmic function to the x output from the previous line after the x has gone through a softmax function.

    network = NN(hparams) # Identifies the neural network.
    optimizer = th.optim.SGD(network.parameters(), lr=learning_rate,momentum=momentum) # Stochastic Gradient Descent with a learning rate of 0.01, and momentum of 0.05.

    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]



    test()
    for epoch in range(1, n_epochs + 1): # Training with the training set and running the trained model with the test set.
      train(epoch)
      test()

    fig = plt.figure() # This prints out the loss vs training examples used.
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.savefig("loss_vs_epoch_opt.png")

    #Looking at the outputs
    with th.no_grad():
      output = network(ex_data)
      output_val = output.data.max(1)[1]
      targets = ex_targets

    returned = skm.confusion_matrix(targets, output_val).ravel()
    accuracy = skm.accuracy_score(targets, output_val)
    f1 = skm.f1_score(targets, output_val, average = 'micro')

    outfile=open("results.dat","w")
    outfile.write("%f\n"%(-1*f1))
    outfile.close()

    with open('COMPLETED', 'w') as content:
        content.write('exit code: 0')
