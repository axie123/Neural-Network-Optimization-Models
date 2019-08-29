# Neural Network Optimization Models

This repository is a group of neural network models to be trained and optimized on the MNIST dataset. The models ideally would be transferred over to be used to predict chemical properties based on the given compound. This work is done under the supervision of Dr. Pascal Frederich.

The hyperparameter optimization method used is the black-box Bayesian Optimization. The general premises is that when template_job_manager.py is run, the program would start MNIST_PyT_hparams.py and train with the hyperparameters given in the template_settings.yml file. An F1 score would be produced at the end of the training, and the information will be sent back to template_job_manager.py for it to give a different set of hyperparameters to train with. Once it tested enough hyperparameters, the program should have an idea on what works best based on the priors and function evaluations. The models are mostly CNN models.

### For more information: 

Bayesian Optimization: https://www.cs.toronto.edu/~rgrosse/courses/csc411_f18/tutorials/tut8_adams_slides.pdf

Hyperparameters: https://machinelearningmastery.com/difference-between-a-parameter-and-a-hyperparameter/

Convolutional Neural Networks (CNN): https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53

### Other Branches:

**GAN** : This contains the GAN model that we created that was going to be used for generating new compounds to compare with old compounds.

**mnist-org** : This was the original MNIST CNN model. The stuff in the master branch evolved from it.

**cifar-10** : We did some work on the CIFAR-10 dataset, but not much.
