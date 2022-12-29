# Neural_Network_Project
This project was originally created for my machine learning class, its use should be for educational purposes only, as it is not properly tested or designed with any other use in mind. 

The project contains 3 convolutional neural networks that can be trained using an Adam Update method for the task of image denoising. 2 of these networks were UNets and for one reason or another do not train well, however the simple CNN using only a single hidden layer works rather well. 

The TestFinalNetwork.jl can be used to test a pretrained network, while the TrainSmallNetwork.jl can be used to train a small network from scratch.

The project is intended to be used with the cifar-100-binary for its image patches which get augmented with randomly generated noise for the generation of training data. The binary files can be inserted into the cifar-100-binary folder.


 @misc{krizhevsky_2009, title={Learning Multiple Layers of Features from Tiny Images}, url={https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf?source=post_page---------------------------}, author={Krizhevsky, Alex}, year={2009}, month={Apr}} 
