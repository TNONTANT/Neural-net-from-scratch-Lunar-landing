# Neural-net-from-scratch-Lunar-landing
This project builds a neural network model from scratch, without external libraries, to gain a deep understanding of deep learning concepts. Collected data is used, and essential components like feedforward and backpropagation are implemented. It enhances comprehension of the underlying principles behind deep learning.

** note: This project not allow to use pandas or numpy to create the algorithm but can be use to mangange the file

util.py Contain all function that use to find the best parameter of NeuralNetwork model including
- pre_processing: to preprocess the data
- NeuralNetwork: to do feedforward and back propergation
- TestNeural: to test performance finding best parameter (avoid over fitting)
- Node: function to find the best fit number of neuron in hidden layer.
- LearningRate: funtion to find the best learning rate.
- Momentum: function to find the best momentum.