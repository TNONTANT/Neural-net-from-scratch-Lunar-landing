# Neural-net-from-scratch-Lunar-landing
This project builds a neural network model from scratch to make autonomous gameplay in the Landing Rocket game, without external libraries, to gain a deep understanding of deep learning concepts. Collected data is used, and essential components like feedforward and backpropagation are implemented. It enhances comprehension of the underlying principles behind deep learning.

** note: This project not allow to use pandas or numpy to create the algorithm but can be use to mangange the file

`util.py` Contain all function that use to find the best parameter of NeuralNetwork model including
- pre_processing: to preprocess the data
- NeuralNetwork: to do feedforward and back propergation
- TestNeural: to test performance finding best parameter (avoid over fitting)
- Node: function to find the best fit number of neuron in hidden layer.
- LearningRate: funtion to find the best learning rate.
- Momentum: function to find the best momentum.

`implement_train_test.py`
- this function will train model and get best parameter for the model (save weight to w_hid.csv, w_out.csv)
- w_hid.csv: contain weight of hidden layer
- w_out.csv: contain weight of output layer

`NeuralNetHolder.py`
- Foward propergation, this file will work with others game logic file in gamecode repo to make game play automatically.

all graph figure of this project are in `individual_presentation.pdf`

Result below

https://github.com/TNONTANT/Neural-net-from-scratch-Lunar-landing/assets/103983840/015e3567-028b-498a-8e0a-38474bdc7d32

