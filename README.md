# Handwritten-Digits-Classification

This is a neural network implementation of Handwritten digits classification.

The dataset is taken from MNIST library of handwritten digits.

The images are of size 28 x 28 = 784 pixels, grayscale value in [0, 255]

These values are scaled in range [0,1] for the network.

The network is structured as {784, 15, 10}

The weights are initialized uniformly using Glorot Randomization.

i.e each weight is uniformly distributed within [-e, +e], e = sqrt(6 / (L_in + L_out))

Sigmoid Function is used as the activation function for Neurons.

Log(x) function is used instead of Sum of Square Residuals to penalize errors more.

Gradient Descent is implemented with mini-batches (BATCH_SIZE >= 784)

Weight regularization is implemented to prevent overfitting.

Average Error on Unseen Examples = 9%
