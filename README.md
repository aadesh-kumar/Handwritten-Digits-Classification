# Handwritten-Digits-Classification

## Introduction

This is a neural network implementation of Handwritten digits classification. The dataset is taken from MNIST library of handwritten digits. There are 60000 training and testing samples. The images are of size 28 x 28 = 784 pixels, grayscale value in range 0 - 255.

## Data Normalization

The image cells are mean normalized in range [-1,1] for prevention of Exploding Gradient.

## Neural Network

The network is structured as $(784, 15, 10)$ The weights are initialized uniformly using Glorot Randomization i.e each weight is uniformly distributed within $[-e, +e]$ $$e = \sqrt{6 \over (L_in + L_out)}$$. Weight regularization is implemented to prevent overfitting. Gradient Descent is implemented with mini-batches (BATCH_SIZE >= 784)

## Activation Function

Sigmoid Function is used as the activation function for Neurons.

## Error function

Log(x) function is used instead of Sum of Square Residuals to penalize errors more.

## Result

Average Error on Unseen Examples = 9%
