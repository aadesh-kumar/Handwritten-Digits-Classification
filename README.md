# Handwritten-Digits-Classification

This is a neural network implementation of Handwritten digits classification.
The network is structured as {784, 35, 10}
The weights are initialized uniformly using Glorot Initialization.
i.e each weight must be uniformly distributed within [-e, +e],
where e = $\sqrt{6 / (L_in + L_out)}$

Cost Function := Log() function with sigmoid activation function

Gradient Descent is implemented with mini-batches and weight regularization

