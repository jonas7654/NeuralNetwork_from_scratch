# mlp.cpp

- This repository hosts a multilayer perceptron (MLP) implementation in C++.
- I created this project for my own educational purposes only.
- Matrix Based uses openBLAS to optimize Matrix operations to speed up computation.
- Backpropagation is done by creating a Computational Graph either based on Value Instances or Matrix Instances.
- An MNIST example can be found in the Matrix Based folder. The mnist data must be downloaded seperately (.csv).
  - `./builds/mnist_training`

# Value Based and Matrix Based computational Graph
## Value Based
  - Value based creates an "Value" Object for each value in the MLP.
  - Each Instance holds a _backward function in order to backpropagate throught the network via autodiff.
## Matrix Based
  - Matrix Based only creates Matrix nodes within the computational Graph and backpropagates through the network via autodiff.
  - Each Matrix Node holds a _backward function
