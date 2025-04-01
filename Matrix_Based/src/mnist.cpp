#include "../include/nn.h"
#include "../include/mnist_parser.h"
#include <cstddef>
#include <typeinfo>

#define OUTPUT_SIZE 10
#define IMAGE_SIZE 784
#define N_IMAGES 60000

// :TODO add a slice method to the matrix class
// implement: Xavier/Glorot initialization
// Pixel normaliziation within parser?
// Look at batch extraction methods which overwrites pointer to the internal array (Might be a very dirty solution)
//

int main() 
{
  // config
  constexpr size_t number_of_layers = 4;
  constexpr size_t layer_config[number_of_layers] = {IMAGE_SIZE, 16, 16, OUTPUT_SIZE};
  const size_t batch_size = 32;
  constexpr bool use_one_hot = true;
  double lr = 0.001;
  double epochs = 50;
  bool verbose = true;

  // Initialize Neural Network
  nn mlp(number_of_layers, layer_config, batch_size, use_one_hot);

  // Read the MNIST dataset
  Matrix* mnist_data = read_mnist("train"); 
  // Divide into label and input data
  Matrix* true_lables = mnist_data->select_col(IMAGE_SIZE);
  Matrix* x_data = mnist_data->slice(0, N_IMAGES - 1, 0, IMAGE_SIZE - 1); // select the first 784 cols
  delete mnist_data;

  mlp.train(x_data, true_lables, lr, epochs, verbose);

  // Test data
  Matrix* mnist_data_test = read_mnist("test");
  // Divide into label and input data
  Matrix* true_lables_test = mnist_data_test->select_col(IMAGE_SIZE);
  Matrix* x_data_test = mnist_data_test->slice(0, N_IMAGES - 1, 0, IMAGE_SIZE - 1); // select the first 784 cols
  delete mnist_data_test;
  
  mlp.predict(x_data_test->slice(0, 11, 0, IMAGE_SIZE - 1));

  for (size_t i = 0; i < 10; i++) {
    std::cout << "True Label: " << true_lables_test->at(i, 0) << std::endl;
  }
  
  return 0;
}
