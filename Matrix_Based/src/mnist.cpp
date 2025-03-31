#include "../include/nn.h"
#include "../include/mnist_parser.h"
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
  constexpr size_t number_of_layers = 3;
  constexpr size_t layer_config[number_of_layers] = {IMAGE_SIZE, 16, OUTPUT_SIZE};
  const size_t batch_size = 32;
  constexpr bool use_one_hot = true;
  double lr = 0.001;
  double epochs = 50;
  bool verbose = true;

  // Initialize Neural Network
  nn mlp(number_of_layers, layer_config, batch_size, use_one_hot);

  // Read the MNIST dataset
  Matrix* mnist_data = read_mnist(); 
  // Divide into label and input data
  Matrix* true_lables = mnist_data->select_col(IMAGE_SIZE);
  mnist_data->n_cols = 784; // THIS IS JUST A HACK RIGHT NOW! : TODO
  

  // Create tiny test dataset
  Matrix* test_input = new Matrix(2, 784, false);
  Matrix* test_labels = new Matrix(2, 10, false);
  mlp.train(test_input, test_labels, lr, epochs, verbose);
  mlp.predict(test_input);
  
  return 0;
}
