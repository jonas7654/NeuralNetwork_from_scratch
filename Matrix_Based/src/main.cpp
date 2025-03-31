#define private public

#include "../include/nn.h"
#include <cstddef>
#include <ostream>


// TODO:
// training loop
// check if mse_loss creates sufficient nodes for backpass
//
// TODO: Check if forward allocates a new matrix every time? IF YES DELETE IT OR DO IT INPLACE!
//

// ROWS CORRESPOND TO A SINGLE OBSERVATION
// COLUMNS CORRESPOND TO THE CONTEXT SIZE

#undef private

Matrix* mse_loss(Matrix *y_pred, Matrix *y_true, size_t batch_size, size_t output_size) {

  size_t total_elements = batch_size * output_size;
  
  Matrix* diff = *y_pred - y_true;
  Matrix* diff_squared = diff->square();

  double sum_squared_errors = cblas_dasum(total_elements, diff_squared->_data, 1);
  double mse = sum_squared_errors / total_elements;
  
  Matrix* loss = new Matrix(1, 1, false);
  loss->fill(mse);
  loss->childs.insert(diff_squared);
  
  loss->_backward = [loss, diff_squared, total_elements] () {
    const double scale = 1.0 / (total_elements);
    for (size_t i = 0; i < total_elements; i++) {
      diff_squared->_gradient[i] +=  scale * loss->_gradient[0];
    }
  };

  return loss;
}


int main() {



  // Train a AND gate
  bool use_one_hot = true;
  size_t batch_size = 4;
  size_t columns = 2;
  constexpr size_t number_of_layers = 3;
  size_t layers[number_of_layers] = {columns, 2, 2};
    
  nn neural_network(number_of_layers, layers, batch_size, use_one_hot);

  Matrix* x = new Matrix(batch_size, columns, false);
  Matrix* y = new Matrix(batch_size, 1, false);
  
  x->at(0, 0) = 0;
  x->at(1, 0) = 0;
  x->at(2, 0) = 1;
  x->at(3, 0) = 1;
  x->at(0, 1) = 0;
  x->at(1, 1) = 1;
  x->at(2, 1) = 0;
  x->at(3, 1) = 1;
  
  y->at(0, 0) = 0;
  y->at(1, 0) = 1;
  y->at(2, 0) = 1;
  y->at(3, 0) = 0; 
  
  std::cout << "Input data:" << std::endl;
  x->print();
  y->print();

  std::cout << "Starting Training" << std::endl;
  neural_network.train(x, y, batch_size, 1, 50000, true);

  neural_network.forward(x)->print();
  neural_network.predict(x);

  return 0;
}
