#include "../include/nn.h"

nn::nn(size_t number_of_layers, size_t layer_sizes[], size_t  batch_size, bool use_one_hot) {
    num_layers = number_of_layers - 1; // Exclude input layer
    this->context_size = layer_sizes[0]; // store the context_size for better code readabillity
    this->output_size = layer_sizes[num_layers];
    this->batch_size = batch_size;
    this->use_one_hot = use_one_hot;

    this->layer_weights = new Matrix*[num_layers]; 
    this->layer_biases = new Matrix*[num_layers];
    for (size_t i = 1; i < number_of_layers ; i++) {
      // Allocate weights 
      Matrix* weights = new Matrix(layer_sizes[i - 1], layer_sizes[i], true);
      layer_weights[i - 1] = weights;

      Matrix* biases = new Matrix(layer_sizes[i], 1, true); // TODO: Dimensions
      layer_biases[i - 1] = biases;
    }
}

nn::~nn() {
  for (size_t i = 0; i < num_layers; i++) {
    delete layer_weights[i];
    delete layer_biases[i];
  }
  delete[] layer_weights;
  delete[] layer_biases;
}


void nn::print() const {
    for (size_t i = 0; i < num_layers; i++) {
        if (i == num_layers - 1) {
          std::cout << "Output layer: \n";
          std::cout << "Weights: \n";
          layer_weights[i]->print();
          std::cout << "bias \n";
          layer_biases[i]->print();
          continue;
        }

      std::cout << "hidden Layer " << i + 1 << std::endl;
        std::cout << "Weights: \n";
        layer_weights[i]->print(); // Print weights for layer i
        std::cout << "bias \n";
        layer_biases[i]->print();
        std::cout << std::endl;
    }
    std::cout << std::endl;
} 
Matrix* nn::forward(Matrix* input) {
  assert(input->n_cols == layer_weights[0]->n_rows);

  for(size_t i = 0; i < num_layers; i++) {
    input = *input * layer_weights[i];
    input = input->add_bias(layer_biases[i]);

    if (i < num_layers - 1) {
      input = input->sigmoid();
    } 
    else {
      if (use_one_hot) {
        input = input->softmax();
      }
      else {
        input = input->sigmoid();
      }
    }
  }
  return input;
}

void nn::update(double& lr) {
  for (int i = num_layers - 1; i >= 0; i--) {

    layer_weights[i]->gradDescent(lr);
    layer_biases[i]->gradDescent(lr);

    layer_biases[i]->zeroGrad();
    layer_weights[i]->zeroGrad(); // Reset gradients
  }
}

Matrix* nn::mse_loss(Matrix *y_pred, Matrix *y_true) {
  if (use_one_hot) {
    assert(y_true->n_cols == output_size);
    assert(y_true->n_rows == batch_size);
  }

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

void nn::train(Matrix *x, Matrix *y, int batch_size, double lr, double epochs, bool verbose) {
  Matrix* output;
  Matrix* cost;
  // Do not delete the input data in DeleteGraph
  x->isPersistent = true;
  if (use_one_hot) {
      y = one_hot(y);
      this->use_one_hot = true;
    }
  y->isPersistent = true;
  for(size_t e = 0; e < epochs; e++) {
    // Forward pass
    output = forward(x); 
    cost = mse_loss(output, y);
    
    // Backpropagation
    cost->backward();
    update(lr);
    

    if (verbose) {
      std::cout << "Epoch " << e << ", Loss: " << cost->at(0, 0) << std::endl;
    }

    cost->resetVisited();
    cost->deleteGraph();
  }

  x->isPersistent = false;
  y->isPersistent = false;
}

Matrix* nn::one_hot(Matrix* x) {
  // I assume that data has dimensions B, 1 for categorical data.
  // => for each batch there is one right indice over the output_size
  assert(x->n_cols == 1);
  assert(x->n_rows == batch_size);
  // Note that a Matrix is initialised with zeros
  Matrix* one_hot_matrix = new Matrix(batch_size, output_size, false);
  
  for (size_t i = 0; i < batch_size; i++) {
    double non_zero_entry = static_cast<int>(x->at(i,0)); 
    one_hot_matrix->at(i, non_zero_entry) = 1.0;
  }

  return one_hot_matrix;
}


void nn::predict(Matrix *input) {
  assert(input->n_cols == layer_weights[0]->n_rows);
  size_t* predicted_index = new size_t[batch_size];

  Matrix* softmax_output = forward(input);

  #pragma omp parallel for
  for (size_t row = 0; row < batch_size; row++) {
    double highest_prob = 0;
    for (size_t col = 0 ; col < output_size; col++) {
      if (softmax_output->_data[row * output_size + col] > highest_prob) {
        highest_prob = softmax_output->_data[row * output_size + col];
        predicted_index[row] = col;
      }
    }
  }

  // For now just print the predictions. Need to address this later.
  for (size_t i = 0; i < batch_size; i++) {
    std::cout << predicted_index[i] << std::endl;
  }

  delete[] predicted_index;
  softmax_output->deleteGraph();
}
