#include "../include/value_matrix.h"
#include <bits/types/FILE.h>
#include <string.h>
#include <string.h>
#include <cstdio>

#define BUFFER_SIZE 10000 // How to set this properly
#define IMAGE_SIZE 784
#define N_IMAGES 60000
#define FILE_PATH "../data/MNIST/mnist_train/mnist_train.csv"

Matrix* read_mnist() {
  FILE* file = fopen(FILE_PATH, "r");
  if(!file) {
    perror("error opening mnist");
    return 0;
  }

  // Allocate train matrix  
  // IMAGE_SIZE + 1 since I need to extract the label as well
  Matrix* mnist_train_matrix = new Matrix(N_IMAGES, IMAGE_SIZE + 1, false);
  size_t n_cols = mnist_train_matrix->n_cols;
  
  char line[BUFFER_SIZE];
  size_t count = 0;

  // populate the matrix
  while(fgets(line, BUFFER_SIZE, file) && count < N_IMAGES) {
    char* tok = strtok(line, ",");

    // Skip empty (just to be sure)
    if (!tok) {
      continue;
    }

    // Extract the label which is saved as the first value
    mnist_train_matrix->_data_at(count * n_cols + (IMAGE_SIZE)) = (double) (atof(tok));

    // extract each comma seperated value from a single row
    for (size_t i = 0; i < IMAGE_SIZE; i++){
      tok = strtok(nullptr, ",");
      if(!tok) break;
      
      mnist_train_matrix->_data_at(count * n_cols + i) = (double) (atof(tok));
    }
    count++;
  }

  fclose(file);

  return mnist_train_matrix;
}



