#include "../include/value_matrix.h"
#include <bits/types/FILE.h>
#include <string.h>
#include <string.h>
#include <cstdio>

#define BUFFER_SIZE 10000 // How to set this properly
#define IMAGE_SIZE 784
#define N_IMAGES 60000

Matrix* read_mnist() {
  FILE* file = fopen("../../data/MNIST/mnist_train.csv", "r");
  if(!file) {
    perror("error opening mnist");
    return 0;
  }

  // Allocate train matrix  
  Matrix* mnist_train_matrix = new Matrix(N_IMAGES, IMAGE_SIZE, false);
  size_t n_cols = mnist_train_matrix->n_cols;
  size_t n_rows = mnist_train_matrix->n_rows;

  char line[BUFFER_SIZE];
  size_t count = 0;

  while(fgets(line, BUFFER_SIZE, file) && count < N_IMAGES) {
    char* tok = strtok(line, ",");
    if (!tok) {
      continue;
    }
    
    for (size_t i; i < IMAGE_SIZE; i++){
      tok = strtok(NULL, ",");
      if(!tok) {break;}
      mnist_train_matrix->_data_at(count * n_cols + i) = (double) (atoi(tok));
    }
    count++;
  }

  fclose(file);

  return mnist_train_matrix;
}


int main() {
  Matrix* test = read_mnist();

  for(int j = 0; j < IMAGE_SIZE; j++) {
    std::cout << test->at(0, j) << " ";
  }

  
  return 0;
}

