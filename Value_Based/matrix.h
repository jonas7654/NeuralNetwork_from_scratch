#ifndef MATRIX_H
#define MATRIX_H

#define MATRIX_H
#include "value.h"
#include <iostream>
#include <cassert>

// Value Matrix
class Matrix {
private:
  Value** _array;
  int n_rows;
  int n_cols;
public:
  Matrix(int rows, int cols);
  ~Matrix();
  int num_rows() const;
  int num_cols() const;

  void print();

  Value* at(int i , int j);


  Matrix* operator + (const Matrix& other);
  Matrix* operator * (Matrix& other);
  Matrix* operator - (const Matrix& other);

};

#endif
