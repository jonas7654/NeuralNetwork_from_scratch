#include "matrix.h"

Matrix::Matrix(int rows, int cols) {
  assert(rows > 0 && cols > 0);
  n_rows = rows;
  n_cols = cols;
  int n_entries = rows * cols;

  _array = new Value*[n_entries];
  
  // Initialise Matrix with Zeros
  for (int i = 0; i < n_entries; i++) {
    _array[i] = new Value(0.0);
  }
}

Matrix::~Matrix() {
  for (int i = 0; i < (n_rows * n_cols) ; i++) {
    delete _array[i];
  }
  delete _array;
}

void Matrix::print() {
  for (int i = 0; i < n_rows; i++)
      {
          for (int j = 0; j < n_cols; j++)
          {
            std::cout << this->at(i, j)->getData() << " ";
          }
          std::cout << std::endl;
      }
}

Value* Matrix::at(int i, int j) {
  assert(i > -1 && i < n_rows && j > -1 && j < n_cols);
  return _array[i * n_cols + j];
}

int Matrix::num_cols() const {
  return n_cols;
}

int Matrix::num_rows() const {
  return n_rows;
}

Matrix* Matrix::operator +(const Matrix& other) {
  assert(n_cols == other.n_cols && n_rows == other.n_rows);
  
  Matrix* addedMatrix = new Matrix(n_rows, n_cols);

  for (int i = 0 ; i < n_rows; i++) {
    for (int j = 0; j < n_cols; j++) {
      (*addedMatrix->at(i, j)) = *(this->at(i, j)) + this->at(i, j);
    }
  }

  return addedMatrix;
}

Matrix* Matrix::operator*(Matrix& other) {
  assert(n_cols == other.n_rows);

  Matrix* mulMatrix = new Matrix(n_rows, other.n_cols);

  for (int k = 0; k < n_rows ; k++) {
    for (int i = 0; i < n_cols; i++) {
       double s = 0;
      for(int j = 0; j < n_cols; j++) {
        s += this->at(i, j) * (*other).at(j, i);
      }
      *(mulMatrix->at(k, i)) = s;
    }
  } 
}


int main() {
  Matrix test(2,2);

  test.print();

  *(test.at(0, 1))= 20.0;
  test.print();
}
