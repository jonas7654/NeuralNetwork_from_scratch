#include "value.h"
#include "nn.h"
#include <iostream>


// This is a function which maps an array to a 2D-Matrix structure
Value* MAT_AT(Value** value_array, int n_cols, int i, int j) {
  return &(*value_array[n_cols * i + j]);
}

Value* cost(Value* y, Value* ypred){
  Value* diff = *y - ypred;
  Value* result = *diff * diff;
  return result;
}

int main() {
  // XOR Gate
  Value* train_data[12] = {
   new Value(0),new Value(0),new Value(0),
   new Value(0),new Value(1),new Value(1),
   new Value(1),new Value(0),new Value(1),
   new Value(1),new Value(1),new Value(0),
};

  Value* train_data2[10] = {
    new Value(1), new Value(2),
    new Value(2), new Value(4),
    new Value(3), new Value(6),
    new Value(4), new Value(7),
    new Value(5), new Value(10)
  };
  // TEST LAYER 
  int epochs = 5000;
  int n_cols = 2;
  int n_rows = 5;
  int layer_sizes[4] = {1, 4, 400, 1};
  int num_layers = sizeof(layer_sizes) / sizeof(layer_sizes[0]);

  nn nn(layer_sizes, num_layers);
  
  for (int e = 0; e < epochs; e++) {
    Value* c = new Value(0.0);
    for (int i = 0; i < n_rows; i++) {
        Value* y_true = MAT_AT(train_data2, n_cols, i, 1);
        Value* input[1] = {MAT_AT(train_data2, n_cols, i, 0)};

        Value** output = nn.forward(input);
        *c += cost(y_true, output[0]);
    }
    std::cout << "Cost " << c->getData() << std::endl;
    c->backward();
    nn.update(0.05);
    c->zeroGrad();
    c->deleteGraph();
  }

  // Check prediction out of sample
  Value* test_data[5] = {new Value(10), new Value(50), new Value(5), new Value(1), new Value(2)};
  for (int i = 0; i < 5; i++) {
      Value* input[1] = {test_data[i]};
      Value** output = nn.forward(input);
      std::cout << input[0]->getData() << "| " << output[0]->getData() << std::endl;
  }

  return 0;
}
