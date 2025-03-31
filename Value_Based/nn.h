#ifndef NN_H
#define NN_H

#include "value.h"
#include "matrix.h"
#include <iostream>
#include <cassert>
#include <random>
#include <stdexcept>
#include <string>

class Neuron {
private:
  Value** weights;
  Value* bias;
  int n_weights;
  
public:
  Neuron(int in_weights);
  ~Neuron();
  // Define a functor to call the Neuron
  Value* forward(Value** x, bool isOutputLayer);
  void printWeights() const;
  void update(double lr);
};


class Layer{
private:
  Neuron** neurons;
  int n_neurons;
public:
  Layer(int n_neurons, int n_inputs);
  Value** forward(Value** x, bool isOutputLayer);
  void update(double lr);
  void printLayer() const;
  int getNumNeurons() const;
};


//////////////////////////////////////////


class nn {
private:
  Layer** layers;
  int total_layers;
  int output_size;
  int input_size;
  int number_of_layers;

  Value* cost(Value* y, Value* ypred){
    Value* diff = *y - ypred;
    Value* result = *diff * diff;
    return result;
  }

public:
  nn(int* layer_sizes, int number_of_layers);
  Value** forward(Value** x);
  void update(double lr);
  void train(Value** input, double lr , int epochs, int n_cols, int n_rows);
  void printNN() const;
};


#endif // !NN_H
