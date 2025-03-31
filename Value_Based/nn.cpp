#include "nn.h"

typedef std::mt19937 rng_type;

Neuron::Neuron(int in_weights) {
    // Initialize random number generator
    std::random_device rd;  // Seed for the random number engine
    std::mt19937 gen(rd()); // Mersenne Twister engine
    std::uniform_real_distribution<double> dist(0.0, 1.0); // Distribution for random doubles

    // Allocate memory for weights
    this->n_weights = in_weights;
    this->weights = new Value*[in_weights]; // Allocate memory for the array
    // Initialize weights with random values
    for (int i = 0; i < n_weights; i++) {
        this->weights[i] = new Value(dist(gen), true); // Generate a random Value() instance
    }

  this->bias = new Value(dist(gen));
}

Neuron::~Neuron() {
        // Delete each weight
        for (int i = 0; i < n_weights; i++) {
            delete weights[i]; // Delete individual weights
        }
        delete[] weights; // Delete the weights array
        delete bias;      // Delete bias
    }

Value* Neuron::forward(Value** x, bool isOutputLayer) {
 // int size_x = sizeof(x);
  // assert(size_x == n_weights);
  
  Value* res = new Value(0);
  for (int i = 0; i < n_weights; i++) {
    *res += *x[i] * weights[i];
  }
  *res += bias;
  // use sigmoid activation
  if (!isOutputLayer){
    res = res->sigmoid();
  }
  return res;
}

void Neuron::update(double lr = 0.01) {
  for (int i = 0 ; i < n_weights; i++) {
    weights[i]->getData() -= lr * weights[i]->getGradient();
  }
  bias->getData() -= lr * bias->getGradient();
}

void Neuron::printWeights() const {
  for(int i = 0; i < n_weights ; i++){
    std::cout << "w" << std::to_string(i) << ": "; weights[i]->printValue();
  }
}



////////////////////////////////////////////

Layer::Layer(int n_neurons, int n_input) {
  this->n_neurons = n_neurons;
  this->neurons = new Neuron*[n_neurons];
  // pouplate Neurons with random weights
  for (int i = 0; i < n_neurons; i++) {
    neurons[i] = new Neuron(n_input);
  }
}

Value** Layer::forward(Value** x, bool isOutputLayer) {
  Value** output = new Value*[n_neurons];

  for (int i = 0; i < n_neurons; i++) {
    output[i] = neurons[i]->forward(x, isOutputLayer);
  }
  return output;
}

void Layer::update(double lr) {
  for (int i = 0; i < n_neurons; i++) {
    this->neurons[i]->update(lr);
  }
}

int Layer::getNumNeurons() const {
  return n_neurons;
}

void Layer::printLayer() const { std::cout << "Number of Neurons: " << n_neurons << std::endl; }

///////////////////
nn::nn(int* layer_sizes, int number_of_layers) {
  this->input_size = layer_sizes[0];
  this->output_size = layer_sizes[number_of_layers - 1];
  this->number_of_layers = number_of_layers - 1; // Exclude the input as its not a "real" Layer
  this->layers = new Layer*[this->number_of_layers];
  // Init first layers
  for (int i = 0; i < number_of_layers; i++) {
    this->layers[i] = new Layer(layer_sizes[i + 1], layer_sizes[i]);
  }
}

Value** nn::forward(Value** x) {
  bool isOutputLayer = false;
  for (int i = 0; i < number_of_layers; i++) {
    if (i == number_of_layers - 1) {
      isOutputLayer = true;
    }
    x = layers[i]->forward(x, isOutputLayer);
  }
  return x;
}

void nn::update(double lr) {
  for (int i = number_of_layers; i >= 0 ; i--) {
    layers[i]->update(lr);
  }
}



