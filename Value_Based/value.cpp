#include <cmath>
#include <iostream>
#include <math.h>
#include "value.h"
#include <vector>
#include <algorithm>  // For std::find

Value::Value(double _data) {
  this->data = _data;
  this->gradient = 0;
  this->op = "None";
  this->isParameter = false;
};

Value::Value(double _data, bool isParameter) {
  this->data = _data;
  this->gradient = 0;
  this->op = "None";
  this->isParameter = isParameter;
};

Value::Value() {
  this->data = 0;
  this->gradient = 0;
  this->op = "None";
  this->isParameter = false;
}

Value::~Value() {}

// :TODO: Is is ok to specifically code this for the use of the Matrix class?
void Value::operator = (double& other) {
  this->data = other; 
}

double& Value::getData() {
  return this->data;
}

double Value::getGradient() const {
  return this->gradient;
}

void Value::printValue() const {
  std::cout << "Value(" << 
    "data: " << this->data << " " <<
    "gradient: " << this->gradient << " "
    << "operator: " << this->op << ")" <<
  std::endl;
}

void Value::printChilds() const {
  std::cout << "Childs: ";
  for (auto child : this->childs) {
    std::cout << child->data << " ";  // Print each child's data
  }
  std::cout << std::endl;  // Print a newline at the end
}

Value* Value::operator +(Value* other){
  Value* result= new Value(this->data + other->data);

  result->op = "+";
  result->childs.insert(this);
  result->childs.insert(other);
  result->_backward = [this, other, result]() {
        this->gradient += result->gradient;
        other->gradient += result->gradient;
    };
  return result;
}


Value* Value::operator -(Value* other) {
  Value* result = new Value(this->data - other->data);
  result->op = "-";
  result->childs.insert(this);
  result->childs.insert(other);
  result->_backward = [this, other, result](){
    this->gradient += result->gradient;
    other->gradient -= result->gradient;
  };

  return result;
}
Value* Value::operator *(Value* other){
  Value* result = new Value(this->data * other->data);
  result->op = "*";
  result->childs.insert(this);
  result->childs.insert(other);

  result->_backward = [this, other, result](){
    this->gradient += other->data * result->gradient;
    other->gradient += this->data * result->gradient;
  };
  return result;
}

Value* Value::sigmoid() {
  double sig = std::exp(this->data) / (1 + std::exp(this->data));
  Value* result = new Value(sig);
  result->childs.insert(this);
  result->op = "sigmoid";
  
  result->_backward = [this, result]() {
    this->gradient += result->gradient * (result->data * (1 - result->data));
  };
  return result;
}

void Value::operator +=(Value* other) { 
  this->data = this->data + other->data;
  this->childs.insert(other);
  this->op = "+=";
  this->_backward = [this] () {
    for (Value* child : this->childs){
      child->gradient += this->gradient;
    }
  };
}

void Value::_zeroGrad() {
  this->gradient = 0.0;
}

void Value::zeroGrad() {
  _zeroGrad();
  if (childs.size() > 0) {
    for (auto& child : childs) {
      child->zeroGrad();
    }
  }
}

void Value::deleteGraph(){
  std::unordered_set<Value*> visited;
  std::stack<Value*> nodes;
  collect_nodes(nodes, visited);

  while(!nodes.empty()) {
    Value* current = nodes.top();
    nodes.pop();
    delete current;
  }
}

void Value::collect_nodes(std::stack<Value*>& collected, std::unordered_set<Value*>& visited) {
  // Here I collect nodes for cleaning up heap allocated memory.
  // I want to ensure that the input data is not deleted.

  const bool is_in = visited.find(this) != visited.end();
  if (is_in | isParameter) {
        return; // Already visited
  }
  visited.insert(this);
  
  // If we reach the leafes (input data) we do not push them onto the stack
  if (childs.size() > 0){
    for (Value* child : childs) {
      child->collect_nodes(collected, visited);
    }
  collected.push(this);
  }
}


void Value::topological_sort(std::vector<Value*> &topo_vector,
                                            std::vector<Value*> &visited) {
  // check if the current Node was already visited.
  // It would probably more efficient to store a bool _visited within Value :TODO
  if (std::find(visited.begin(), visited.end(), this) != visited.end()) {
    return;
  }
  visited.emplace_back(this);
  for (Value* child : childs) {
    child->topological_sort(topo_vector, visited);
  }
  // Add the first node to the vector (This will be called at at the first node)
  topo_vector.push_back(this);
}

void Value::backward() {
  this->gradient = 1.0; // df/df = 1.0

  std::vector<Value*> topo_vector;
  std::vector<Value*> visited;
  topological_sort(topo_vector, visited);
  
  for (std::vector<Value*>::iterator it_end = topo_vector.end() ; it_end != topo_vector.begin();){
    it_end--;
    if ((*it_end)->_backward) {
      (*it_end)->_backward();
    }
  }
}

