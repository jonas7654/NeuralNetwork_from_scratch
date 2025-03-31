#ifndef VALUE_H
#define VALUE_H
#include <vector>
#include <string>
#include <functional>
#include <unordered_set>
#include <stack>

class Value {
private:
  double data;
  double gradient;
  bool isParameter;
  std::string op;
  std::unordered_set<Value*> childs;
  void topological_sort(std::vector<Value*>& topo_vector,
                                       std::vector<Value*>& visited);

  std::function<void()> _backward;
  void _zeroGrad();
public:
  Value(double _data);
  Value(double _data, bool isParameter);
  Value();
  ~Value();

  double& getData();
  double getGradient() const;

  Value* operator +(Value* other);
  Value* operator -(Value* other);
  Value* operator *(Value* other);
  Value* operator /(Value* other);   
  Value* sigmoid();
  Value* operator = (Value* other);  
  void operator += (Value* other);
  void operator = (double& other);
  
  void backward();
  void zeroGrad();
  void deleteGraph();
  void collect_nodes(std::stack<Value*>& collected, std::unordered_set<Value*>& visited);

 
  void printChilds() const;
  void printValue() const;

};
#endif // !VALUE_H;

