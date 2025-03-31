#include "value.h"

int main() {
  Value* test = new Value(20);
  //Value* test_ptr = &test;
  Value* a = new Value(2);
  Value* b = new Value(3);

  Value* c = *a + b;
  Value* d = *c * a;

  Value* n = *test + d;
  
  a->printValue();
  b->printValue();
  c->printValue();
  d->printValue();
  n->printValue();

  n->deleteGraph();


  return 0;
}
