#!/bin/bash
cd src
#clang++ -Wall -o main main.cpp value_matrix.cpp nn.cpp -lopenblas
clang++ -Wall -o mnist_parser ../util/mnist_parser.cpp value_matrix.cpp -lopenblas
exit
