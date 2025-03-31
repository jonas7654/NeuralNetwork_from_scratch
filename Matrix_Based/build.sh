#!/bin/bash

build_dir="../build"

cd src
clang++ -Wall -o "$build_dir/main" main.cpp value_matrix.cpp nn.cpp -lopenblas
clang++ -Wall -o "$build_dir/mnist_parser" ../util/mnist_parser.cpp value_matrix.cpp -lopenblas


exit
