#!/bin/bash
mkdir build
cd build
cmake .. && make -j
cd ..
./build/bin/tester 1024 1024 1024 111
