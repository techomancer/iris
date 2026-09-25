#!/bin/bash

echo "Compiling basic OpenGL 1.0 tests..."
gcc -o gltest main.c -lX11 -lGL -lm
gcc -o glprim glprim.c -lX11 -lGL -lm
echo "Done. Run ./gltest or ./glprim --help"
