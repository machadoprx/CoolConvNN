#include <iostream>
#include "NeuralNet.h"

int main() {

    //double input[4][2] = {{1, 0}, {0, 0}, {1, 1}, {0, 1}};
    auto **input = new double*[4];
    for (int i = 0; i < 4; i++) {
        input[i] = new double[2];
    }
    input[0][0] = 1;
    input[0][1] = 0;
    input[1][0] = 0;
    input[1][1] = 0;
    input[2][0] = 1;
    input[2][1] = 1;
    input[3][0] = 0;
    input[3][1] = 1;

    auto *output = new int[4];
    output[0] = 1;
    output[1] = 0;
    output[2] = 0;
    output[3] = 1;



    auto *nn = new NeuralNet(2, 2, 0 , 100, 1);
    nn->train(input, output, 4, 100);

    delete nn;

    return 0;
}
