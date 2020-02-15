#include <iostream>
#include "NeuralNet.h"
#include "png2array/png2array.h"

int main(int argc, char const *argv[]) {

    const char *path = "/home/vmachado/Documents/cppdata.dat";
    const char *pathNN = "/home/vmachado/Documents/NeuralNetcpp2.dat";
    const char *dataPath = "/home/vmachado/Documents/quickdraw2";

    //processData(dataPath, 1000, 28, 15, path);

    int labels, samplesLabels, featuresDim;
    double *mean, *deviation, **input;

    loadData(path, input, mean, deviation, labels, samplesLabels, featuresDim);
    auto targets = genTargets(labels, samplesLabels);

    int w, h;
    //const char * vp = "/home/vmachado/Documents/quiquidrawtest/testapple.png";
    std::string validatePath = std::string(argv[7]);
    auto sample = decodeTwoSteps(validatePath, w, h);
    auto test = new Matrix(1, w * h);
    for (int i = 0; i < w * h; i++) {
        test->data[i] = (sample[i] - mean[i]) / deviation[i];
    }

    NeuralNet *nn;
    if (atoi(argv[1]) == 1) {
        nn = new NeuralNet(pathNN);
        nn->setLearningRate(atof(argv[4]));
    }
    else {
        nn = new NeuralNet(featuresDim, labels, atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atof(argv[5]));
    }
    
    nn->train(input, targets, labels * samplesLabels, atoi(argv[6]));
    nn->saveState(pathNN);

    auto result = nn->forwardStep(test, true);

    std::cout << "\n";
    for (int i = 0; i < labels; i++) {
        std::cout << "label: " << i << " prob: " << (int)(result->data[i] * 100) << "\n";
    }

    delete result;
    delete[] sample;
    delete[] mean;
    delete[] deviation;
    delete[] targets;
    for (int i = 0; i < labels * samplesLabels; i++) {
        delete[] input[i];
    }
    delete[] input;
    delete nn;

    return 0;
}
