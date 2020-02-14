#include <iostream>
#include "NeuralNet.h"

double **loadData(const char* path, double* &mean, double* &deviation, int &labels, int &samplesPerLabels, int &featuresDimension) {

    FILE *fp = fopen(path, "rb");
    double **data;

    fread(&labels, sizeof(int), 1, fp);
    fread(&samplesPerLabels, sizeof(int), 1, fp);
    fread(&featuresDimension, sizeof(int), 1, fp);

    mean = new double[featuresDimension];
    deviation = new double[featuresDimension];
    data = new double*[labels * samplesPerLabels];

    for (int i = 0; i < labels * samplesPerLabels; i++) {
        data[i] = new double[featuresDimension];
    }

    fread(mean, sizeof(double) * featuresDimension, 1, fp);
    fread(deviation, sizeof(double) * featuresDimension, 1, fp);

    for (int i = 0; i < labels * samplesPerLabels; i++) {
        fread(data[i], sizeof(double) * featuresDimension, 1, fp);
    }

    fclose(fp);
    return data;
}

int *genTargets(int labels, int samplesPerLabels) {

    auto targets = new int[labels * samplesPerLabels];

    for (int i = 0; i < labels; i++) {
        for (int j = 0; j < samplesPerLabels; j++) {
            targets[i * samplesPerLabels + j] = i;
        }
    }

    return targets;
}

int main() {

    const char *path = "/home/vmachado/Documents/cppdata.dat";
    int labels, samplesLabels, featuresDim;
    double *mean, *deviation, **input;

    input = loadData(path, mean, deviation, labels, samplesLabels, featuresDim);
    int *targets = genTargets(labels, samplesLabels);

    auto nn = new NeuralNet(featuresDim, labels, 0, 512, 32);
    nn->train(input, targets, labels * samplesLabels, 100);

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
