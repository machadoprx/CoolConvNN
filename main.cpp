#include <iostream>
#include "NeuralNet.h"
#include "omp.h"
#include "png2array/png2array.h"

using namespace std;

int main(int argc, char const *argv[]) {

    //assert(THREADS % 2 == 0);

    const char *data_file = "data_processed.dat";
    const char *nn_file = "nn_state.dat";
    const char *mode = argv[1];

    if (strcmp(mode, "getdata") == 0) {
        int samplesPerLabel = atoi(argv[2]);
        int width = atoi(argv[3]);
        int labels = atoi(argv[4]);
        const char *source_folder = argv[5];
        processData(source_folder, samplesPerLabel, width, labels, data_file);
        exit(0);
    }
    else if (strcmp(mode, "new") == 0 || strcmp(mode, "continue") == 0) {

        int labels, samplesPerLabels, featuresDimension;
        float *mean, *deviation, **input;

        loadData(data_file, input, mean, deviation, labels, samplesPerLabels, featuresDimension);
        auto targets = genTargets(labels, samplesPerLabels);

        NeuralNet *nn;
        if (strcmp(mode, "new") == 0) {
            int hiddenLayers = atoi(argv[2]);
            int layersDimension = atoi(argv[3]);
            int batches = atoi(argv[4]);

            assert(batches % 2 == 0 && batches >= THREADS);

            float learningRate = atof(argv[5]);
            int epochs = atoi(argv[6]);
            nn = new NeuralNet(featuresDimension, labels, hiddenLayers, layersDimension, batches, learningRate);
            nn->train(input, targets, labels * samplesPerLabels, epochs);
        }
        else {
            float learningRate = atof(argv[2]);
            int epochs = atoi(argv[3]);
            nn = new NeuralNet(nn_file);
            nn->setLearningRate(learningRate);
            nn->train(input, targets, labels * samplesPerLabels, epochs);
        }

        nn->saveState(nn_file);

        delete nn;
        delete[] mean;
        delete[] deviation;
        delete[] targets;
        for (int i = 0; i < labels * samplesPerLabels; i++) {
            delete[] input[i];
        }
        delete[] input;
    }
    else if (strcmp(mode, "test") == 0) {

        int w, h;
        const char *test_path = argv[2];

        int labels, samplesPerLabels, featuresDimension;
        float *mean, *deviation, **input;
        loadData(data_file, input, mean, deviation, labels, samplesPerLabels, featuresDimension);
        for (int i = 0; i < labels * samplesPerLabels; i++) {
            delete[] input[i];
        }
        delete[] input;

        auto sample = decodeTwoSteps(test_path, w, h);
        auto test = new Matrix(32, w * h);
        for (int i = 0; i < w * h; i++) {
            test->data[i] = (sample[i] - mean[i]) / deviation[i];
        }
        auto nn = new NeuralNet(nn_file);
        auto result = nn->forwardStep(test, true);

        for (int i = 0; i < labels; i++) {
            std::cout << "label: " << i << " prob: " << (int)(result->data[i] * 100) << "\n";
        }
        delete nn;
        delete result;
        delete[] mean;
        delete[] deviation;
    }
    return 0;
}
