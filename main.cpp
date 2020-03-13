#include <iostream>
#include "ConvNeuralNet.h"
#include "omp.h"
#include "png2array/png2array.h"

using namespace std;

int main(int argc, char const *argv[]) {

    assert(THREADS % 2 == 0);

    const char *data_file = "data_processed.dat";
    const char *cnn_file = "cnn_state2.dat";
    const char *param_file = "params.ini";
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

        if (strcmp(mode, "new") == 0) {
            int epochs = atoi(argv[2]);
            auto cnn = new ConvNeuralNet(param_file);
            cnn->train(input, targets, labels * samplesPerLabels, epochs);
            cnn->saveState(cnn_file);
            delete cnn;
        }
        else {
            int epochs = atoi(argv[2]);
            auto cnn = new ConvNeuralNet(param_file, cnn_file);
            cnn->train(input, targets, labels * samplesPerLabels, epochs);
            cnn->saveState(cnn_file);
            delete cnn;
        }

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
        auto cnn = new ConvNeuralNet(param_file, cnn_file);
        auto result = cnn->forwardStep(test, true);

        for (int i = 0; i < labels; i++) {
            std::cout << "label: " << i << " prob: " << (int)(result->data[i] * 100) << "\n";
        }
        delete cnn;
        delete result;
        delete[] mean;
        delete[] deviation;
    }
    return 0;
}
