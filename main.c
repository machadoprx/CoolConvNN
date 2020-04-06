#include <stdio.h>
#include "cnn.h"
#include "omp.h"
#include "png2array/png2array.h"

int main(int argc, char const *argv[]) {

    int nProcessors = omp_get_max_threads();
    omp_set_num_threads(nProcessors);
    
    const char *data_file = "data_processed.dat";
    const char *cnn_file = "cnn_state.dat";
    const char *param_file = "params.ini";
    const char *mode = argv[1];

    int labels, samplesPerLabels, featuresDimension;
    float *mean, *deviation, **input;
    load_data(data_file, &input, &mean, &deviation, &labels, &samplesPerLabels, &featuresDimension);
    int *targets = gen_targets(labels, samplesPerLabels);
    cnn *net = NULL;

    printf("Features Dimension: %d\n", featuresDimension);
    printf("Number of labels: %d\n", labels);
    printf("Samples/Label: %d\n\n", samplesPerLabels);

    if (strcmp(mode, "new") == 0 || strcmp(mode, "continue") == 0) {
        int epochs = atoi(argv[2]);
        if (strcmp(mode, "new") == 0) {
            net = cnn_alloc(param_file);
        }
        else {
            net = cnn_load(param_file, cnn_file);
        }
        cnn_train(net, input, targets, labels * samplesPerLabels, epochs);
        cnn_save(net, cnn_file);
    }
    else if (strcmp(mode, "test") == 0) {

        int w, h;
        const char *test_path = argv[2];

        float *sample = decode_png(test_path, &w, &h);
        matrix *test = matrix_alloc(1, w * h);
        for (int i = 0; i < w * h; i++) {
            test->data[i] = (sample[i] - mean[i]) / deviation[i];
        }
        cnn *net = cnn_load(param_file, cnn_file);
        matrix *result = cnn_forward(net, test, false);

        for (int i = 0; i < labels; i++) {
            printf("label: %d prob: %f\n", i, result->data[i] * 100.0f);
        }
        matrix_free(result);
        matrix_free(test);
        free(sample);
    }
    if (net != NULL)
        cnn_free(net);
    free(mean);
    free(deviation);
    free(targets);
    for (int i = 0; i < labels * samplesPerLabels; i++) {
        free(input[i]);
    }
    free(input);
    return 0;
}
