#include <stdio.h>
#include "cnn.h"
#include "omp.h"
#include "png2array/png2array.h"
#include "parse_data.h"

int main(int argc, char const *argv[]) {

    int nProcessors = omp_get_max_threads();
    omp_set_num_threads(nProcessors);
    
    const char *data_file = "data.csv";
    const char *cnn_file = "cnn_state.dat";
    const char *param_file = "params.ini";
    const char *mode = argv[1];

    int *targets, samples, labels_n;
    float *mean, *std;
    float **input;
    char **label_names;

    cnn *net = NULL;

    if (strcmp(mode, "new") == 0 || strcmp(mode, "continue") == 0) {
        parse_csv(data_file, &input, &mean, &std, &targets, &label_names, &samples, &labels_n, 0);
        int epochs = atoi(argv[2]);
        float split = atof(argv[3]);
        if (strcmp(mode, "new") == 0) {
            net = cnn_alloc(param_file);
        }
        else {
            net = cnn_load(param_file, cnn_file);
        }
        
        printf("Number of samples: %d\n", samples);
        printf("Val split: %.2f\n", split);
        printf("Learning Rate: %.4f\n", net->l_rate);
        printf("Batch Size: %d\n\n", net->batch_size);

        cnn_train(net, input, targets, samples, split, epochs);
        cnn_save(net, cnn_file);
        free(targets);
        for (int i = 0; i < samples; i++) {
            free(input[i]);
        }
        free(input);
        free(mean);
        free(std);
        for (int i = 0; i < labels_n; i++) {
            free(label_names[i]);
        }
        free(label_names);
    }
    else if (strcmp(mode, "test") == 0) {
        parse_csv(data_file, &input, &mean, &std, &targets, &label_names, &samples, &labels_n, 1);
        int w, h;
        const char *test_path = argv[2];

        float *sample = decode_png(test_path, &w, &h);
        matrix *test = matrix_alloc(1, w * h);
        for (int i = 0; i < w * h; i++) {
            test->data[i] = (sample[i] - mean[i]) / std[i];
        }
        cnn *net = cnn_load(param_file, cnn_file);
        matrix *result = cnn_forward(net, test, false);

        for (int i = 0; i < labels_n; i++) {
            printf("%s prob: %f\n", label_names[i], result->data[i] * 100.0f);
        }
        matrix_free(result);
        matrix_free(test);
        free(sample);
        free(mean);
        free(std);
        for (int i = 0; i < labels_n; i++) {
            free(label_names[i]);
        }
        free(label_names);
    }
    if (net != NULL)
        cnn_free(net);

    return 0;
}
