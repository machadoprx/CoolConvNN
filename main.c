#include <stdio.h>
#include "cool_nn.h"
#include "omp.h"
#include "png2array/png2array.h"
#include "parse_data.h"

int main(int argc, char const *argv[]) {

    int n_proc = omp_get_max_threads();
    omp_set_num_threads(n_proc);
    
    const char *data_file = "data.csv";
    const char *nn_file = "nn_state.dat";
    const char *param_file = "params.ini";
    const char *mode = argv[1];

    int *targets, samples, labels_n;
    float *mean, *std;
    float **input;
    char **label_names;

    cool_nn *net = NULL;

    if (strcmp(mode, "new") == 0 || strcmp(mode, "continue") == 0) {
        parse_csv(data_file, &input, &mean, &std, &targets, &label_names, &samples, &labels_n, 0);
        float split = atof(argv[2]);
        float l_rate = atof(argv[3]);
        float l_reg = atof(argv[4]);
        int batch_size = atof(argv[5]);
        int epochs = atoi(argv[6]);
        if (strcmp(mode, "new") == 0) {
            net = cool_alloc(param_file);
        }
        else {
            net = cool_load(param_file, nn_file);
        }
        printf("Number of samples: %d\n", samples);
        printf("Val split: %g\n", split);
        printf("Learning Rate: %g\n", l_rate);
        printf("Batch Size: %d\n\n", batch_size);

        cool_train(net, input, targets, samples, split, l_rate, l_reg, batch_size, epochs);
        cool_save(net, nn_file);
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
        net = cool_load(param_file, nn_file);
        matrix *result = cool_forward(net, test, false);

        for (int i = 0; i < labels_n; i++) {
            printf("%s prob: %g\n", label_names[i], result->data[i] * 100.0f);
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
    if (net != NULL){
        cool_free(net);
    }

    return 0;
}
