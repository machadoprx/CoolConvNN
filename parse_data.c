#include "parse_data.h"

void parse_csv(const char *csv_name, float ***data, float **mean, float **std, int **labels, char ***label_names, int *samples, int *lab_n, int test) {
    FILE *csv = fopen(csv_name, "r");
    char buffer[64];
    char names[8192];
    int n, labels_n, features_len;

    fscanf(csv, "%d,%d,%d,%[^\n]\n", &n, &features_len, &labels_n, names);
    (*label_names) = calloc(labels_n, sizeof(char*));
    
    char *names_ptr = names;
    for (int i = 0; i < labels_n; i++) {
        (*label_names)[i] = calloc(32, sizeof(char));
        int k = 0;
        while (*names_ptr != ',' && *names_ptr != '\0') {
            (*label_names)[i][k] = *names_ptr;
            k++;
            names_ptr++;
        }
        (*label_names)[i][k] = 0;
        names_ptr++;
    }

    if (test == 0) {
        (*data) = aalloc(n * sizeof(float*));
        (*labels) = aalloc(n * sizeof(int));
    }
        
    (*mean) = aalloc(features_len * sizeof(float));
    (*std) = aalloc(features_len * sizeof(float));
    
    *samples = n;
    *lab_n = labels_n;

    for (int k = 0; k < features_len; k++) {
        fscanf(csv, "%[^,\n]", buffer);
        (*mean)[k] = atof(buffer);
        fseek(csv, sizeof(char), SEEK_CUR);
    }

    for (int k = 0; k < features_len; k++) {
        fscanf(csv, "%[^,\n]", buffer);
        (*std)[k] = atof(buffer);
        fseek(csv, sizeof(char), SEEK_CUR);
    }
    if (test == 1) {
        goto out;
    }

    for (int i = 0; i < n; i++) {
        (*data)[i] = aalloc(features_len * sizeof(float));
        for (int k = 0; k < features_len; k++) {
            fscanf(csv, "%[^,]", buffer);
            (*data)[i][k] = atof(buffer);
            fseek(csv, sizeof(char), SEEK_CUR);
        }
        fscanf(csv, "%[^\n]\n", buffer);
        (*labels)[i] = atoi(buffer);
    }
    out:
    fclose(csv);
}