#include "cool_nn.h"

cool_nn* cool_alloc(const char* nn_config) {
    
    cool_nn *net = aalloc(sizeof(*net));
    FILE *cfg_fp = fopen(nn_config, "r");
    
    fscanf(cfg_fp, "%d\n", &net->layers_n);

    net->layers = aalloc(sizeof(*net->layers) * net->layers_n);
    net->layers_type = aalloc(sizeof(*net->layers_type) * net->layers_n);
    for (int i = 0; i < net->layers_n; i++) {
        char type_name[16];
        fscanf(cfg_fp, "%s ", type_name);
        int type = dec_layer_type(type_name);
        int in_c, out_c, stride, f_size, padd, in_w, in_h, fc_size, activ;

        net->layers_type[i] = type;
        if (type == CONV) {
            fscanf(cfg_fp, "%d %d %d %d %d %d %d\n", &in_c, &out_c, &stride, &f_size, &padd, &in_w, &in_h);
            net->layers[i] = conv_alloc(in_c, in_w, in_h, out_c, f_size, stride, padd);
        }
        else if (type == FC) {
            fscanf(cfg_fp, "%d %d\n", &in_c, &fc_size);
            net->layers[i] = fc_alloc(in_c, fc_size);
        }
        else if (type == MAX_POOL) {
            fscanf(cfg_fp, "%d %d %d %d %d %d\n", &stride, &f_size, &in_c, &padd, &in_w, &in_h);
            net->layers[i] = pool_alloc(in_c, in_w, in_h, f_size, stride, padd);
        }
        else if (type == BN) {
            fscanf(cfg_fp, "%d %d\n", &in_c, &f_size);
            net->layers[i] = bn_alloc(in_c, f_size);
        }
        else if (type == ACTIVATION) {
            char activ_name[16];
            fscanf(cfg_fp, "%s\n", activ_name);
            activ = dec_activation(activ_name);
            net->layers[i] = activations_alloc(activ);
        }
    }

    fscanf(cfg_fp, "%d\n", &net->batch_size);
    fclose(cfg_fp);

    return net;
}

cool_nn* cool_load(const char* nn_config, const char* nn_state) {

    FILE *nn_state_fp = fopen(nn_state, "rb");
    cool_nn *net = cool_alloc(nn_config);
    
    for (int i = 0; i < net->layers_n; i++) {
        if (net->layers_type[i] == CONV) {
            conv_layer *l = (conv_layer*)net->layers[i];
            int w_size = l->out_c * l->f_size * l->f_size * l->in_c;
            fread(l->filters->data, sizeof(float), w_size, nn_state_fp);
        }
        else if (net->layers_type[i] == FC) {
            fc_layer *l = (fc_layer*)net->layers[i];
            int in_dim = l->weights->rows, out_dim = l->weights->columns;
            fread(l->weights->data, sizeof(float), in_dim * out_dim, nn_state_fp);
        }
        else if (net->layers_type[i] == BN) {
            bn_layer *l = (bn_layer*)net->layers[i];

            fread(l->gamma->data, sizeof(float), l->in_channels, nn_state_fp);
            fread(l->beta->data, sizeof(float), l->in_channels, nn_state_fp);
            fread(l->run_mean->data, sizeof(float), l->in_channels, nn_state_fp);
            fread(l->run_var->data, sizeof(float), l->in_channels, nn_state_fp);
        }
    }

    fclose(nn_state_fp);
    return net;
}

void cool_save(cool_nn *net, const char* nn_state) {

    FILE *nn_state_fp = fopen(nn_state, "wb");
    
    for (int i = 0; i < net->layers_n; i++) {
        if (net->layers_type[i] == CONV) {
            conv_layer *l = (conv_layer*)net->layers[i];
            int w_size = l->out_c * l->f_size * l->f_size * l->in_c;
            fwrite(l->filters->data, sizeof(float), w_size, nn_state_fp);
        }
        else if (net->layers_type[i] == FC) {
            fc_layer *l = (fc_layer*)net->layers[i];
            int in_dim = l->weights->rows, out_dim = l->weights->columns;
            fwrite(l->weights->data, sizeof(float), in_dim * out_dim, nn_state_fp);
        }
        else if (net->layers_type[i] == BN) {
            bn_layer *l = (bn_layer*)net->layers[i];

            fwrite(l->gamma->data, sizeof(float), l->in_channels, nn_state_fp);
            fwrite(l->beta->data, sizeof(float), l->in_channels, nn_state_fp);
            fwrite(l->run_mean->data, sizeof(float), l->in_channels, nn_state_fp);
            fwrite(l->run_var->data, sizeof(float), l->in_channels, nn_state_fp);
        }
    }

    fclose(nn_state_fp);
}

void cool_free(cool_nn *net) {
    for (int i = 0; i < net->layers_n; i++) {
        if (net->layers_type[i] == CONV) {
            conv_free((conv_layer*)net->layers[i]);
        }
        else if (net->layers_type[i] == FC) {
            fc_free((fc_layer*)net->layers[i]);
        }
        else if (net->layers_type[i] == MAX_POOL) {
            pool_free((pool_layer*)net->layers[i]);
        }
        else if (net->layers_type[i] == BN) {
            bn_free((bn_layer*)net->layers[i]);
        }
        else if (net->layers_type[i] == ACTIVATION) {
            activations_free((activations_layer*)net->layers[i]);
        }
    }
    free(net->layers_type);
    free(net->layers);
    free(net);
}

matrix* cool_forward(cool_nn *net, matrix *batch, bool training) {

    matrix *curr = NULL, *in = mat_copy(batch);
    
    for (int i = 0; i < net->layers_n; i++) {
        if (curr != NULL) {
            in = mat_copy(curr);
            matrix_free(curr);
        }
        if (net->layers_type[i] == CONV) {
            curr = conv_forward((conv_layer*)net->layers[i], in);
        }
        else if (net->layers_type[i] == FC) {
            curr = fc_forward((fc_layer*)net->layers[i], in);
        }
        else if (net->layers_type[i] == MAX_POOL) {
            curr = pool_forward((pool_layer*)net->layers[i], in);
        }
        else if (net->layers_type[i] == BN) {
            curr = bn_forward((bn_layer*)net->layers[i], in, training);
        }
        else if (net->layers_type[i] == ACTIVATION) {
            curr = activations_forward((activations_layer*)net->layers[i], in);
        }
        matrix_free(in);
    }

    softmax(curr);
    return curr;
}

void cool_backward(cool_nn *net, matrix *prob, matrix *batch, int *indices, int *labels, float l_rate, float l_reg) {
    
    matrix *in = prob_del(prob, indices, labels);
    matrix *curr = NULL;

    for (int i = net->layers_n - 1; i >= 0; i--) {
        if (curr != NULL) {
            in = mat_copy(curr);
            matrix_free(curr);
        }
        if (net->layers_type[i] == CONV) {
            curr = conv_backward((conv_layer*)net->layers[i], in, l_rate);
        }
        else if (net->layers_type[i] == FC) {
            curr = fc_backward((fc_layer*)net->layers[i], in, l_reg, l_rate);
        }
        else if (net->layers_type[i] == MAX_POOL) {
            curr = pool_backward((pool_layer*)net->layers[i], in);
        }
        else if (net->layers_type[i] == BN) {
            curr = bn_backward((bn_layer*)net->layers[i], in, l_rate);
        }
        else if (net->layers_type[i] == ACTIVATION) {
            curr = activations_backward((activations_layer*)net->layers[i], in);
        }
        matrix_free(in);
    }

    matrix_free(curr);
}

void cool_train(cool_nn *net, float **data_set, int *labels, int samples, float val_split, float l_rate, float l_reg, int epochs) {

    int val_len = samples * val_split;
    int train_samples = samples - val_len;
    int *val_indices = aalloc(sizeof(int) * val_len);

    for (int i = 0; i < val_len; i++) {
        val_indices[i] = i + train_samples;
    }

    int num_batches = train_samples % net->batch_size != 0 ?
                    (train_samples / net->batch_size) + 1
                    : train_samples / net->batch_size;

    int input_dim = net->layers_type[0] == CONV ? (((conv_layer*)net->layers[0])->in_dim)
                                                : (((fc_layer*)net->layers[0])->weights->rows);

    for (int e = 1; e <= epochs; e++) {

        int data_index = 0;
        float total_loss = 0;
        int *indices = random_indices(train_samples);
        float _reg_loss = 0;

        for (int k = 0; k < num_batches; k++) {
            
            // prepare batch
            int batch_len = (data_index + net->batch_size >= train_samples) ?
                            train_samples - data_index : net->batch_size;
            
            matrix *batch = matrix_alloc(batch_len, input_dim);

            get_batch(indices + data_index, data_set, batch_len, input_dim, batch);

            //forward step
            matrix *prob = cool_forward(net, batch, true);

            // get correct probabilities for each class
            matrix *corr_prob = correct_prob(prob, indices + data_index, labels);

            // compute loss
            _reg_loss = reg_loss(net->layers, net->layers_type, net->layers_n, l_reg);
            total_loss += loss(corr_prob) + _reg_loss;

            // backpropagation step
            cool_backward(net, prob, batch, indices + data_index, labels, l_rate, l_reg);

            // update data index
            data_index += batch_len;

            //clean
            matrix_free(prob);
            matrix_free(corr_prob);
            matrix_free(batch);
        }

        if (val_len > 0) {
            matrix *val = matrix_alloc(val_len, input_dim);

            get_batch(val_indices, data_set, val_len, input_dim, val);

            //forward step
            matrix *val_prob = cool_forward(net, val, false);

            // get correct probabilities for each class
            matrix *val_corr_prob = correct_prob(val_prob, val_indices, labels);

            // compute loss
            float val_loss = loss(val_corr_prob) + _reg_loss;
            printf("epoch: %d loss: %g val_loss: %g\n", e, total_loss / (float)num_batches, val_loss);
            matrix_free(val);
            matrix_free(val_prob);
            matrix_free(val_corr_prob);
        }
        else {
            printf("epoch: %d loss: %g\n", e, total_loss / (float)num_batches);
        }
        free(indices);
    }
    free(val_indices);
}