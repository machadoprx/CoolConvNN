//
// Created by vmachado on 2/11/20.
//

#include "cnn.h"

cnn* cnn_alloc(const char* cnn_cfg) {

    cnn *net = aligned_alloc(CACHE_LINE, sizeof(*net));

    FILE *cfg_fp = fopen(cnn_cfg, "r");
    fscanf(cfg_fp, "%d\n", &net->conv_pool_layers);

    net->conv = aligned_alloc(CACHE_LINE, sizeof(*net->conv) * net->conv_pool_layers);
    net->pool = aligned_alloc(CACHE_LINE, sizeof(*net->pool) * net->conv_pool_layers);
    
    for (int i = 0; i < net->conv_pool_layers; i++) {
        int in_c, out_c, stride, f_size, padd, in_w, in_h;
        
        fscanf(cfg_fp, "%d %d %d %d %d %d %d\n", &in_c, &out_c, &stride, &f_size, &padd, &in_w, &in_h);
        net->conv[i] = conv_alloc(in_c, in_w, in_h, out_c, f_size, stride, padd);

        fscanf(cfg_fp, "%d %d %d %d %d %d\n", &stride, &f_size, &in_c, &padd, &in_w, &in_h);
        net->pool[i] = pool_alloc(in_c, in_w, in_h, f_size, stride, padd);
    }

    fscanf(cfg_fp, "%d\n", &net->fc_add);
    fscanf(cfg_fp, "%d %d %d\n", &net->fc_in, &net->fc_out, &net->fc_lsize);
    net->fc = aligned_alloc(CACHE_LINE, sizeof(*net->fc) * (2 + net->fc_add));

    int i = 1;
    net->fc[0] = fc_alloc(net->fc_in, net->fc_lsize, true);
    for (; i <= net->fc_add; i++) {
        net->fc[i] = fc_alloc(net->fc_lsize, net->fc_lsize, true);
    }
    net->fc[i] = fc_alloc(net->fc_lsize, net->fc_out, false);

    fscanf(cfg_fp, "%d %f\n", &net->batch_size, &net->l_rate);

    net->l_reg = 1e-3;

    fclose(cfg_fp);

    return net;
}

cnn* cnn_load(const char* cnn_config, const char* cnn_state) { // array alignment

    FILE *cnn_state_fp = fopen(cnn_state, "rb");
    cnn *net = cnn_alloc(cnn_config);
    
    for (int i = 0; i < net->conv_pool_layers; i++) {
        
        conv_layer *l = net->conv[i];
        int w_size = l->out_c * l->f_size * l->f_size * l->in_c;

        fread(l->filters->data, sizeof(float) * w_size, 1, cnn_state_fp);
        fread(l->gamma->data, sizeof(float) * l->in_c, 1, cnn_state_fp);
        fread(l->beta->data, sizeof(float) * l->in_c, 1, cnn_state_fp);
        fread(l->run_mean->data, sizeof(float) * l->in_c, 1, cnn_state_fp);
        fread(l->run_var->data, sizeof(float) * l->in_c, 1, cnn_state_fp);
    }

    for (int i = 0; i < net->fc_add + 2; i++) {
        fc_layer *l = net->fc[i];
        int in_dim = l->weights->rows, out_dim = l->weights->columns;

        fread(l->weights->data, sizeof(float) * in_dim * out_dim, 1, cnn_state_fp);
        fread(l->gamma->data, sizeof(float) * in_dim, 1, cnn_state_fp);
        fread(l->beta->data, sizeof(float) * in_dim, 1, cnn_state_fp);
        fread(l->run_mean->data, sizeof(float) * in_dim, 1, cnn_state_fp);
        fread(l->run_var->data, sizeof(float) * in_dim, 1, cnn_state_fp);

    }

    fclose(cnn_state_fp);
    return net;
}

void cnn_free(cnn *net) {
    for (int i = 0; i < net->conv_pool_layers; i++) {
        conv_free(net->conv[i]);
        pool_free(net->pool[i]);
    }
    for (int i = 0; i < net->fc_add + 2; i++) {
        fc_free(net->fc[i]);
    }
    free(net->pool);
    free(net->conv);
    free(net->fc);
    free(net);
}

void cnn_save(cnn *net, const char* cnn_state) {

    FILE *cnn_state_fp = fopen(cnn_state, "wb");
    
    for (int i = 0; i < net->conv_pool_layers; i++) {
        conv_layer *l = net->conv[i];
        int w_size = l->out_c * l->f_size * l->f_size * l->in_c;

        fwrite(l->filters->data, sizeof(float) * w_size, 1, cnn_state_fp);
        fwrite(l->gamma->data, sizeof(float) * l->in_c, 1, cnn_state_fp);
        fwrite(l->beta->data, sizeof(float) * l->in_c, 1, cnn_state_fp);
        fwrite(l->run_mean->data, sizeof(float) * l->in_c, 1, cnn_state_fp);
        fwrite(l->run_var->data, sizeof(float) * l->in_c, 1, cnn_state_fp);
    }

    for (int i = 0; i < net->fc_add + 2; i++) {
        fc_layer *l = net->fc[i];
        int in_dim = l->weights->rows, out_dim = l->weights->columns;

        fwrite(l->weights->data, sizeof(float) * in_dim * out_dim, 1, cnn_state_fp);
        fwrite(l->gamma->data, sizeof(float) * in_dim, 1, cnn_state_fp);
        fwrite(l->beta->data, sizeof(float) * in_dim, 1, cnn_state_fp);
        fwrite(l->run_mean->data, sizeof(float) * in_dim, 1, cnn_state_fp);
        fwrite(l->run_var->data, sizeof(float) * in_dim, 1, cnn_state_fp);
    }

    fclose(cnn_state_fp);
}

matrix* cnn_forward(cnn *net, matrix *batch, bool training) {

    int i;
    matrix *curr = mat_copy(batch), *tmp;
    
    for (i = 0; i < net->conv_pool_layers; i++) {
        tmp = conv_forward(net->conv[i], curr, training);
        matrix_free(curr);
        curr = pool_forward(net->pool[i], tmp);
        matrix_free(tmp);
    }

    for (i = 0; i < net->fc_add + 2; i++) {
        tmp = fc_forward(net->fc[i], curr, training);
        matrix_free(curr);
        curr = tmp;
    }

    normalize(curr);

    return curr;
}

void cnn_backward(cnn *net, matrix *prob, matrix *batch, int *labels) {
    
    int i;
    matrix *tmp;
    matrix *dout = prob_del(prob, labels);

    for (i = net->fc_add + 1; i >= 0; --i) {
        tmp = fc_backward(net->fc[i], dout, net->l_reg, net->l_rate);
        matrix_free(dout);
        dout = tmp;
    }

    for (i = net->conv_pool_layers - 1; i >= 0; --i) {
        tmp = pool_backward(net->pool[i], dout);
        matrix_free(dout);
        dout = conv_backward(net->conv[i], tmp, net->l_rate);
        matrix_free(tmp);
    }

    matrix_free(dout);
}

void cnn_train(cnn *net, float **data_set, int *labels, int samples, int epochs) {

    int num_batches = samples % net->batch_size != 0 ?
                    (samples / net->batch_size) + 1
                    : samples / net->batch_size;

    for (int e = 1; e <= epochs; e++) {

        int data_index = 0;
        float total_loss = 0;

        // shuffle dataset
        int *indices = random_indices(samples);

        for (int k = 0; k < num_batches; k++) {

            // prepare batch
            int batch_len = (data_index + net->batch_size >= samples) ?
                            samples - data_index : net->batch_size;
            
            int *batch_labels = aligned_alloc(CACHE_LINE, sizeof(int) * batch_len);
            matrix *batch = matrix_alloc(batch_len, net->conv[0]->in_dim);
            
            get_batch(indices + data_index, data_set, labels, batch_len, net->conv[0]->in_dim, batch, batch_labels);

            //forward step
            matrix *prob = cnn_forward(net, batch, true);

            // get correct probabilities for each class
            matrix *corr_prob = correct_prob(prob, batch_labels);

            // compute loss
            total_loss += loss(corr_prob) + reg_loss(net->fc, 2 + net->fc_add, net->l_reg);

            // backpropagation step
            cnn_backward(net, prob, batch, batch_labels);

            // update data index
            data_index += batch_len;

            //clean
            matrix_free(batch);
            matrix_free(prob);
            matrix_free(corr_prob);
            free(batch_labels);
        }
        printf("epoch: %d loss: %f\n", e, total_loss / (float)num_batches);
        free(indices);
    }
}