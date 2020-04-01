#include "image.h"

//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE

void iam2cool(float *im, int in_c, int in_w, int in_h, int f_size, int stride, int padd, int col_w, int col_h, int col_c, float *out) {
    #pragma omp parallel for
    for (int c = 0; c < col_c; c++) {
        int w_off = c % f_size;
        int h_off = (c / f_size) % f_size;
        int im_chan = c / (f_size * f_size);
        for (int y = 0; y < col_h; y++) {
            for (int x = 0; x < col_w; x++) {
                int im_row = (h_off + (y * stride)) - padd;
                int im_col = (w_off + (x * stride)) - padd;
                int col_index = (c * col_h + y) * col_w + x;
                if (im_row < 0 || im_col < 0 || im_row >= in_h || im_col >= in_w) {
                    out[col_index] = 0;
                }
                else {
                    int im_index = im_col + in_w * (im_row + in_h * im_chan);
                    out[col_index] = im[im_index];
                }
            }
        }
    }
}

void cool2ami(float *cols, int in_c, int in_w, int in_h, int f_size, int stride, int padd, int col_w, int col_h, int col_c, float *out) {
    #pragma omp parallel for
    for (int c = 0; c < col_c; c++) {
        int w_off = c % f_size;
        int h_off = (c / f_size) % f_size;
        int im_chan = c / (f_size * f_size);
        for (int y = 0; y < col_h; y++) {
            for (int x = 0; x < col_w; x++) {
                int im_row = (h_off + (y * stride)) - padd;
                int im_col = (w_off + (x * stride)) - padd;
                if (im_row < 0 || im_col < 0 || im_row >= in_h || im_col >= in_w) {
                    continue;
                }
                else {
                    int im_index = im_col + in_w * (im_row + in_h * im_chan);
                    int col_index = (c * col_h + y) * col_w + x;
                    out[im_index] += cols[col_index];
                }
            }
        }
    }
}