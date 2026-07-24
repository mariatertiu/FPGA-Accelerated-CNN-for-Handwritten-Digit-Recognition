#ifndef CNN_H
#define CNN_H

#include <ap_fixed.h>

// =======================================================
// Dimensiuni rețea
// =======================================================

#define INPUT_H 28
#define INPUT_W 28

#define CONV1_K 3
#define CONV1_IN_C 1
#define CONV1_OUT_C 4
#define CONV1_OUT_H 26
#define CONV1_OUT_W 26

#define POOL1_K 2
#define POOL1_OUT_H 13
#define POOL1_OUT_W 13
#define POOL1_OUT_C 4

#define CONV2_K 3
#define CONV2_IN_C 4
#define CONV2_OUT_C 8
#define CONV2_OUT_H 11
#define CONV2_OUT_W 11

#define POOL2_K 2
#define POOL2_OUT_H 5
#define POOL2_OUT_W 5
#define POOL2_OUT_C 8

#define FLAT_SIZE 200

#define FC1_OUT 32
#define FC2_OUT 10

// =======================================================
// Tipuri numerice pentru varianta 8_4 economică
// =======================================================

// Intrări + greutăți
typedef ap_fixed<8,4, AP_RND, AP_SAT> data_t;

// Activări intermediare
typedef ap_fixed<10,5, AP_RND, AP_SAT> act_t;

// Ieșirea stratului FC1
typedef ap_fixed<10,6, AP_RND, AP_SAT> fc_t;

// Acumulări interne
typedef ap_fixed<12,7, AP_RND, AP_SAT> acc_t;

// Scoruri finale
typedef ap_fixed<11,6, AP_RND, AP_SAT> out_t;

// =======================================================
// Prototip funcție CNN
// =======================================================

void cnn_forward(
    data_t input[INPUT_H][INPUT_W],
    out_t output[FC2_OUT]
);

#endif