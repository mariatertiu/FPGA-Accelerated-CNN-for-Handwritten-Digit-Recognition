#include "cnn.h"
// contine dimensiunile retelei si tipurile fixed-point


// aici sunt greutatile si bias-urile exportate din model
#include "cnn_weights_8_4.h"


// calculez pozitia unei greutati din conv1 in vectorul liniar
static int idx_conv1(int kh, int kw, int ic, int oc) {
    return (((kh * CONV1_K + kw) * CONV1_IN_C + ic) * CONV1_OUT_C + oc);
    // ordinea este aceeasi ca la greutatile exportate din Keras
}

// calculez pozitia unei greutati din conv2
static int idx_conv2(int kh, int kw, int ic, int oc) {
    return (((kh * CONV2_K + kw) * CONV2_IN_C + ic) * CONV2_OUT_C + oc);
    // include si canalul de intrare, pentru ca conv2 primeste 4 harti
}

// index pentru matricea de greutati fc1
static int idx_fc1(int i, int o) {
    return i * FC1_OUT + o;
    // i = intrare, o = neuronul de iesire
}

// index pentru matricea de greutati fc2
static int idx_fc2(int i, int o) {
    return i * FC2_OUT + o;
    // i = intrare din fc1, o = clasa 0-9
}


// ReLU pentru valorile intermediare dupa convolutii/pooling
static act_t relu_acc_to_act(acc_t x) {
    return (x > 0) ? (act_t)x : (act_t)0;
    // daca suma e negativa, o fac 0
}


// ReLU pentru iesirea stratului fc1
static fc_t relu_acc_to_fc(acc_t x) {
    return (x > 0) ? (fc_t)x : (fc_t)0;
    // acelasi lucru, dar convertesc la tipul folosit in fc1
}


// functia principala care face inferenta CNN
void cnn_forward(
    data_t input[INPUT_H][INPUT_W],
    out_t output[FC2_OUT]
) {
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL
// registre de control pentru start/done
#pragma HLS INTERFACE s_axilite port=input bundle=CTRL
// intrarea apare ca port controlat prin AXI-Lite in aceasta varianta
#pragma HLS INTERFACE s_axilite port=output bundle=CTRL
// iesirea apare ca port controlat prin AXI-Lite

    act_t conv1_out[CONV1_OUT_H][CONV1_OUT_W][CONV1_OUT_C];
    // aici se salveaza rezultatul 26x26x4
    act_t pool1_out[POOL1_OUT_H][POOL1_OUT_W][POOL1_OUT_C];
    // dupa pooling ramane 13x13x4

    act_t conv2_out[CONV2_OUT_H][CONV2_OUT_W][CONV2_OUT_C];
    // rezultatul conv2 este 11x11x8
    act_t pool2_out[POOL2_OUT_H][POOL2_OUT_W][POOL2_OUT_C];
    // dupa pool2 ramane 5x5x8

    act_t flat[FLAT_SIZE];
    // vectorul care intra in fc1
    fc_t fc1_out[FC1_OUT];
    // cele 32 activari de la fc1

#pragma HLS ARRAY_PARTITION variable=fc1_out complete dim=1
// permite acces separat la toate valorile fc1_out
#pragma HLS ARRAY_PARTITION variable=output complete dim=1
// iesirile finale sunt separate pentru acces mai rapid

    
    
    
    
    
    

// optimizarea O2: impart flat ca sa pot citi mai multe valori odata
#pragma HLS ARRAY_PARTITION variable=flat cyclic factor=4 dim=1
// impart si greutatile fc1 pentru acces paralel
#pragma HLS ARRAY_PARTITION variable=fc1_w cyclic factor=4 dim=1

    
    
    

    // primul strat: Conv1 + ReLU
    for (int i = 0; i < CONV1_OUT_H; i++) {
        for (int j = 0; j < CONV1_OUT_W; j++) {
            for (int oc = 0; oc < CONV1_OUT_C; oc++) {
#pragma HLS PIPELINE II=1
// pipeline pe bucla principala a stratului curent

                acc_t sum = (acc_t)conv1_b[oc];
                // suma incepe cu bias-ul filtrului curent

                for (int kh = 0; kh < CONV1_K; kh++) {
                    for (int kw = 0; kw < CONV1_K; kw++) {
                        data_t pixel = input[i + kh][j + kw];
                        // pixelul din fereastra 3x3
                        data_t weight = conv1_w[idx_conv1(kh, kw, 0, oc)];
                        // greutatea corespunzatoare acelui pixel

                        sum += (acc_t)pixel * (acc_t)weight;
                        // inmultire si acumulare
                    }
                }

                conv1_out[i][j][oc] = relu_acc_to_act(sum);
                // salvez rezultatul dupa ReLU
            }
        }
    }

    
    
    

    // primul max-pooling
    for (int i = 0; i < POOL1_OUT_H; i++) {
        for (int j = 0; j < POOL1_OUT_W; j++) {
            for (int c = 0; c < POOL1_OUT_C; c++) {
#pragma HLS PIPELINE II=1
// pipeline pe bucla principala a stratului curent

                act_t max_val = conv1_out[i * 2][j * 2][c];
                // pornesc maximul cu primul element din fereastra 2x2

                for (int pi = 0; pi < POOL1_K; pi++) {
                    for (int pj = 0; pj < POOL1_K; pj++) {
                        act_t val = conv1_out[i * 2 + pi][j * 2 + pj][c];
                        // valoare candidata din fereastra 2x2

                        if (val > max_val) {
                        // daca gasesc o valoare mai mare, actualizez maximul
                            max_val = val;
                        }
                    }
                }

                pool1_out[i][j][c] = max_val;
                // pun maximul in harta redusa
            }
        }
    }

    
    
    

    // al doilea strat: Conv2 + ReLU
    for (int i = 0; i < CONV2_OUT_H; i++) {
        for (int j = 0; j < CONV2_OUT_W; j++) {
            for (int oc = 0; oc < CONV2_OUT_C; oc++) {
#pragma HLS PIPELINE II=1
// pipeline pe bucla principala a stratului curent

                acc_t sum = (acc_t)conv2_b[oc];
                // suma pentru filtrul curent din conv2

                for (int kh = 0; kh < CONV2_K; kh++) {
                    for (int kw = 0; kw < CONV2_K; kw++) {
                        for (int ic = 0; ic < CONV2_IN_C; ic++) {
                            act_t pixel = pool1_out[i + kh][j + kw][ic];
                            // conv2 lucreaza pe iesirea pool1
                            data_t weight = conv2_w[idx_conv2(kh, kw, ic, oc)];
                            // greutate pentru pozitia si canalul curent

                            sum += (acc_t)pixel * (acc_t)weight;
                            // inmultire si acumulare
                        }
                    }
                }

                conv2_out[i][j][oc] = relu_acc_to_act(sum);
                // iesirea conv2 dupa ReLU
            }
        }
    }

    
    
    

    // al doilea max-pooling
    for (int i = 0; i < POOL2_OUT_H; i++) {
        for (int j = 0; j < POOL2_OUT_W; j++) {
            for (int c = 0; c < POOL2_OUT_C; c++) {
#pragma HLS PIPELINE II=1
// pipeline pe bucla principala a stratului curent

                act_t max_val = conv2_out[i * 2][j * 2][c];
                // primul element din fereastra de pooling

                for (int pi = 0; pi < POOL2_K; pi++) {
                    for (int pj = 0; pj < POOL2_K; pj++) {
                        act_t val = conv2_out[i * 2 + pi][j * 2 + pj][c];
                        // verific fiecare element din fereastra 2x2

                        if (val > max_val) {
                        // daca gasesc o valoare mai mare, actualizez maximul
                            max_val = val;
                        }
                    }
                }

                pool2_out[i][j][c] = max_val;
                // salvez valoarea maxima
            }
        }
    }

    
    
    

    // flatten: transform iesirea 5x5x8 intr-un vector de 200 valori
    int index = 0;

    // al doilea max-pooling
    for (int i = 0; i < POOL2_OUT_H; i++) {
        for (int j = 0; j < POOL2_OUT_W; j++) {
            for (int c = 0; c < POOL2_OUT_C; c++) {
#pragma HLS PIPELINE II=1
// pipeline pentru bucla de calcul

                flat[index] = pool2_out[i][j][c];
                // copiez elementul 3D in vectorul flat
                index++;
                // trec la urmatoarea pozitie din vector
            }
        }
    }

    
    
    
    
    
    
    
    

    // FC1 + ReLU
    for (int o = 0; o < FC1_OUT; o++) {

        acc_t sum = (acc_t)fc1_b[o];
        // fiecare neuron porneste de la bias-ul lui

        for (int i = 0; i < FLAT_SIZE; i += 4) {
#pragma HLS PIPELINE II=1
// pipeline pentru bucla de calcul

            act_t x0 = flat[i];
            // iau 4 valori consecutive din flat
            act_t x1 = flat[i + 1];
            // a doua valoare din grup
            act_t x2 = flat[i + 2];
            // a treia valoare din grup
            act_t x3 = flat[i + 3];
            // a patra valoare din grup

            data_t w0 = fc1_w[idx_fc1(i, o)];
            // greutatea pentru x0
            data_t w1 = fc1_w[idx_fc1(i + 1, o)];
            // greutatea pentru x1
            data_t w2 = fc1_w[idx_fc1(i + 2, o)];
            // greutatea pentru x2
            data_t w3 = fc1_w[idx_fc1(i + 3, o)];
            // greutatea pentru x3

            acc_t prod0 = (acc_t)x0 * (acc_t)w0;
            // produsul 1
            acc_t prod1 = (acc_t)x1 * (acc_t)w1;
            // produsul 2
            acc_t prod2 = (acc_t)x2 * (acc_t)w2;
            // produsul 3
            acc_t prod3 = (acc_t)x3 * (acc_t)w3;
            // produsul 4

            sum += prod0 + prod1 + prod2 + prod3;
            // adaug 4 produse intr-o singura iteratie
        }

        fc1_out[o] = relu_acc_to_fc(sum);
        // salvez iesirea neuronului dupa ReLU
    }

    
    
    

    // FC2 calculeaza cele 10 scoruri finale
    for (int o = 0; o < FC2_OUT; o++) {

        acc_t sum = (acc_t)fc2_b[o];
        // suma porneste de la bias-ul clasei curente

        for (int i = 0; i < FC1_OUT; i++) {
#pragma HLS PIPELINE II=1
// pipeline pentru bucla de calcul

            fc_t x = fc1_out[i];
            // valoare venita din fc1
            data_t w = fc2_w[idx_fc2(i, o)];
            // greutatea catre clasa curenta

            sum += (acc_t)x * (acc_t)w;
            // acumulez produsul pentru scorul clasei
        }

        output[o] = (out_t)sum;
        // scor final pentru cifra o, fara ReLU
    }
}
