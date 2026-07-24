#include <iostream>
#include <fstream>
#include <iomanip>
#include <hls_stream.h>
#include <ap_int.h>

#include "cnn_axis.h"
#include "cnn.h"

using namespace std;

// =======================================================
// Alege testul
// 1 = MNIST 10000 imagini
// 0 = CUSTOM imagini
// =======================================================

#define TEST_MNIST_10000 1

#if TEST_MNIST_10000
    const int NUM_IMAGES = 10000;
    const char* IMAGE_FILE = "mnist_10000_images.txt";
    const char* LABEL_FILE = "mnist_10000_labels.txt";
    const char* TEST_NAME  = "MNIST TEST 10000";
#else
    const int NUM_IMAGES = 700;
    const char* IMAGE_FILE = "custom_images.txt";
    const char* LABEL_FILE = "custom_labels.txt";
    const char* TEST_NAME  = "CUSTOM DATASET";
#endif

// =======================================================
// Creează pachet AXI pentru intrare FIXED-POINT RAW
//
// Imaginea din fișier este citită ca float normalizat 0..1,
// dar NU este trimisă ca float.
// Este convertită în data_t, apoi se trimit biții raw.
// =======================================================

static axis_t make_input_packet(float pixel, bool last) {
    axis_t pkt;

    data_t fixed_pixel = (data_t)pixel;

    ap_int<32> raw32 = 0;
    raw32.range(7, 0) = fixed_pixel.range(7, 0);

    pkt.data = raw32;
    pkt.keep = -1;
    pkt.strb = -1;
    pkt.last = last ? 1 : 0;

    return pkt;
}

// =======================================================
// Citește pachet AXI de ieșire FIXED-POINT RAW
//
// cnn_axis trimite raw out_t pe 11 biți.
// Aici reconstruim out_t și îl convertim în float doar
// pentru afișare și argmax în testbench.
// =======================================================

static float read_output_packet(axis_t pkt) {
    out_t fixed_score;

    fixed_score.range(10, 0) = pkt.data.range(10, 0);

    return (float)fixed_score;
}

int main() {
    ifstream img_file(IMAGE_FILE);
    ifstream label_file(LABEL_FILE);

    if (!img_file.is_open()) {
        cout << "EROARE: nu pot deschide fisierul de imagini: "
             << IMAGE_FILE << endl;
        return 1;
    }

    if (!label_file.is_open()) {
        cout << "EROARE: nu pot deschide fisierul de label-uri: "
             << LABEL_FILE << endl;
        return 1;
    }

    int correct = 0;
    int total = 0;

    int true_count[10] = {0};
    int pred_count[10] = {0};
    int correct_count[10] = {0};

    int confusion[10][10];

    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            confusion[i][j] = 0;
        }
    }

    cout << "======================================" << endl;
    cout << " TEST ACURATETE CNN AXI - FIXED RAW" << endl;
    cout << " Set test:    " << TEST_NAME << endl;
    cout << " Images file: " << IMAGE_FILE << endl;
    cout << " Labels file: " << LABEL_FILE << endl;
    cout << " Nr. imagini: " << NUM_IMAGES << endl;
    cout << "======================================" << endl;

    for (int n = 0; n < NUM_IMAGES; n++) {
        hls::stream<axis_t> in_stream;
        hls::stream<axis_t> out_stream;

        float sum_pixels = 0.0f;
        float max_pixel = 0.0f;
        float min_pixel = 999999.0f;

        // ===================================================
        // Citește imaginea n: 28 x 28 = 784 valori
        // Fișierul rămâne cu valori float normalizate.
        // Pe AXI Stream se trimite data_t raw, nu float.
        // ===================================================

        for (int i = 0; i < INPUT_H; i++) {
            for (int j = 0; j < INPUT_W; j++) {
                float value;

                if (!(img_file >> value)) {
                    cout << "EROARE: fisierul de imagini nu are suficiente valori." << endl;
                    cout << "Fisier: " << IMAGE_FILE << endl;
                    cout << "Imaginea: " << n
                         << ", pixel: " << i << "," << j << endl;
                    return 1;
                }

                sum_pixels += value;

                if (value > max_pixel) {
                    max_pixel = value;
                }

                if (value < min_pixel) {
                    min_pixel = value;
                }

                bool last = (i == INPUT_H - 1 && j == INPUT_W - 1);

                axis_t pkt = make_input_packet(value, last);

                in_stream.write(pkt);
            }
        }

        // ===================================================
        // Citește label-ul imaginii n
        // ===================================================

        int true_label;

        if (!(label_file >> true_label)) {
            cout << "EROARE: fisierul de label-uri nu are suficiente valori." << endl;
            cout << "Fisier: " << LABEL_FILE << endl;
            cout << "Imaginea: " << n << endl;
            return 1;
        }

        if (true_label < 0 || true_label > 9) {
            cout << "EROARE: label invalid la imaginea "
                 << n << ": " << true_label << endl;
            return 1;
        }

        // ===================================================
        // Rulează acceleratorul
        // ===================================================

        cnn_axis(in_stream, out_stream);

        // ===================================================
        // Citește cele 10 scoruri / logits
        // Ieșirea este raw out_t, reconstruită aici în float.
        // ===================================================

        float scores[FC2_OUT];

        for (int k = 0; k < FC2_OUT; k++) {
            if (out_stream.empty()) {
                cout << "EROARE: out_stream este gol prea devreme." << endl;
                cout << "Imaginea: " << n << ", scor: " << k << endl;
                return 1;
            }

            axis_t pkt_out = out_stream.read();

            scores[k] = read_output_packet(pkt_out);

            if (k == FC2_OUT - 1 && pkt_out.last != 1) {
                cout << "AVERTISMENT: ultimul pachet nu are TLAST=1 la imaginea "
                     << n << endl;
            }
        }

        // ===================================================
        // Argmax
        // ===================================================

        int predicted = 0;
        float max_val = scores[0];

        for (int k = 1; k < FC2_OUT; k++) {
            if (scores[k] > max_val) {
                max_val = scores[k];
                predicted = k;
            }
        }

        // ===================================================
        // Statistici
        // ===================================================

        true_count[true_label]++;
        pred_count[predicted]++;
        confusion[true_label][predicted]++;

        if (predicted == true_label) {
            correct++;
            correct_count[true_label]++;
        }

        total++;

        // Primele 30 exemple pentru diagnostic
        if (n < 30) {
            cout << "Img " << setw(5) << n
                 << " | true=" << true_label
                 << " | pred=" << predicted
                 << " | sum_pixels=" << fixed << setprecision(2) << sum_pixels
                 << " | min_pixel=" << min_pixel
                 << " | max_pixel=" << max_pixel
                 << " | scores=[";

            for (int k = 0; k < FC2_OUT; k++) {
                cout << fixed << setprecision(3) << scores[k];

                if (k != FC2_OUT - 1) {
                    cout << ", ";
                }
            }

            cout << "]" << endl;
        }

        // Progres
        if ((n + 1) % 100 == 0 || (n + 1) == NUM_IMAGES) {
            float acc_partial = 100.0f * correct / total;

            cout << "Procesate: "
                 << setw(5) << (n + 1)
                 << " / " << NUM_IMAGES
                 << " | Corecte: " << setw(5) << correct
                 << " | Accuracy partial: "
                 << fixed << setprecision(2) << acc_partial << "%"
                 << endl;
        }
    }

    // =======================================================
    // Rezultat final
    // =======================================================

    float accuracy = 100.0f * correct / total;

    cout << endl;
    cout << "======================================" << endl;
    cout << " REZULTAT FINAL" << endl;
    cout << "======================================" << endl;
    cout << "Set test:      " << TEST_NAME << endl;
    cout << "Total imagini: " << total << endl;
    cout << "Corecte:       " << correct << " / " << total << endl;
    cout << "Accuracy:      " << fixed << setprecision(4) << accuracy << "%" << endl;
    cout << "Accuracy frac: " << fixed << setprecision(6) << (accuracy / 100.0f) << endl;

    cout << endl;
    cout << "Distributie label-uri reale:" << endl;
    for (int i = 0; i < 10; i++) {
        cout << "Label " << i << ": " << true_count[i] << endl;
    }

    cout << endl;
    cout << "Distributie predictii:" << endl;
    for (int i = 0; i < 10; i++) {
        cout << "Pred " << i << ": " << pred_count[i] << endl;
    }

    cout << endl;
    cout << "Corecte pe clasa:" << endl;
    for (int i = 0; i < 10; i++) {
        float acc_class = 0.0f;

        if (true_count[i] > 0) {
            acc_class = 100.0f * correct_count[i] / true_count[i];
        }

        cout << "Clasa " << i
             << ": " << correct_count[i]
             << " / " << true_count[i]
             << " = " << fixed << setprecision(2) << acc_class << "%"
             << endl;
    }

    cout << endl;
    cout << "Matrice confuzie: rand=true, coloana=pred" << endl;
    cout << "      ";
    for (int p = 0; p < 10; p++) {
        cout << setw(5) << p;
    }
    cout << endl;

    for (int t = 0; t < 10; t++) {
        cout << "T" << t << " | ";
        for (int p = 0; p < 10; p++) {
            cout << setw(5) << confusion[t][p];
        }
        cout << endl;
    }

    cout << "======================================" << endl;

    img_file.close();
    label_file.close();

    return 0;
}