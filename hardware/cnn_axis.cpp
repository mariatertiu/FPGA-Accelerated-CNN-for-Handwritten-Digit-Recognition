#include "cnn_axis.h"
#include "cnn.h"

void cnn_axis(
    hls::stream<axis_t>& in_stream,
    hls::stream<axis_t>& out_stream
) {
#pragma HLS INTERFACE axis port=in_stream
#pragma HLS INTERFACE axis port=out_stream
#pragma HLS INTERFACE s_axilite port=return

    data_t input[INPUT_H][INPUT_W];
    out_t output[FC2_OUT];

#pragma HLS ARRAY_PARTITION variable=output complete dim=1

    // ===================================================
    // Citire input AXI Stream
    // Intrarea NU mai este float.
    // Se primește reprezentarea raw a lui data_t.
    //
    // data_t = ap_fixed<8,4, AP_RND, AP_SAT>
    // deci are 8 biți total.
    // Folosim cei mai puțini 8 biți din TDATA.
    // ===================================================

    for (int i = 0; i < INPUT_H; i++) {
        for (int j = 0; j < INPUT_W; j++) {
#pragma HLS PIPELINE II=1

            axis_t pkt_in = in_stream.read();

            data_t val;
            val.range(7, 0) = pkt_in.data.range(7, 0);

            input[i][j] = val;
        }
    }

    // Rulează rețeaua CNN
    cnn_forward(input, output);

    // ===================================================
    // Scriere output AXI Stream
    // Ieșirea NU mai este float.
    // Se trimite reprezentarea raw a lui out_t.
    //
    // out_t = ap_fixed<11,6, AP_RND, AP_SAT>
    // deci are 11 biți total.
    // Trimitem valoarea semn-extinsă pe 32 de biți.
    // ===================================================

    for (int i = 0; i < FC2_OUT; i++) {
#pragma HLS PIPELINE II=1

        axis_t pkt_out;

        ap_int<11> raw_out;
        raw_out.range(10, 0) = output[i].range(10, 0);

        ap_int<32> raw32 = raw_out;  // semn-extindere corectă

        pkt_out.data = raw32;
        pkt_out.keep = -1;
        pkt_out.strb = -1;
        pkt_out.last = (i == FC2_OUT - 1) ? 1 : 0;

        out_stream.write(pkt_out);
    }
}