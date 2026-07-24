#ifndef CNN_AXIS_H
#define CNN_AXIS_H

#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <ap_int.h>
#include "cnn.h"

// AXI Stream pe 32 biți.
// Trimitem și primim valori ca float pe stream,
// apoi în cnn_axis se face conversia către data_t / out_t.
typedef hls::axis<ap_int<32>, 0, 0, 0> axis_t;

void cnn_axis(
    hls::stream<axis_t>& in_stream,
    hls::stream<axis_t>& out_stream
);

#endif