#include "ap_int.h"
#include "hls_stream.h"

// ---------- Struct for multiplier output ----------
struct MultResult {
    ap_int<16> acc;          // 13.3 fixed-point product
    ap_uint<4> exp;          // Exponent from FP8
    bool fp_sign;            // Sign of FP8
    bool int_sign;           // Sign bit of INT
    ap_uint<4> mant_out;     // Mantissa in fixed-point (1.m), original (1 << 3) | mant
    ap_uint<4> precision;    // INT bitwidth
};

// ---------- Multiplier: FP8 x INT[precision] ----------
void fp8_bitserial_mul_signed(
    hls::stream<ap_uint<8>>& act_fp8,         // FP8 input
    hls::stream<bool>& w_bit,                 // INT bit-serial input (LSB to MSB)
    hls::stream<MultResult>& result_out,      // Output result
    ap_uint<4> precision,                     // INT bitwidth
    bool valid,                               // Start signal
    bool& ready                               // Ready for next input
) {
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS PIPELINE off

    static enum {IDLE, COMPUTE, DONE} state = IDLE;
    static ap_uint<8> act_reg;
    static ap_uint<16> fixed_mant;
    static ap_uint<4> exponent;
    static ap_uint<3> mant;                  // Store mantissa bits
    static ap_uint<4> count = 0;
    static ap_int<16> acc = 0;
    static bool int_sign = false;
    static bool fp_sign = false;

    // ✅ Now always reflect whether the unit is ready for a new input
    ready = (state == IDLE);

    switch (state) {
        case IDLE:
            // ✅ Accept input immediately when ready and valid
            if (valid && !act_fp8.empty()) {
                act_reg = act_fp8.read();
                fp_sign = act_reg[7];
                exponent = act_reg.range(6, 3);
                mant = act_reg.range(2, 0);                  // Save original mant
                fixed_mant = (1 << 3) | mant;                // 1.m = 8 + mant

                acc = 0;
                count = 0;
                state = COMPUTE;
            }
            break;

        case COMPUTE:
            if (count < precision && !w_bit.empty()) {
                bool bit = w_bit.read();
                if (count == precision - 1) {
                    int_sign = bit;
                } else {
                    if (bit) acc += fixed_mant;
                    fixed_mant <<= 1;
                }
                count++;
            }

            if (count == precision) {
                state = DONE;
            }
            break;

        case DONE: {
            MultResult result;
            result.acc = acc;
            result.exp = exponent;
            result.fp_sign = fp_sign;
            result.int_sign = int_sign;
            result.mant_out = (1 << 3) | mant;
            result.precision = precision;
            result_out.write(result);
            state = IDLE;  // ✅ Go back to IDLE immediately
            break;
        }
    }
}


// ---------- Accumulator ----------
void fp_accumulator(
    hls::stream<MultResult>& result_in,
    ap_uint<4> exp_set,
    ap_int<16>& accum,
    bool& out_valid
) {
#pragma HLS PIPELINE II=1
    if (!result_in.empty()) {
        MultResult res = result_in.read();

        ap_int<16> val = res.acc;

        // Correction for int_sign
        if (res.int_sign) {
            ap_int<16> correction = ((1 << 3) | res.mant_out) << (res.precision - 1);
            val -= correction;
        }

        // Correction for fp_sign
        if (res.fp_sign) {
            val = (~val) + 1;
        }

        // Align exponent
        int shift = int(res.exp) - int(exp_set);
        ap_int<16> aligned_val = (shift >= 0) ? (val << shift) : (val >> -shift);

        accum += aligned_val;
        out_valid = true;
    } else {
        out_valid = false;
    }
}

// ---------- Top-level pipeline ----------
void fp8_mac_pipeline(
    hls::stream<ap_uint<8>>& act_fp8,
    hls::stream<bool>& w_bit,
    ap_uint<4> exp_set,
    ap_uint<4> precision,
    ap_int<16>& accum,
    bool valid,
    bool& ready,
    bool& out_valid
) {
#pragma HLS DATAFLOW
    static hls::stream<MultResult> result_stream("result_stream");
#pragma HLS STREAM variable=result_stream depth=4

    fp8_bitserial_mul_signed(act_fp8, w_bit, result_stream, precision, valid, ready);
    fp_accumulator(result_stream, exp_set, accum, out_valid);
}
