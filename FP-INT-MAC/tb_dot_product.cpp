#include <iostream>
#include <iomanip>
#include <cmath>
#include "ap_int.h"
#include "hls_stream.h"

// DUT
void fp8_mac_pipeline(
    hls::stream<ap_uint<8>>& act_fp8,
    hls::stream<bool>& w_bit,
    ap_uint<4> exp_set,
    ap_uint<4> precision,
    ap_int<16>& accum,
    bool valid,
    bool& ready,
    bool& out_valid
);

// Decode E4M3 FP8 value
float decode_fp8(ap_uint<8> val) {
    bool sign = val[7];
    int exponent = int(val.range(6, 3)) - 7;
    float mant = 1.0f + float(val.range(2, 0)) / 8.0f;
    float result = std::ldexp(mant, exponent);
    return sign ? -result : result;
}

int main() {
    hls::stream<ap_uint<8>> act_fp8;
    hls::stream<bool> w_bit;
    ap_int<16> accum = 0;

    ap_uint<4> precision = 4;
    ap_uint<4> exp_set = 7;
    bool valid = true;
    bool ready = false;
    bool out_valid = false;

    double scale = std::pow(2.0, int(exp_set) - 7) / 8.0;

    // Example dot product: [-1.0, 0.5, -2.0] × [2, -1, 3]
    ap_uint<8> fp8_vals[] = {
        0b10111000, // -1.0
        0b00110100, //  0.5
        0b11000000  // -2.0
    };
    ap_int<4> int_vals[] = {
        2, -1, 3
    };
    const int N = 3;

    // Decode and compute expected result
    double expected = 0.0;
    for (int i = 0; i < N; ++i)
        expected += decode_fp8(fp8_vals[i]) * int_vals[i];

    std::cout << "Expected result = " << expected << "\n";

    // Feed inputs to DUT, one FP8 + INT4 pair at a time
    for (int i = 0; i < N; ++i) {
        std::cout << "\n--- Feeding pair " << i << ": FP8 = 0x" << std::hex << int(fp8_vals[i]) << ", INT = " << std::dec << int_vals[i] << "\n";

        act_fp8.write(fp8_vals[i]);
        for (int b = 0; b < 4; ++b)
            w_bit.write(int_vals[i][b]);

        // Simulate until out_valid is true
        bool done = false;
        for (int cycle = 0; cycle < 20; ++cycle) {
            fp8_mac_pipeline(act_fp8, w_bit, exp_set, precision, accum, valid, ready, out_valid);
            if (out_valid) {
                double result = double(accum) * scale;
                std::cout << "[Cycle " << cycle << "] Accum = " << accum << "   Scaled = " << result << "\n";
                done = true;
                break;
            }
        }

        if (!done) {
            std::cout << "❌ Timeout waiting for output from MAC unit.\n";
            return 1;
        }
    }

    double final_result = double(accum) * scale;
    std::cout << "\n===== FINAL RESULT =====\n";
    std::cout << std::fixed << std::setprecision(6)
              << "Accumulated (scaled): " << final_result << "\n"
              << "Expected             : " << expected << "\n"
              << "Status               : " << (std::abs(final_result - expected) < 1e-3 ? "✅ PASS" : "❌ FAIL") << "\n";

    return 0;
}
