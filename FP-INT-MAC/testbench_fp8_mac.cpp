#include <iostream>
#include <iomanip>
#include <cmath>
#include "ap_int.h"
#include "hls_stream.h"

// Declare the top function
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

int main() {
    hls::stream<ap_uint<8>> act_fp8;
    hls::stream<bool> w_bit;
    ap_int<16> accum = 0;

    ap_uint<4> precision = 4;
    ap_uint<4> exp_set = 7;
    bool valid = true;
    bool ready = false;
    bool out_valid = false;

    double scale = std::pow(2.0, static_cast<int>(exp_set) - 7) / 8.0;

    // Test: FP8 = -1.0 (sign=1, exp=7, mant=000), INT = -2
    ap_uint<8> test_fp8 = 0b10111000;
    ap_int<4> test_int = 7;

    // Decode FP8
    bool fp_sign = test_fp8 >> 7;
    ap_uint<4> fp_exp = (test_fp8 >> 3) & 0b1111;
    ap_uint<3> fp_mant = test_fp8 & 0b111;

    double mant_val = 1.0 + static_cast<double>(fp_mant) / 8.0;
    int exp_val = static_cast<int>(fp_exp) - 7;
    double fp_val = std::ldexp(mant_val, exp_val);
    if (fp_sign) fp_val = -fp_val;

    double expected = fp_val * static_cast<double>(test_int);

    // Feed FP8 input
    act_fp8.write(test_fp8);

    // Feed INT4 bits
    for (int i = 0; i < 4; ++i) {
        w_bit.write(test_int[i]);
        std::cout << "Write INT bit " << i << ": " << test_int[i] << "\n";
    }

    // Simulate for cycles
    for (int cycle = 0; cycle < 10; ++cycle) {
        fp8_mac_pipeline(act_fp8, w_bit, exp_set, precision, accum, valid, ready, out_valid);

        if (out_valid) {
            double scaled_result = static_cast<double>(accum) * scale;

            std::cout << std::fixed << std::setprecision(6);
            std::cout << "\n[Cycle " << cycle << "]\n";
            std::cout << "  Accumulated (scaled): " << scaled_result
                      << "   Raw: " << accum << "   Bin: " << accum.to_string(2, true) << "\n";
            std::cout << "  FP8 decoded          : " << fp_val << "\n";
            std::cout << "  INT value            : " << static_cast<int>(test_int) << "\n";
            std::cout << "  Expected             : " << expected << "\n";

            bool pass = std::abs(scaled_result - expected) < 1e-3;
            std::cout << "  Status               : " << (pass ? "✅ PASS" : "❌ FAIL") << "\n";
            return pass ? 0 : 1;
        }
    }

    std::cout << "❌ FAIL: Output not ready after 10 cycles.\n";
    return 1;
}
