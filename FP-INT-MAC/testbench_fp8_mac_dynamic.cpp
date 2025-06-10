#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>
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
    const int num_tests = 100;
    int num_pass = 0, num_fail = 0;

    std::random_device rd;
    std::mt19937 gen(rd());

    // Generate FP8 values in a reasonable range for valid (normalized) encoding
    std::uniform_int_distribution<int> fp8_dist(56, 127); // bias-7, avoids denorms/NaNs
    // std::uniform_int_distribution<int> fp8_dist(194, 255); // bias-7, avoids denorms/NaNs
    std::uniform_int_distribution<int> int4_dist(-8, 7);  // Signed INT4 range

    const ap_uint<4> precision = 4;
    const ap_uint<4> exp_set = 7;
    const double scale = std::pow(2.0, static_cast<int>(exp_set) - 7) / 8.0;

    for (int test_id = 0; test_id < num_tests; ++test_id) {
        // Streams and control
        hls::stream<ap_uint<8>> act_fp8;
        hls::stream<bool> w_bit;
        ap_int<16> accum = 0;
        bool valid = true;
        bool ready = false;
        bool out_valid = false;

        // Generate random inputs
        int raw_fp8 = fp8_dist(gen);
        int raw_int = int4_dist(gen);
        ap_int<4> int_val = raw_int;

        act_fp8.write(raw_fp8);
        for (int i = 0; i < 4; ++i)
            w_bit.write(int_val[i]);

        // Decode FP8
        bool fp_sign = raw_fp8 >> 7;
        ap_uint<4> fp_exp = (raw_fp8 >> 3) & 0b1111;
        ap_uint<3> fp_mant = raw_fp8 & 0b111;

        double mant_val = 1.0 + static_cast<double>(fp_mant) / 8.0;
        int exp_val = static_cast<int>(fp_exp) - 7;
        double fp_val = std::ldexp(mant_val, exp_val);
        if (fp_sign) fp_val = -fp_val;

        double expected = fp_val * static_cast<double>(raw_int);
        bool output_received = false;

        // Run simulation
        for (int cycle = 0; cycle < 12; ++cycle) {
            fp8_mac_pipeline(act_fp8, w_bit, exp_set, precision, accum, valid, ready, out_valid);

            if (out_valid) {
                double scaled_result = static_cast<double>(accum) * scale;
                bool pass = std::abs(scaled_result - expected) < 1e-3;

                std::cout << "\n===== Test Case " << test_id << " =====\n";
                std::cout << std::fixed << std::setprecision(6);
                std::cout << "  Raw FP8           : 0x" << std::hex << raw_fp8 << std::dec
                          << "   INT4: " << raw_int << "\n";
                std::cout << "  Decoded FP8       : " << fp_val << "\n";
                std::cout << "  Expected Result   : " << expected << "\n";
                std::cout << "  Output (scaled)   : " << scaled_result
                          << "   Raw Accum: " << accum
                          << "   Bin: " << accum.to_string(2, true) << "\n";
                std::cout << (pass ? "  ✅ PASS\n" : "  ❌ FAIL\n");

                if (pass) ++num_pass;
                else ++num_fail;

                output_received = true;
                break;
            }
        }

        if (!output_received) {
            std::cout << "\n===== Test Case " << test_id << " =====\n";
            std::cout << "❌ FAIL: No output received after 12 cycles.\n";
            std::cout << "  FP8: 0x" << std::hex << raw_fp8 << std::dec
                      << "   INT4: " << raw_int << "\n";
            ++num_fail;
        }
    }

    // Summary
    std::cout << "\n========= TEST SUMMARY =========\n";
    std::cout << "Total tests: " << num_tests << "\n";
    std::cout << "Passed     : " << num_pass << "\n";
    std::cout << "Failed     : " << num_fail << "\n";
    std::cout << "Pass Rate  : " << (100.0 * num_pass / num_tests) << " %\n";

    return 0;
}
