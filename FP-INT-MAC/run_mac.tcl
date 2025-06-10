# Create and open a new HLS project
open_project mac_project

# Set the top-level function name
set_top fp8_bitserial_mul_signed

# Add design and testbench files
add_files fp8_bitserial_mac.cpp
# add_files -tb testbench_fp8_mac.cpp
# add_files -tb testbench_fp8_mac_dynamic.cpp
add_files -tb tb_dot_product.cpp

# Open a solution and configure settings
open_solution "sol1"
set_part {xc7z020clg400-1}
create_clock -period 10 -name default

# Run C simulation
csim_design

# # Run synthesis
# csynth_design

# # Run C/RTL co-simulation
# cosim_design

# Export RTL (optional)
# export_design -format ip_catalog