// Base Plan L1-08 shared constants (synthesizable).
// Golden reference: bw_1g_seed_a / tap80 / Q3.13 sweep combo.
// Regenerate coeff/l1_08_fir_coeff_reset.svh when these change.

package base_plan_l1_08_pkg;

    // FIR geometry (config_base_plan_sweep.json: l1_08.tap_num = 80)
    localparam int TAP_NUM_DEFAULT          = 80;
    localparam int GROUP_DELAY_SAMPLES      = (TAP_NUM_DEFAULT - 1) / 2;
    localparam int FIR_SETTLE_SAMPLES       = TAP_NUM_DEFAULT - 1;

    // I/Q datapath (to be aligned with ADC width in integration)
    localparam int DATA_WIDTH_DEFAULT       = 16;

    // Coefficient fixed-point: Q3.13 in 16-bit signed (coeff_total_bits=16, coeff_frac_bits=13)
    localparam int COEFF_WIDTH_DEFAULT      = 16;
    localparam int COEFF_FRAC_BITS_DEFAULT  = 13;
    localparam int COEFF_TOTAL_BITS_DEFAULT = 16;

    // Accumulator headroom: DATA + COEFF + log2(TAP_NUM)
    localparam int ACCUM_EXTRA_BITS_DEFAULT = 7;
    localparam int ACCUM_WIDTH_DEFAULT =
        DATA_WIDTH_DEFAULT + COEFF_WIDTH_DEFAULT + ACCUM_EXTRA_BITS_DEFAULT;

    // Pipelined adder-tree depth for TAP_NUM_DEFAULT (80 -> 40 -> 20 -> 10 -> 5 -> 3 -> 2 -> 1)
    localparam int MAC_LATENCY_DEFAULT = 7;

endpackage

// `include must use a macro path (not a package string).
`ifndef BASE_PLAN_L1_08_COEFF_RESET_SVH
`define BASE_PLAN_L1_08_COEFF_RESET_SVH coeff/l1_08_fir_coeff_reset.svh
`endif
