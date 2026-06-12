// Base Plan L1-09 shared constants (synthesizable).
// Golden reference: bw_1g_seed_a / 8 sections / Q3.15 sweep combo.
// Regenerate coeff/l1_09_allpass_coeff_reset.svh when these change.

package base_plan_l1_09_pkg;

    // All-pass IIR geometry (config_base_plan.json: l1_09.allpass.sections = 8)
    localparam int SECTION_COUNT_DEFAULT      = 8;
    localparam int IIR_STATE_SAMPLES_PER_SEC  = 2;

    // I/Q datapath (aligned with L1-08 output / ADC width)
    localparam int DATA_WIDTH_DEFAULT         = 16;

    // Coefficient fixed-point: Q3.15 in 18-bit signed (coeff_total_bits=18, coeff_frac_bits=15)
    localparam int COEFF_WIDTH_DEFAULT        = 18;
    localparam int COEFF_FRAC_BITS_DEFAULT    = 15;
    localparam int COEFF_TOTAL_BITS_DEFAULT   = 18;
    localparam int COEFF_UNITY_INT_DEFAULT    = 1 << COEFF_FRAC_BITS_DEFAULT;

    // Biquad MAC headroom: DATA + COEFF + log2(5 products)
    localparam int ACCUM_EXTRA_BITS_DEFAULT   = 3;
    localparam int ACCUM_WIDTH_DEFAULT =
        DATA_WIDTH_DEFAULT + COEFF_WIDTH_DEFAULT + ACCUM_EXTRA_BITS_DEFAULT;

    // One biquad section per clock in the cascaded wavefront pipeline.
    localparam int IIR_LATENCY_DEFAULT        = SECTION_COUNT_DEFAULT;

endpackage

`ifndef BASE_PLAN_L1_09_COEFF_RESET_SVH
`define BASE_PLAN_L1_09_COEFF_RESET_SVH coeff/l1_09_allpass_coeff_reset.svh
`endif
