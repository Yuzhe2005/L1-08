// Base Plan L1-09 — synthesizable RTL (coeff preload on reset, method A).
//
// Coefficients: async reset loads all SECTION_COUNT SOS rows from
//   coeff/l1_09_allpass_coeff_reset.svh (regenerate from allpass_coefficients_fixed.csv).
//
// Structure: 8-section cascaded 2nd-order all-pass IIR (Direct Form I), real coeffs on I/Q.
// Reference: scipy.signal.sosfilt cold-start, L1_09_fixed_point_quantizer.py (Q3.15).
//
// Pipeline: wavefront — one biquad section per clock, IIR_LATENCY = SECTION_COUNT.

import base_plan_l1_09_pkg::*;

// SOS coefficient bank: loaded once on async reset, then held until next reset.
module l1_09_allpass_coeff_bank #(
    parameter int SECTION_COUNT = SECTION_COUNT_DEFAULT,
    parameter int COEFF_WIDTH   = COEFF_WIDTH_DEFAULT
) (
    input  logic                               clk,
    input  logic                               reset_n,
    output logic                               coeffs_ready,
    output logic signed [COEFF_WIDTH-1:0]      b0 [SECTION_COUNT],
    output logic signed [COEFF_WIDTH-1:0]      b1 [SECTION_COUNT],
    output logic signed [COEFF_WIDTH-1:0]      b2 [SECTION_COUNT],
    output logic signed [COEFF_WIDTH-1:0]      a1 [SECTION_COUNT],
    output logic signed [COEFF_WIDTH-1:0]      a2 [SECTION_COUNT]
);
    logic signed [COEFF_WIDTH-1:0] coeff_b0 [SECTION_COUNT];
    logic signed [COEFF_WIDTH-1:0] coeff_b1 [SECTION_COUNT];
    logic signed [COEFF_WIDTH-1:0] coeff_b2 [SECTION_COUNT];
    logic signed [COEFF_WIDTH-1:0] coeff_a1 [SECTION_COUNT];
    logic signed [COEFF_WIDTH-1:0] coeff_a2 [SECTION_COUNT];

    assign b0           = coeff_b0;
    assign b1           = coeff_b1;
    assign b2           = coeff_b2;
    assign a1           = coeff_a1;
    assign a2           = coeff_a2;
    assign coeffs_ready = reset_n;

    always_ff @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            `include `BASE_PLAN_L1_09_COEFF_RESET_SVH
        end
    end
endmodule

// One DF-I biquad stage: y[n] = (b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]) / 2^FRAC
module iir_biquad_df1_stage #(
    parameter int DATA_W          = DATA_WIDTH_DEFAULT,
    parameter int COEFF_W         = COEFF_WIDTH_DEFAULT,
    parameter int COEFF_FRAC_BITS = COEFF_FRAC_BITS_DEFAULT,
    parameter int ACC_W           = ACCUM_WIDTH_DEFAULT
) (
    input  logic                             clk,
    input  logic                             reset_n,
    input  logic                             stage_en,
    input  logic signed [DATA_W-1:0]         x_in,
    input  logic signed [COEFF_W-1:0]        b0,
    input  logic signed [COEFF_W-1:0]        b1,
    input  logic signed [COEFF_W-1:0]        b2,
    input  logic signed [COEFF_W-1:0]        a1,
    input  logic signed [COEFF_W-1:0]        a2,
    output logic signed [DATA_W-1:0]         y_out
);
    localparam int MULT_W = DATA_W + COEFF_W;

    logic signed [DATA_W-1:0]      x1;
    logic signed [DATA_W-1:0]      x2;
    logic signed [DATA_W-1:0]      y1;
    logic signed [DATA_W-1:0]      y2;
    logic signed [MULT_W-1:0]      prod_b0;
    logic signed [MULT_W-1:0]      prod_b1;
    logic signed [MULT_W-1:0]      prod_b2;
    logic signed [MULT_W-1:0]      prod_a1;
    logic signed [MULT_W-1:0]      prod_a2;
    logic signed [ACC_W-1:0]       acc;
    logic signed [DATA_W-1:0]      y_next;

    function automatic logic signed [DATA_W-1:0] round_sat_qx(
        input logic signed [ACC_W-1:0] value
    );
        logic signed [ACC_W-1:0] rounded;
        logic signed [DATA_W-1:0] truncated;
        begin
            rounded = (COEFF_FRAC_BITS == 0) ? value : (value + (1 <<< (COEFF_FRAC_BITS - 1)));
            truncated = rounded >>> COEFF_FRAC_BITS;
            if (truncated > $signed({1'b0, {DATA_W-1{1'b1}}})) begin
                round_sat_qx = {1'b0, {DATA_W-1{1'b1}}};
            end else if (truncated < $signed({1'b1, {DATA_W-1{1'b0}}})) begin
                round_sat_qx = {1'b1, {DATA_W-1{1'b0}}};
            end else begin
                round_sat_qx = truncated[DATA_W-1:0];
            end
        end
    endfunction

    function automatic logic signed [ACC_W-1:0] sign_extend_prod(
        input logic signed [MULT_W-1:0] value
    );
        return {{(ACC_W - MULT_W){value[MULT_W-1]}}, value};
    endfunction

    always_comb begin
        prod_b0 = x_in * b0;
        prod_b1 = x1   * b1;
        prod_b2 = x2   * b2;
        prod_a1 = y1   * a1;
        prod_a2 = y2   * a2;
        acc = sign_extend_prod(prod_b0)
            + sign_extend_prod(prod_b1)
            + sign_extend_prod(prod_b2)
            - sign_extend_prod(prod_a1)
            - sign_extend_prod(prod_a2);
        y_next = round_sat_qx(acc);
    end

    always_ff @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            x1    <= '0;
            x2    <= '0;
            y1    <= '0;
            y2    <= '0;
            y_out <= '0;
        end else if (stage_en) begin
            y_out <= y_next;
            x2    <= x1;
            x1    <= x_in;
            y2    <= y1;
            y1    <= y_next;
        end
    end
endmodule

// Cascaded SOS pipeline: spatial wavefront, one section per clock of latency.
module iir_sos_cascade #(
    parameter int SECTION_COUNT   = SECTION_COUNT_DEFAULT,
    parameter int DATA_W          = DATA_WIDTH_DEFAULT,
    parameter int COEFF_W         = COEFF_WIDTH_DEFAULT,
    parameter int COEFF_FRAC_BITS = COEFF_FRAC_BITS_DEFAULT,
    parameter int ACC_W           = ACCUM_WIDTH_DEFAULT
) (
    input  logic                             clk,
    input  logic                             reset_n,
    input  logic                             stage_en,
    input  logic signed [DATA_W-1:0]         x_in,
    input  logic signed [COEFF_W-1:0]         b0 [SECTION_COUNT],
    input  logic signed [COEFF_W-1:0]         b1 [SECTION_COUNT],
    input  logic signed [COEFF_W-1:0]         b2 [SECTION_COUNT],
    input  logic signed [COEFF_W-1:0]         a1 [SECTION_COUNT],
    input  logic signed [COEFF_W-1:0]         a2 [SECTION_COUNT],
    output logic signed [DATA_W-1:0]         y_out
);
    logic signed [DATA_W-1:0] stage_in  [SECTION_COUNT];
    logic signed [DATA_W-1:0] stage_out [SECTION_COUNT];

    assign stage_in[0] = x_in;

    for (genvar sec = 0; sec < SECTION_COUNT; sec++) begin : g_sec
        if (sec != 0) begin : g_pipe
            always_ff @(posedge clk or negedge reset_n) begin
                if (!reset_n) begin
                    stage_in[sec] <= '0;
                end else if (stage_en) begin
                    stage_in[sec] <= stage_out[sec - 1];
                end
            end
        end

        iir_biquad_df1_stage #(
            .DATA_W(DATA_W),
            .COEFF_W(COEFF_W),
            .COEFF_FRAC_BITS(COEFF_FRAC_BITS),
            .ACC_W(ACC_W)
        ) u_biquad (
            .clk(clk),
            .reset_n(reset_n),
            .stage_en(stage_en),
            .x_in(stage_in[sec]),
            .b0(b0[sec]),
            .b1(b1[sec]),
            .b2(b2[sec]),
            .a1(a1[sec]),
            .a2(a2[sec]),
            .y_out(stage_out[sec])
        );
    end

    assign y_out = stage_out[SECTION_COUNT-1];
endmodule

module L1_09 #(
    parameter int SECTION_COUNT   = SECTION_COUNT_DEFAULT,
    parameter int DATA_WIDTH      = DATA_WIDTH_DEFAULT,
    parameter int COEFF_WIDTH     = COEFF_WIDTH_DEFAULT,
    parameter int COEFF_FRAC_BITS = COEFF_FRAC_BITS_DEFAULT,
    parameter int ACCUM_EXTRA_BITS = 3,
    parameter int ACCUM_WIDTH     = DATA_WIDTH + COEFF_WIDTH + ACCUM_EXTRA_BITS,
    parameter int IIR_LATENCY     = IIR_LATENCY_DEFAULT
) (
    input  logic                             clk,
    input  logic                             reset_n,

    input  logic signed [DATA_WIDTH-1:0]     i_in,
    input  logic signed [DATA_WIDTH-1:0]     q_in,
    input  logic                             in_valid,

    output logic signed [DATA_WIDTH-1:0]     o_i,
    output logic signed [DATA_WIDTH-1:0]     o_q,
    output logic                             o_valid,

    input  logic                             bypass
);
    logic signed [COEFF_WIDTH-1:0] coeff_b0 [SECTION_COUNT];
    logic signed [COEFF_WIDTH-1:0] coeff_b1 [SECTION_COUNT];
    logic signed [COEFF_WIDTH-1:0] coeff_b2 [SECTION_COUNT];
    logic signed [COEFF_WIDTH-1:0] coeff_a1 [SECTION_COUNT];
    logic signed [COEFF_WIDTH-1:0] coeff_a2 [SECTION_COUNT];

    logic signed [DATA_WIDTH-1:0]   filt_i;
    logic signed [DATA_WIDTH-1:0]   filt_q;

    logic                           coeffs_ready;
    logic                           run_valid;
    logic                           iir_valid_in;
    logic                           iir_out_valid;
    logic [IIR_LATENCY-1:0]         iir_valid_pipe;

    assign run_valid     = in_valid && coeffs_ready;
    assign iir_valid_in  = run_valid && !bypass;
    assign iir_out_valid = iir_valid_pipe[IIR_LATENCY-1];

    l1_09_allpass_coeff_bank #(
        .SECTION_COUNT(SECTION_COUNT),
        .COEFF_WIDTH(COEFF_WIDTH)
    ) u_coeff_bank (
        .clk(clk),
        .reset_n(reset_n),
        .coeffs_ready(coeffs_ready),
        .b0(coeff_b0),
        .b1(coeff_b1),
        .b2(coeff_b2),
        .a1(coeff_a1),
        .a2(coeff_a2)
    );

    iir_sos_cascade #(
        .SECTION_COUNT(SECTION_COUNT),
        .DATA_W(DATA_WIDTH),
        .COEFF_W(COEFF_WIDTH),
        .COEFF_FRAC_BITS(COEFF_FRAC_BITS),
        .ACC_W(ACCUM_WIDTH)
    ) u_iir_i (
        .clk(clk),
        .reset_n(reset_n),
        .stage_en(run_valid),
        .x_in(i_in),
        .b0(coeff_b0),
        .b1(coeff_b1),
        .b2(coeff_b2),
        .a1(coeff_a1),
        .a2(coeff_a2),
        .y_out(filt_i)
    );

    iir_sos_cascade #(
        .SECTION_COUNT(SECTION_COUNT),
        .DATA_W(DATA_WIDTH),
        .COEFF_W(COEFF_WIDTH),
        .COEFF_FRAC_BITS(COEFF_FRAC_BITS),
        .ACC_W(ACCUM_WIDTH)
    ) u_iir_q (
        .clk(clk),
        .reset_n(reset_n),
        .stage_en(run_valid),
        .x_in(q_in),
        .b0(coeff_b0),
        .b1(coeff_b1),
        .b2(coeff_b2),
        .a1(coeff_a1),
        .a2(coeff_a2),
        .y_out(filt_q)
    );

    always_ff @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            iir_valid_pipe <= '0;
            o_i            <= '0;
            o_q            <= '0;
            o_valid        <= 1'b0;
        end else begin
            o_valid <= 1'b0;

            iir_valid_pipe <= {iir_valid_pipe[IIR_LATENCY-2:0], iir_valid_in};

            if (in_valid && coeffs_ready && bypass) begin
                o_i     <= i_in;
                o_q     <= q_in;
                o_valid <= 1'b1;
            end else if (iir_out_valid) begin
                o_i     <= filt_i;
                o_q     <= filt_q;
                o_valid <= 1'b1;
            end
        end
    end
endmodule
