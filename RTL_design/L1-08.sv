// Base Plan L1-08 — synthesizable RTL (coeff preload on reset, method A).
//
// Coefficients: async reset loads all TAP_NUM values from
//   coeff/l1_08_fir_coeff_reset.svh (regenerate from h2_fir_coefficients_fixed.csv).
// Product flow: NVM tables are baked into per-tape-out / per-workpoint reset images.
//
// MAC datapath: parallel multiply + pipelined adder tree (see L1-08_planning/L1-08.md).
// L1-08 o_* is intended to feed L1-09 later.

import base_plan_l1_08_pkg::*;

// Coefficient register bank: loaded once on async reset, then held until next reset.
module l1_08_fir_coeff_bank #(
    parameter int TAP_NUM     = TAP_NUM_DEFAULT,
    parameter int COEFF_WIDTH = COEFF_WIDTH_DEFAULT
) (
    input  logic                               clk,
    input  logic                               reset_n,
    output logic                               coeffs_ready,
    output logic signed [COEFF_WIDTH-1:0]      coeff [TAP_NUM]
);
    logic signed [COEFF_WIDTH-1:0] coeff_mem [TAP_NUM];

    assign coeff        = coeff_mem;
    assign coeffs_ready = reset_n;

    always_ff @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            `include `BASE_PLAN_L1_08_COEFF_RESET_SVH
        end
    end
endmodule

// Parallel multiply + pipelined binary adder tree.
// Latency: MAC_LATENCY cycles from valid sample presentation to y_out update.
module fir_mac #(
    parameter int TAP_NUM   = TAP_NUM_DEFAULT,
    parameter int DATA_W    = DATA_WIDTH_DEFAULT,
    parameter int COEFF_W   = COEFF_WIDTH_DEFAULT,
    parameter int ACC_W     = ACCUM_WIDTH_DEFAULT,
    parameter int MAC_LATENCY = MAC_LATENCY_DEFAULT
) (
    input  logic                             clk,
    input  logic                             reset_n,
    input  logic signed [DATA_W-1:0]         samples [TAP_NUM],
    input  logic signed [COEFF_W-1:0]        coeffs  [TAP_NUM],
    output logic signed [ACC_W-1:0]          y_out
);
    localparam int MULT_W = DATA_W + COEFF_W;

    // Level sizes for TAP_NUM=80: 80 -> 40 -> 20 -> 10 -> 5 -> 3 -> 2 -> 1
    localparam int N1 = (TAP_NUM + 1) / 2;
    localparam int N2 = (N1 + 1) / 2;
    localparam int N3 = (N2 + 1) / 2;
    localparam int N4 = (N3 + 1) / 2;
    localparam int N5 = (N4 + 1) / 2;
    localparam int N6 = (N5 + 1) / 2;

    logic signed [MULT_W-1:0]    mult_result [TAP_NUM];
    logic signed [ACC_W-1:0]     sum_l0 [TAP_NUM];
    logic signed [ACC_W-1:0]     sum_l1 [N1];
    logic signed [ACC_W-1:0]     sum_l2 [N2];
    logic signed [ACC_W-1:0]     sum_l3 [N3];
    logic signed [ACC_W-1:0]     sum_l4 [N4];
    logic signed [ACC_W-1:0]     sum_l5 [N5];
    logic signed [ACC_W-1:0]     sum_l6 [N6];

    function automatic logic signed [ACC_W-1:0] sign_extend_mult(
        input logic signed [MULT_W-1:0] value
    );
        return {{(ACC_W - MULT_W){value[MULT_W-1]}}, value};
    endfunction

    always_comb begin
        for (int tap = 0; tap < TAP_NUM; tap++) begin
            mult_result[tap] = samples[tap] * coeffs[tap];
        end
    end

    always_ff @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            for (int k = 0; k < TAP_NUM; k++) sum_l0[k] <= '0;
            for (int k = 0; k < N1;     k++) sum_l1[k] <= '0;
            for (int k = 0; k < N2;     k++) sum_l2[k] <= '0;
            for (int k = 0; k < N3;     k++) sum_l3[k] <= '0;
            for (int k = 0; k < N4;     k++) sum_l4[k] <= '0;
            for (int k = 0; k < N5;     k++) sum_l5[k] <= '0;
            for (int k = 0; k < N6;     k++) sum_l6[k] <= '0;
            y_out <= '0;
        end else begin
            for (int k = 0; k < TAP_NUM; k++) begin
                sum_l0[k] <= sign_extend_mult(mult_result[k]);
            end

            for (int k = 0; k < N1; k++) begin
                if ((2 * k + 1) < TAP_NUM) begin
                    sum_l1[k] <= sum_l0[2 * k] + sum_l0[2 * k + 1];
                end else begin
                    sum_l1[k] <= sum_l0[2 * k];
                end
            end

            for (int k = 0; k < N2; k++) begin
                if ((2 * k + 1) < N1) begin
                    sum_l2[k] <= sum_l1[2 * k] + sum_l1[2 * k + 1];
                end else begin
                    sum_l2[k] <= sum_l1[2 * k];
                end
            end

            for (int k = 0; k < N3; k++) begin
                if ((2 * k + 1) < N2) begin
                    sum_l3[k] <= sum_l2[2 * k] + sum_l2[2 * k + 1];
                end else begin
                    sum_l3[k] <= sum_l2[2 * k];
                end
            end

            for (int k = 0; k < N4; k++) begin
                if ((2 * k + 1) < N3) begin
                    sum_l4[k] <= sum_l3[2 * k] + sum_l3[2 * k + 1];
                end else begin
                    sum_l4[k] <= sum_l3[2 * k];
                end
            end

            for (int k = 0; k < N5; k++) begin
                if ((2 * k + 1) < N4) begin
                    sum_l5[k] <= sum_l4[2 * k] + sum_l4[2 * k + 1];
                end else begin
                    sum_l5[k] <= sum_l4[2 * k];
                end
            end

            for (int k = 0; k < N6; k++) begin
                if ((2 * k + 1) < N5) begin
                    sum_l6[k] <= sum_l5[2 * k] + sum_l5[2 * k + 1];
                end else begin
                    sum_l6[k] <= sum_l5[2 * k];
                end
            end

            if (N6 == 1) begin
                y_out <= sum_l6[0];
            end else begin
                y_out <= sum_l6[0] + sum_l6[1];
            end
        end
    end

endmodule

module L1_08 #(
    parameter int TAP_NUM          = TAP_NUM_DEFAULT,
    parameter int DATA_WIDTH       = DATA_WIDTH_DEFAULT,
    parameter int COEFF_WIDTH      = COEFF_WIDTH_DEFAULT,
    parameter int COEFF_FRAC_BITS  = COEFF_FRAC_BITS_DEFAULT,
    parameter int ACCUM_EXTRA_BITS = $clog2(TAP_NUM),
    parameter int ACCUM_WIDTH      = DATA_WIDTH + COEFF_WIDTH + ACCUM_EXTRA_BITS,
    parameter int MAC_LATENCY      = MAC_LATENCY_DEFAULT
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
    localparam int SettleSamples = TAP_NUM - 1;

    logic signed [DATA_WIDTH-1:0]  x_i [TAP_NUM];
    logic signed [DATA_WIDTH-1:0]  x_q [TAP_NUM];
    logic signed [COEFF_WIDTH-1:0] tap_coeff [TAP_NUM];
    logic signed [ACCUM_WIDTH-1:0] acc_i;
    logic signed [ACCUM_WIDTH-1:0] acc_q;

    logic                          coeffs_ready;
    logic                          run_valid;
    logic                          mac_valid_in;
    logic                          mac_out_valid;
    logic [MAC_LATENCY-1:0]        mac_valid_pipe;
    logic [$clog2(TAP_NUM+1)-1:0]  sample_count;

    assign run_valid     = in_valid && coeffs_ready;
    assign mac_valid_in  = run_valid && (sample_count >= SettleSamples) && !bypass;
    assign mac_out_valid = mac_valid_pipe[MAC_LATENCY-1];

    l1_08_fir_coeff_bank #(
        .TAP_NUM(TAP_NUM),
        .COEFF_WIDTH(COEFF_WIDTH)
    ) u_coeff_bank (
        .clk(clk),
        .reset_n(reset_n),
        .coeffs_ready(coeffs_ready),
        .coeff(tap_coeff)
    );

    fir_mac #(
        .TAP_NUM(TAP_NUM),
        .DATA_W(DATA_WIDTH),
        .COEFF_W(COEFF_WIDTH),
        .ACC_W(ACCUM_WIDTH),
        .MAC_LATENCY(MAC_LATENCY)
    ) u_mac_i (
        .clk(clk),
        .reset_n(reset_n),
        .samples(x_i),
        .coeffs(tap_coeff),
        .y_out(acc_i)
    );

    fir_mac #(
        .TAP_NUM(TAP_NUM),
        .DATA_W(DATA_WIDTH),
        .COEFF_W(COEFF_WIDTH),
        .ACC_W(ACCUM_WIDTH),
        .MAC_LATENCY(MAC_LATENCY)
    ) u_mac_q (
        .clk(clk),
        .reset_n(reset_n),
        .samples(x_q),
        .coeffs(tap_coeff),
        .y_out(acc_q)
    );

    function automatic logic signed [DATA_WIDTH-1:0] round_sat_q15(
        input logic signed [ACCUM_WIDTH-1:0] value
    );
        logic signed [ACCUM_WIDTH-1:0] rounded;
        logic signed [DATA_WIDTH-1:0]  truncated;
        begin
            rounded = (COEFF_FRAC_BITS == 0) ? value : (value + (1 <<< (COEFF_FRAC_BITS - 1)));
            truncated = rounded >>> COEFF_FRAC_BITS;
            if (truncated > $signed({1'b0, {DATA_WIDTH-1{1'b1}}})) begin
                round_sat_q15 = {1'b0, {DATA_WIDTH-1{1'b1}}};
            end else if (truncated < $signed({1'b1, {DATA_WIDTH-1{1'b0}}})) begin
                round_sat_q15 = {1'b1, {DATA_WIDTH-1{1'b0}}};
            end else begin
                round_sat_q15 = truncated[DATA_WIDTH-1:0];
            end
        end
    endfunction

    always_ff @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            for (int tap = 0; tap < TAP_NUM; tap++) begin
                x_i[tap] <= '0;
                x_q[tap] <= '0;
            end
            sample_count   <= '0;
            mac_valid_pipe <= '0;
            o_i            <= '0;
            o_q            <= '0;
            o_valid        <= 1'b0;
        end else begin
            o_valid <= 1'b0;

            if (run_valid) begin
                for (int tap = TAP_NUM - 1; tap > 0; tap--) begin
                    x_i[tap] <= x_i[tap - 1];
                    x_q[tap] <= x_q[tap - 1];
                end
                x_i[0] <= i_in;
                x_q[0] <= q_in;

                if (sample_count < SettleSamples) begin
                    sample_count <= sample_count + 1'b1;
                end
            end

            mac_valid_pipe <= {mac_valid_pipe[MAC_LATENCY-2:0], mac_valid_in};

            if (in_valid && coeffs_ready && bypass) begin
                o_i     <= i_in;
                o_q     <= q_in;
                o_valid <= 1'b1;
            end else if (mac_out_valid) begin
                o_i     <= round_sat_q15(acc_i);
                o_q     <= round_sat_q15(acc_q);
                o_valid <= 1'b1;
            end
        end
    end
endmodule

module base_plan_top #(
    parameter int TAP_NUM           = TAP_NUM_DEFAULT,
    parameter int DATA_WIDTH        = DATA_WIDTH_DEFAULT,
    parameter int COEFF_WIDTH       = COEFF_WIDTH_DEFAULT,
    parameter int COEFF_FRAC_BITS   = COEFF_FRAC_BITS_DEFAULT,
    parameter int L1_09_SECTION_COUNT = 8,
    parameter int L1_09_COEFF_WIDTH   = 18,
    parameter int L1_09_COEFF_FRAC_BITS = 15
) (
    input  logic                             clk,
    input  logic                             reset_n,
    input  logic signed [DATA_WIDTH-1:0]     i_in,
    input  logic signed [DATA_WIDTH-1:0]     q_in,
    input  logic                             in_valid,
    input  logic                             l1_08_bypass,
    input  logic                             l1_09_bypass,
    output logic signed [DATA_WIDTH-1:0]     o_i,
    output logic signed [DATA_WIDTH-1:0]     o_q,
    output logic                             o_valid
);
    logic signed [DATA_WIDTH-1:0] l1_08_o_i;
    logic signed [DATA_WIDTH-1:0] l1_08_o_q;
    logic                         l1_08_o_valid;

    L1_08 #(
        .TAP_NUM(TAP_NUM),
        .DATA_WIDTH(DATA_WIDTH),
        .COEFF_WIDTH(COEFF_WIDTH),
        .COEFF_FRAC_BITS(COEFF_FRAC_BITS)
    ) u_l1_08 (
        .clk(clk),
        .reset_n(reset_n),
        .i_in(i_in),
        .q_in(q_in),
        .in_valid(in_valid),
        .o_i(l1_08_o_i),
        .o_q(l1_08_o_q),
        .o_valid(l1_08_o_valid),
        .bypass(l1_08_bypass)
    );

    L1_09 #(
        .SECTION_COUNT(L1_09_SECTION_COUNT),
        .DATA_WIDTH(DATA_WIDTH),
        .COEFF_WIDTH(L1_09_COEFF_WIDTH),
        .COEFF_FRAC_BITS(L1_09_COEFF_FRAC_BITS)
    ) u_l1_09 (
        .clk(clk),
        .reset_n(reset_n),
        .i_in(l1_08_o_i),
        .q_in(l1_08_o_q),
        .in_valid(l1_08_o_valid),
        .o_i(o_i),
        .o_q(o_q),
        .o_valid(o_valid),
        .bypass(l1_09_bypass)
    );
endmodule
