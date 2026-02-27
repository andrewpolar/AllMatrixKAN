// License
// If the end user somehow manages to make billions of US dollars using this code,
// and happens to meet the developer begging for change outside a McDonald's,
// he or she is under no obligation to buy the developer a sandwich.

// Symmetry Clause
// Likewise, if the developer becomes rich and famous by publishing this code,
// and meets an unfortunate end user who went bankrupt using it,
// the developer is also under no obligation to buy the end user a sandwich.

//Code by Andrew Polar, concept Andrew Polar and Mike Poluektov.
//Training KAN to predict determinants of random 3 by 3 matrices. 
//Relative error for unseen data for this truncated version is below 2%.
//All integer solition for FPGA board. 

#include <stdio.h>
#include <math.h>
#include <cstdint>
#include <memory>
#include <vector>
#include <random>
#include <ctime>

//random digits generators
static uint16_t seed[9] = { 0xACE1, 0xBEEF, 0x1234, 0x4321, 0xA777, 0xAAAA, 0x1F2F, 0xCAFE, 0xBABE };
static uint16_t seed16 = 0xDA0;

static inline uint16_t lfsr16(uint16_t* state) {
	uint16_t bit = ((*state >> 15) ^ (*state >> 13) ^ (*state >> 12) ^ (*state >> 10)) & 1;
	*state = (uint16_t)((*state << 1) | bit);
	if (*state == 0) *state = 1;
	return *state;
}

static inline uint8_t rnd_byte(uint16_t* state) {
	uint8_t v = (uint8_t)(lfsr16(state) & 0xFF);
	if (v == 0)   v = 1;
	if (v == 255) v = 254;
	return v;
}

static inline int16_t det3x3_int(const uint8_t m[9]) {
	int32_t a = m[0], b = m[1], c = m[2];
	int32_t d = m[3], e = m[4], f = m[5];
	int32_t g = m[6], h = m[7], i = m[8];
	int32_t det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
	det >>= 10;
	return (int16_t)det;
}

static inline int32_t rnd_range_fast(uint16_t* state, int32_t left, int32_t right) {
	uint32_t span = (uint32_t)(right - left + 1);
	uint32_t r = lfsr16(state);        
	uint32_t scaled = (r * span) >> 16;   
	return left + (int32_t)scaled;
}

//KAN's functions
int Compute(int diff, int delta_shift, const std::vector<int>& row, int& offset, int& index) {
	index = diff >> delta_shift;
	offset = diff & ((1 << delta_shift) - 1);
	long long Q = row[index + 1] - row[index];
	Q *= offset;

	Q >>= delta_shift;
	Q += row[index];
	return (int)Q;
}

int GetInnerResidual(int outer_residual, int outer_shift, const std::vector<int>& row, int index) {
	long long product = (row[index + 1] - row[index]) * outer_residual;
	return (int)(product >> outer_shift);
}

int ReduceVector(const std::vector<int>& V, int mult, int denom, int min, int max) {
	long long x = 0;
	for (short i = 0; i < (short)V.size(); ++i) {
		x += V[i];
	}
	x *= mult;
	x >>= denom;
	if (x <= min) x = min + 1;
	if (x >= max) x = max - 1;
	return (int)x;
}

int main() {
	//data
	const int nFeatures = 9;  //features are one byte long
	const int FeatureMin = 0;
	const int TargetMin = -20'480;
	const int TargetMax = 20'480;

	//network inner
	const int nInner = 6;  //this is what is blocking more accurate model. it is picked to match RTL version 
	const int nInnerPoints = 3;
	const int nInnerShift = 7;      //interval [0,255], 3 points, 2 linear segments, 1 << 7 = 128 lenght of the segment
	const int nInnerLearningShift = 4;
	//this is division by number of features = 9, multiplication by 113 and division by 1 << 10
	const short MultInner = 113;
	const short DenomInner = 10;

	//network outer
	const int nOuter = 1;
	const int nOuterPoints = 21;
	const int nOuterShift = 11;  //interval [-20'480, 20480], 21 points, 20 linear segments, 1 << 11 = 2048 length of the segment
	const int nOuterLearningShift = 3;
	//this is division by number of inner blocks
	const short MultOuter = 21;
	const short DenomOuter = 7;

	//data size
	const int nProcessedRecords = 50'000;

	clock_t start_application = clock();
	clock_t current_time = clock();

	//initialize models
	//int range = (TargetMax - TargetMin) >> 2;
	//int mean = (TargetMax + TargetMin) >> 1;
	//int left = mean - range;
	//int right = mean + range;

	int left = -10240;
	int right = 10240;

	std::vector<std::vector<int>> INNER(nInner * nFeatures, std::vector<int>(nInnerPoints, 0));
	auto innerOffset = std::vector<int>(nInner * nFeatures);
	auto innerIndex = std::vector<int>(nInner * nFeatures);
	for (int i = 0; i < nInner * nFeatures; ++i) {
		for (int j = 0; j < nInnerPoints; ++j) {
			INNER[i][j] = rnd_range_fast(&seed16, left, right);
		}
	}

	std::vector<std::vector<int>> OUTER(nInner, std::vector<int>(nOuterPoints, 0));
	auto outerOffset = std::vector<int>(nInner);
	auto outerIndex = std::vector<int>(nInner);
	for (int i = 0; i < nInner; ++i) {
		for (int j = 0; j < nOuterPoints; ++j) {
			OUTER[i][j] = rnd_range_fast(&seed16, left, right);
		}
	}

	//auxiliary buffers
	uint8_t features[nFeatures];
	std::vector<int> error_buffer(1 << 8);
	std::vector<std::vector<int>> intermediate_matrix(nInner, std::vector<int>(nFeatures, 0));
	std::vector<int> intermediate_vector(nInner);
	std::vector<int> outer_vector(nInner);
	std::vector<int> inner_residual(nInner);
	std::vector<int> delta_F1(nInner);
	std::vector<int> delta_F2(nInner);
	std::vector<std::vector<int>> delta_F1_matrix(nInner, std::vector<int>(nFeatures, 0));
	std::vector<std::vector<int>> delta_F2_matrix(nInner, std::vector<int>(nFeatures, 0));
	int predicted;
	int outer_residual;

	int cnt = 0;
	int actualTargetMin = -1;
	int actualTargetMax = 1;
	while (true) {
		for (int k = 0; k < nFeatures; k++) {
			features[k] = rnd_byte(&seed[k]);
		}
		int target = det3x3_int(features);
		if (target < actualTargetMin) actualTargetMin = target;
		if (target > actualTargetMax) actualTargetMax = target;

		//1 FPGA cycle
		for (int k = 0; k < nInner; ++k) {
			for (int j = 0; j < nFeatures; ++j) {
				int m = k * nFeatures + j;
				intermediate_matrix[k][j] = Compute(features[j] - FeatureMin, nInnerShift, INNER[m], innerOffset[m], innerIndex[m]);
			}
		}
		//1 FPGA cycle
		for (int k = 0; k < nInner; ++k) {
			intermediate_vector[k] = ReduceVector(intermediate_matrix[k], MultInner, DenomInner, TargetMin, TargetMax);
		}
		//1 FPGA cycle
		for (int j = 0; j < nInner; ++j) {
			outer_vector[j] = Compute(intermediate_vector[j] - TargetMin, nOuterShift, OUTER[j], outerOffset[j], outerIndex[j]);
		}

		//1 FPGA cycle
		predicted = ReduceVector(outer_vector, MultOuter, DenomOuter, TargetMin, TargetMax);

		//here forward pass is completed and backprop started

		//1 FPGA cycle
		outer_residual = target - predicted;
		error_buffer[cnt & 0xff] = target - predicted;

		//1 FPGA cycle
		for (int j = 0; j < nInner; ++j) {
			inner_residual[j] = GetInnerResidual(outer_residual, nOuterShift, OUTER[j], outerIndex[j]);
		}

		//1 FPGA cycle
		for (int j = 0; j < nInner; ++j) {
			delta_F1[j] = ((outer_residual >> nOuterLearningShift) * outerOffset[j]) >> nOuterShift;
		}

		//1 FPGA cycle
		for (int j = 0; j < nInner; ++j) {
			delta_F2[j] = (outer_residual >> nOuterLearningShift) - delta_F1[j];
		}

		//1 FPGA cycle
		for (int k = 0; k < nInner; ++k) {
			for (int j = 0; j < nFeatures; ++j) {
				delta_F1_matrix[k][j] = ((inner_residual[k] >> nInnerLearningShift) * innerOffset[k * nFeatures + j]) >> nInnerShift;
			}
		}

		//1 FPGA cycle
		for (int k = 0; k < nInner; ++k) {
			for (int j = 0; j < nFeatures; ++j) {
				delta_F2_matrix[k][j] = (inner_residual[k] >> nInnerLearningShift) - delta_F1_matrix[k][j];
			}
		}

		//here backprop is comleted and update started

		//1 FPGA cycle
		for (int j = 0; j < nInner; ++j) {
			OUTER[j][outerIndex[j] + 1] += delta_F1[j];
		}

		//1 FPGA cycle
		for (int j = 0; j < nInner; ++j) {
			OUTER[j][outerIndex[j]] += delta_F2[j];
		}

		//1 FPGA cycle
		for (int k = 0; k < nInner; ++k) {
			for (int j = 0; j < nFeatures; ++j) {
				int m = k * nFeatures + j;
				INNER[m][innerIndex[m] + 1] += delta_F1_matrix[k][j];
			}
		}

		//1 FPGA cycle
		for (int k = 0; k < nInner; ++k) {
			for (int j = 0; j < nFeatures; ++j) {
				int m = k * nFeatures + j;
				INNER[m][innerIndex[m]] += delta_F2_matrix[k][j];
			}
		}

		if (++cnt >= nProcessedRecords) break;
	}

	current_time = clock();
	printf("Time %2.3f seconds\n", (double)(current_time - start_application) / CLOCKS_PER_SEC);

	int sum = 0;
	for (int i = 0; i < (1 << 8); ++i) {
		if (error_buffer[i] < 0) sum -= error_buffer[i];
		else sum += error_buffer[i];
	}
	sum /= (1 << 8);
	printf("Averate error %d, target ranges [%d, %d]\n", sum, actualTargetMin, actualTargetMax);
}

//Below is RTL version
////This is training of Kolmogorov-Arnold network.  It only trains 
////to predict determinants of 3 * 3 matrices. 
////RTL code by Andrew Polar. The concept of Mike Poluektov and Andrew Polar. 
////Expected error value 655 for unseen data, which is about 2% relative error.
////The accuracy is limited by the hardware, more accurate model needs more advanced equipment.
////The details can be found on OpenKAN.org. 
//
//module DD33 #(
//    //data
//    parameter shortint N_FEATURES = 9,
//    parameter shortint N_XMIN = 0,
//    parameter shortint N_TARGETMIN = -20480,
//    parameter shortint N_TARGETMAX = 20480,
//
//    //network inner
//    parameter shortint N_INNER = 6,   //this limits the model accuracy, classic KAN needs 19
//    parameter shortint N_INNER_POINTS = 3,
//    parameter shortint N_INNER_SHIFT = 7,
//    parameter shortint N_INNER_LEARNING_SHIFT = 4,
//    parameter shortint MULT_INNER = 113,
//    parameter shortint DENOMINNER = 10,
//
//    //network outer
//    parameter shortint N_OUTER = 1,
//    parameter shortint N_OUTER_POINTS = 21,
//    parameter shortint N_OUTER_SHIFT = 11,
//    parameter shortint N_OUTER_LEARNING_SHIFT = 3,
//    parameter shortint MULT_OUTER = 21,
//    parameter shortint DENOMOUTER = 7,
//
//    parameter shortint LEFT = -10240,
//    parameter shortint RIGHT = 10240,
//
//    //training
//    parameter int N_PROCESSED_RECORDS = 50000
//    )(
//        input  logic        CLK100MHZ,
//        input  logic        CPU_RESETN,
//        output logic[15:0] LED
//        );
//
//logic signed[31:0] intermediate_matrix[0:N_INNER - 1][0:N_FEATURES - 1];
//logic signed[31:0] intermediate_vector[0:N_INNER - 1];
//logic signed[31:0] INNER[0:N_INNER * N_FEATURES - 1][0:N_INNER_POINTS - 1];
//logic signed[31:0] innerOffset[0:N_INNER * N_FEATURES - 1];
//logic signed[31:0] innerIndex[0:N_INNER * N_FEATURES - 1];
//
//logic signed[31:0] outer_vector[0:N_INNER - 1];
//logic signed[31:0] OUTER[0:N_INNER - 1][0:N_OUTER_POINTS - 1];
//logic signed[31:0] outerOffset[0:N_INNER - 1];
//logic signed[31:0] outerIndex[0:N_INNER - 1];
//logic signed[31:0] m;
//logic[7:0] features[0:N_FEATURES - 1];
//logic signed[31:0] predicted;
//logic signed[31:0] outer_residual;
//logic signed[31:0] error_buffer[0:255];
//logic signed[31:0] inner_residual[0:N_INNER - 1];
//logic[7:0] err_wr_ptr;
//
//logic signed[31:0] delta_F1_matrix[0:N_INNER - 1][0:N_FEATURES - 1];
//logic signed[31:0] delta_F2_matrix[0:N_INNER - 1][0:N_FEATURES - 1];
//logic signed[31:0] delta_F1[0:N_INNER - 1];
//logic signed[31:0] delta_F2[0:N_INNER - 1];
//
//logic[15:0] seed[9] = '{
//16'hACE1, 16'hBEEF, 16'h1234,
//16'h4321, 16'hA777, 16'hAAAA,
//16'h1F2F, 16'hCAFE, 16'hBABE
//    };
//    logic[15:0] seed16 = 16'h0DA0;  
//        logic[7:0] rnd_bytes[9];
//    int cnt;
//    logic signed[31:0] actualTargetMin;
//    logic signed[31:0] actualTargetMax;
//    logic signed[31:0] target;
//    logic signed[31:0] sum;
//
//    //    initial begin 
//
//    //        //initialization of models
//    //	    for (int i = 0; i < N_INNER*N_FEATURES; i++) begin
//    //            for (int j = 0; j < N_INNER_POINTS; j++) begin
//    //                INNER[i][j] = rnd_range_fast(seed16, LEFT, RIGHT);
//    //            end
//    //        end
//
//    //        for (int i = 0; i < N_INNER; i++) begin
//    //            for (int j = 0; j < N_OUTER_POINTS; j++) begin
//    //                OUTER[i][j] = rnd_range_fast(seed16, LEFT, RIGHT);
//    //            end
//    //        end
//
//    //        cnt = 0;
//    //        actualTargetMin = -1;
//    //	    actualTargetMax = 1;
//    //	    err_wr_ptr = 0;
//    //        while (1) begin
//    //            //making new record
//    //            rnd9(features);
//    //		    target = det3x3(features);
//
//    //		    if (target < actualTargetMin) actualTargetMin = target;
//    //		    if (target > actualTargetMax) actualTargetMax = target;
//
//    //		    //forward pass
//    //		    for (int k = 0; k < N_INNER; ++k) begin
//    //			    for (int j = 0; j < N_FEATURES; ++j) begin
//    //				    m = k * N_FEATURES + j;
//    //				    intermediate_matrix[k][j] = ComputeInner(features[j] - N_XMIN, N_INNER_SHIFT, INNER[m], innerOffset[m], innerIndex[m]);
//    //			     end
//    //		    end
//
//    //		    for (int k = 0; k < N_INNER; ++k) begin
//    //			    intermediate_vector[k] = ReduceInner(intermediate_matrix[k], MULT_INNER, DENOMINNER, N_TARGETMIN, N_TARGETMAX);
//    //		    end
//
//    //		    for (int j = 0; j < N_INNER; ++j) begin
//    //			     outer_vector[j] = ComputeOuter(intermediate_vector[j] - N_TARGETMIN, N_OUTER_SHIFT, OUTER[j], outerOffset[j], outerIndex[j]);
//    //		    end
//
//    //		    predicted = ReduceOuter(outer_vector, MULT_OUTER, DENOMOUTER, N_TARGETMIN, N_TARGETMAX);
//    //            //end of forward pass
//
//    //            //start of the back propagation
//    //            outer_residual = target - predicted;
//    //            error_buffer[err_wr_ptr] = target - predicted;
//    //            err_wr_ptr = err_wr_ptr + 1;
//
//    //            for (int j = 0; j < N_INNER; ++j) begin
//    //			   inner_residual[j] = GetInnerResidual(outer_residual, N_OUTER_SHIFT, OUTER[j], outerIndex[j]);
//    //		    end
//
//    //		    for (int j = 0; j < N_INNER; ++j) begin
//    //			   delta_F1[j] = ((outer_residual >>> N_OUTER_LEARNING_SHIFT) * outerOffset[j]) >>> N_OUTER_SHIFT;
//    //		    end
//
//    //		    for (int j = 0; j < N_INNER; ++j) begin
//    //                delta_F2[j] = (outer_residual >>> N_OUTER_LEARNING_SHIFT) - delta_F1[j];
//    //            end
//
//    //            for (int k = 0; k < N_INNER; ++k) begin
//    //                for (int j = 0; j < N_FEATURES; ++j) begin
//    //                    m = k * N_FEATURES + j;
//    //                    delta_F1_matrix[k][j] = ((inner_residual[k] >>> N_INNER_LEARNING_SHIFT) * innerOffset[m]) >>> N_INNER_SHIFT;
//    //                end
//    //            end
//
//    //            for (int k = 0; k < N_INNER; ++k) begin
//    //                for (int j = 0; j < N_FEATURES; ++j) begin
//    //                    delta_F2_matrix[k][j] = (inner_residual[k] >>> N_INNER_LEARNING_SHIFT) - delta_F1_matrix[k][j];
//    //                end
//    //            end
//    //            //end of back propagation step
//
//    //            //next is update step
//    //            for (int j = 0; j < N_INNER; ++j) begin
//    //                OUTER[j][outerIndex[j] + 1] += delta_F1[j];
//    //            end
//
//    //            for (int j = 0; j < N_INNER; ++j) begin
//    //                OUTER[j][outerIndex[j]] += delta_F2[j];
//    //            end
//
//    //            for (int k = 0; k < N_INNER; ++k) begin
//    //                for (int j = 0; j < N_FEATURES; ++j) begin
//    //                    m = k * N_FEATURES + j;
//    //                    INNER[m][innerIndex[m] + 1] += delta_F1_matrix[k][j];
//    //                end
//    //            end
//
//    //            for (int k = 0; k < N_INNER; ++k) begin
//    //                for (int j = 0; j < N_FEATURES; ++j) begin
//    //                    m = k * N_FEATURES + j;
//    //                    INNER[m][innerIndex[m]] += delta_F2_matrix[k][j];
//    //                end
//    //            end
//    //            //end update and record processing step
//
//    //		    if (++cnt >= N_PROCESSED_RECORDS) break;
//    //		    if ((cnt & 16'h3FFF) == 0) begin
//    //                $display("Training record: %0d", cnt);
//    //            end
//    //        end //end while loop
//
//    //        sum = AvgAbsError(error_buffer);
//    //        $display("Average error %0d, target ranges [%0d, %0d]", sum, actualTargetMin, actualTargetMax);
//
//    //    end //initial
//
//    function automatic logic[15:0] lfsr16_next(ref logic[15:0] state);
//    logic bitval;
//    bitval = (state >> 15) ^ (state >> 13) ^ (state >> 12) ^ (state >> 10);
//    state = (state << 1) | (bitval & 1'b1);
//        if (state == 16'h0000) state = 16'h0001;
//            return state;
//            endfunction
//
//            function automatic logic[7:0] rnd_byte(ref logic[15:0] state);
//            logic[15:0] s;
//            logic[7:0]  v;
//            s = lfsr16_next(state);
//            v = s[7:0];
//            if (v == 8'd0) v = 8'd1;
//                if (v == 8'd255) v = 8'd254;
//                    return v;
//                    endfunction
//
//                    function automatic void rnd9(output logic[7:0] data[9]);
//                    for (int i = 0; i < 9; i++) begin
//                        data[i] = rnd_byte(seed[i]);
//                    end
//                        endfunction
//
//                        function automatic logic signed[15:0] det3x3(input logic[7:0] m[0:8]);
//                    logic signed[31:0] a, b, c, d, e, f, g, h, i;
//    logic signed[31:0] det;
//    begin
//        a = m[0]; b = m[1]; c = m[2];
//    d = m[3]; e = m[4]; f = m[5];
//    g = m[6]; h = m[7]; i = m[8];
//    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
//    det = det >> > 10;
//    det3x3 = det[15:0];
//    end
//        endfunction
//
//        function automatic logic signed[31:0] ComputeInner(
//            input  logic signed[31:0] diff,
//            input  int delta_shift,
//            input  logic signed[31:0] row[0:N_INNER_POINTS - 1],
//            output logic signed[31:0] offset,
//            output int index);
//
//    logic signed[31:0] diff_row;
//    logic signed[47:0] mult;
//    logic signed[31:0] result;
//
//    index = diff >> > delta_shift;
//    offset = diff & ((1 << delta_shift) - 1);
//
//    diff_row = row[index + 1] - row[index];
//
//    mult = diff_row * offset;
//    result = row[index] + (mult >> > delta_shift);
//
//    return result;
//    endfunction
//
//        function automatic logic signed[31:0] ComputeOuter(
//            input  logic signed[31:0] diff,
//            input  int delta_shift,
//            input  logic signed[31:0] row[0:N_OUTER_POINTS - 1],
//            output logic signed[31:0] offset,
//            output int index);
//
//    logic signed[31:0] diff_row;
//    logic signed[47:0] mult;
//    logic signed[31:0] result;
//
//    index = diff >> > delta_shift;
//    offset = diff & ((1 << delta_shift) - 1);
//
//    diff_row = row[index + 1] - row[index];
//
//    mult = diff_row * offset;
//    result = row[index] + (mult >> > delta_shift);
//
//    return result;
//    endfunction
//
//        function automatic int rnd_range_fast(ref logic[15:0] state, input int left, input int right);
//    logic[31:0] span;
//    logic[31:0] r;
//    logic[31:0] scaled;
//
//    span = right - left + 1;
//    r = lfsr16_next(state);
//    scaled = (r * span) >> 16;
//    return left + int'(scaled);
//        endfunction
//
//        function automatic int ReduceInner(input logic signed[31:0] V[0:N_FEATURES - 1],
//            input int mult, input int denom, input int min_val, input int max_val);
//    longint x;
//    x = 0;
//    for (int i = 0; i < N_FEATURES; i++) begin
//        x += V[i];
//    end
//        x *= mult;
//    x >>>= denom;
//    if (x <= min_val) x = min_val + 1;
//    if (x >= max_val) x = max_val - 1;
//    return int'(x);
//        endfunction
//
//        function automatic int ReduceOuter(input logic signed[31:0] V[0:N_INNER - 1],
//            input int mult, input int denom, input int min_val, input int max_val);
//    longint x;
//    x = 0;
//    for (int i = 0; i < N_INNER; i++) begin
//        x += V[i];
//    end
//        x *= mult;
//    x >>>= denom;
//    if (x <= min_val) x = min_val + 1;
//    if (x >= max_val) x = max_val - 1;
//    return int'(x);
//        endfunction
//
//        function automatic int GetInnerResidual(input int outer_residual, input int outer_shift,
//            input logic signed[31:0] row[0:N_OUTER_POINTS - 1], input int index);
//    longint product;
//    begin
//        product = (row[index + 1] - row[index]) * outer_residual;
//    GetInnerResidual = int'(product >>> outer_shift); 
//        end
//        endfunction
//
//        function automatic logic signed[31:0] AvgAbsError(input logic signed[31:0] error_buffer[0:255]);
//    logic signed[31:0] sum;
//    begin
//        sum = 0;
//    for (int i = 0; i < 256; i++) begin
//        if (error_buffer[i] < 0)
//            sum += -error_buffer[i];
//        else
//            sum += error_buffer[i];
//    end
//        AvgAbsError = sum >> > 8;
//    end
//        endfunction
//
//        typedef enum logic[7:0]{
//            S_START, S_INIT_INNER, S_INIT_OUTER, S_INIT_FEATURES, S_COMPUTE_TARGET,
//            S_COMPUTE_INNER, S_REDUCE_INNER, S_COMPUTE_OUTER, S_REDUCE_OUTER,
//            S_GET_OUTER_RESIDUAL, S_GET_INNER_RESIDUAL,
//            DELTA_F1, DELTA_F2, DELTA_F1_MATRIX, DELTA_F2_MATRIX,
//            UPDATE_OUTER_F1, UPDATE_OUTER_F2, UPDATE_INNER_F1, UPDATE_INNER_F2, S_COMPUTE_ERROR,
//            S_GOTO_NEXT_OR_FINISH,
//            S_TEST,
//            S_DONE
//    } state_t;
//    state_t state;
//    logic signed[31:0] test;
//    int  i, j;
//
//    always_ff @(posedge CLK100MHZ or negedge CPU_RESETN) begin
//        if (!CPU_RESETN) begin
//            seed[0] <= 16'hACE1;
//            seed[1] <= 16'hBEEF;
//            seed[2] <= 16'h1234;
//            seed[3] <= 16'h4321;
//            seed[4] <= 16'hA777;
//            seed[5] <= 16'hAAAA;
//            seed[6] <= 16'h1F2F;
//            seed[7] <= 16'hCAFE;
//            seed[8] <= 16'hBABE;
//            seed16 = 16'h0DA0;
//            state <= S_START;
//    end
//        else begin
//            case(state)
//            S_START: begin
//            cnt <= 0;
//    actualTargetMin <= -1;
//    actualTargetMax <= 1;
//    err_wr_ptr <= 0;
//    i <= 0;
//    j <= 0;
//    state <= S_INIT_INNER;
//    end
//        S_INIT_INNER : begin
//        INNER[i][j] <= rnd_range_fast(seed16, LEFT, RIGHT);
//    if (j == N_INNER_POINTS - 1) begin
//        j <= 0;
//    if (i == N_INNER * N_FEATURES - 1) begin
//        i <= 0;
//    state <= S_INIT_OUTER;
//    end else begin
//        i <= i + 1;
//    end
//        end else begin
//        j <= j + 1;
//    end
//        end
//        S_INIT_OUTER : begin
//        OUTER[i][j] <= rnd_range_fast(seed16, LEFT, RIGHT);
//    if (j == N_OUTER_POINTS - 1) begin
//        j <= 0;
//    if (i == N_INNER - 1) begin
//        state <= S_INIT_FEATURES;
//    end else begin
//        i <= i + 1;
//    end
//        end else begin
//        j <= j + 1;
//    end
//        end
//        S_INIT_FEATURES : begin
//        rnd9(features);
//    state <= S_COMPUTE_TARGET;
//    end
//        S_COMPUTE_TARGET : begin
//        target <= det3x3(features);
//    state <= S_COMPUTE_INNER;
//    end
//        S_COMPUTE_INNER : begin
//        if (target < actualTargetMin) actualTargetMin <= target;
//    if (target > actualTargetMax) actualTargetMax <= target;
//    for (int k = 0; k < N_INNER; ++k) begin
//        for (int j = 0; j < N_FEATURES; ++j) begin
//            intermediate_matrix[k][j] <= ComputeInner(features[j] - N_XMIN, N_INNER_SHIFT,
//                INNER[k * N_FEATURES + j], innerOffset[k * N_FEATURES + j], innerIndex[k * N_FEATURES + j]);
//    end
//        end
//        state <= S_REDUCE_INNER;
//    end
//        S_REDUCE_INNER : begin
//        for (int k = 0; k < N_INNER; ++k) begin
//            intermediate_vector[k] <= ReduceInner(intermediate_matrix[k], MULT_INNER, DENOMINNER, N_TARGETMIN, N_TARGETMAX);
//    end
//        state <= S_COMPUTE_OUTER;
//    end
//        S_COMPUTE_OUTER : begin
//        for (int j = 0; j < N_INNER; ++j) begin
//            outer_vector[j] <= ComputeOuter(intermediate_vector[j] - N_TARGETMIN, N_OUTER_SHIFT, OUTER[j], outerOffset[j], outerIndex[j]);
//    end
//        state <= S_REDUCE_OUTER;
//    end
//        S_REDUCE_OUTER : begin
//        predicted <= ReduceOuter(outer_vector, MULT_OUTER, DENOMOUTER, N_TARGETMIN, N_TARGETMAX);
//    state <= S_GET_OUTER_RESIDUAL;
//    end
//        S_GET_OUTER_RESIDUAL : begin
//        outer_residual <= target - predicted;
//    error_buffer[err_wr_ptr] <= target - predicted;
//    state <= S_GET_INNER_RESIDUAL;
//    end
//        S_GET_INNER_RESIDUAL : begin
//        for (int j = 0; j < N_INNER; ++j) begin
//            inner_residual[j] <= GetInnerResidual(outer_residual, N_OUTER_SHIFT, OUTER[j], outerIndex[j]);
//    end
//        state <= DELTA_F1;
//    end
//        DELTA_F1 : begin
//        for (int j = 0; j < N_INNER; ++j) begin
//            delta_F1[j] <= ((outer_residual >> > N_OUTER_LEARNING_SHIFT) * outerOffset[j]) >> > N_OUTER_SHIFT;
//    end
//        state <= DELTA_F2;
//    end
//        DELTA_F2 : begin
//        for (int j = 0; j < N_INNER; ++j) begin
//            delta_F2[j] <= (outer_residual >> > N_OUTER_LEARNING_SHIFT) - delta_F1[j];
//    end
//        state <= DELTA_F1_MATRIX;
//    end
//        DELTA_F1_MATRIX : begin
//        for (int k = 0; k < N_INNER; ++k) begin
//            for (int j = 0; j < N_FEATURES; ++j) begin
//                delta_F1_matrix[k][j] <= ((inner_residual[k] >> > N_INNER_LEARNING_SHIFT) * innerOffset[k * N_FEATURES + j]) >> > N_INNER_SHIFT;
//    end
//        end
//        state <= DELTA_F2_MATRIX;
//    end
//        DELTA_F2_MATRIX : begin
//        for (int k = 0; k < N_INNER; ++k) begin
//            for (int j = 0; j < N_FEATURES; ++j) begin
//                delta_F2_matrix[k][j] <= (inner_residual[k] >> > N_INNER_LEARNING_SHIFT) - delta_F1_matrix[k][j];
//    end
//        end
//        state <= UPDATE_OUTER_F1;
//    end
//        UPDATE_OUTER_F1 : begin
//        for (int j = 0; j < N_INNER; ++j) begin
//            OUTER[j][outerIndex[j] + 1] <= OUTER[j][outerIndex[j] + 1] + delta_F1[j];
//    end
//        state <= UPDATE_OUTER_F2;
//    end
//        UPDATE_OUTER_F2 : begin
//        for (int j = 0; j < N_INNER; ++j) begin
//            OUTER[j][outerIndex[j]] <= OUTER[j][outerIndex[j]] + delta_F2[j];
//    end
//        state <= UPDATE_INNER_F1;
//    end
//        UPDATE_INNER_F1 : begin
//        for (int k = 0; k < N_INNER; ++k) begin
//            for (int j = 0; j < N_FEATURES; ++j) begin
//                INNER[k * N_FEATURES + j][innerIndex[k * N_FEATURES + j] + 1] <=
//                INNER[k * N_FEATURES + j][innerIndex[k * N_FEATURES + j] + 1] + delta_F1_matrix[k][j];
//    end
//        end
//        state <= UPDATE_INNER_F2;
//    end
//        UPDATE_INNER_F2 : begin
//        for (int k = 0; k < N_INNER; ++k) begin
//            for (int j = 0; j < N_FEATURES; ++j) begin
//                INNER[k * N_FEATURES + j][innerIndex[k * N_FEATURES + j]] <=
//                INNER[k * N_FEATURES + j][innerIndex[k * N_FEATURES + j]] + delta_F2_matrix[k][j];
//    end
//        end
//        state <= S_GOTO_NEXT_OR_FINISH;
//    end
//        S_GOTO_NEXT_OR_FINISH : begin
//        if (cnt >= N_PROCESSED_RECORDS - 1)
//            state <= S_COMPUTE_ERROR;
//        else begin
//            cnt <= cnt + 1;
//    err_wr_ptr <= err_wr_ptr + 1;
//    state <= S_INIT_FEATURES;
//    end
//        end
//        S_COMPUTE_ERROR : begin
//        test <= AvgAbsError(error_buffer);
//    state <= S_DONE;
//    end
//        S_DONE : begin
//        state <= S_DONE;
//    end
//        endcase
//        end //else
//        end //always
//
//        assign LED = test;
//
//    endmodule

