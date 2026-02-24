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
//Relative error for unseen data is betwee 1% and 1.5%. 
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
	const int nInner = 16;
	const int nInnerPoints = 3;
	const int nInnerShift = 7;      //interval [0,255], 3 points, 2 linear segments, 1 << 7 = 128 lenght of the segment
	const int nInnerLearningShift = 4;
	//this is division by number of features = 9, multiplication by 113 and division by 1 << 10
	const short MultInner = 113;
	const short DenomInner = 10;

	//network outer
	const int nOuter = 1;
	const int nOuterPoints = 21;
	const int nOuterShift = 11;  //interval [-20'480, 20480], 21 points, 20 linear segments, 1 << 11 = 2028 length of the segment
	const int nOuterLearningShift = 2;
	//this is division by number of inner blocks, multiplication by 1 and division by 1 << 4
	const short MultOuter = 1;
	const short DenomOuter = 4;

	//data size
	const int nProcessedRecords = 50'000;

	clock_t start_application = clock();
	clock_t current_time = clock();

	//initialize models
	int range = (TargetMax - TargetMin) >> 2;
	int mean = (TargetMax + TargetMin) >> 1;
	int left = mean - range;
	int right = mean + range;

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

		//1 FPGA cycle
		outer_residual = target - predicted;
		error_buffer[cnt & 0xff] = target - predicted;

		//here forward pass is completed and backprop started

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

