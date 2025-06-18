//
// (C) Copyright 2022 Sergey A. Babkin.
// This file is a part of Triceps.
// See the file COPYRIGHT for the copyright notice and license information
//
//
// Test of the floating-point simple NN implementation.

#include <utest/Utest.h>
#include <nn/test/TestFloatNn.h>

// Weights that define a parabolic function from one argument to one result.
// The parabola in the range [-1, 1] is defined as 2 straight line segments in
// the positive half, and the same in the negative half. The results for
// negative and positive inputs are defined as 2 seratae variables. It would be
// possible to just add them together, since a parabola produces positive
// values only, but it's more interesting for testing in tyhe separated way.
//
// The level sizes are:
//   input [0] : 1
//         [1] : 6 (3 steps for positive and negative each)
//  output [2] : 2  (split result).
//
// The positive side is formed as:
//   y0 = relu(relu(x*1/3) + relu(x*2/3 - 2/9) + relu(x*2/3 - 4/9))
// The negative side:
//   y1 = relu(relu(x*-1/3) + relu(x*-2/3 - 2/9) + relu(x*-2/3 - 4/9))

constexpr size_t PARA_LEVEL_1_SZ = 6 * 2;
float paraLevel1[PARA_LEVEL_1_SZ] = {
	// Input to neurons responsible for the positive side.
	1./3.,  // 0 -> 0
	2./3.,  // 0 -> 1
	2./3.,  // 0 -> 2
	// Input to neurons responsible for the negative side.
	-1./3.,  // 0 -> 3
	-2./3.,  // 0 -> 4
	-2./3.,  // 0 -> 5
	// Bias to neurons responsible for the positive side.
	0.,  // bias -> 0
	-2./9.,  // bias -> 1
	-4./9.,  // bias -> 2
	// Bias to neurons responsible for the negative side.
	0.,  // bias -> 3
	-2./9.,  // bias -> 4
	-4./9.,  // bias -> 5
};

constexpr size_t PARA_LEVEL_2_SZ = 2 * 7;
float paraLevel2[PARA_LEVEL_2_SZ] = {
	1.,  // 0 -> 0
	0.,  // 0 -> 1
	1.,  // 1 -> 0
	0.,  // 1 -> 1
	1.,  // 2 -> 0
	0.,  // 2 -> 1
	0.,  // 3 -> 0
	1.,  // 3 -> 1
	0.,  // 4 -> 0
	1.,  // 4 -> 1
	0.,  // 5 -> 0
	1.,  // 5 -> 1
	0.,  // bias -> 0
	0.,  // bias -> 1
};

UTESTCASE construct(Utest *utest)
{
	FloatNeuralNet::LevelSizeVector levels = {1, 6, 2};
	TestFloatNn nn(levels, FloatNeuralNet::RELU);
	UT_NOERROR(nn.getErrors());

	std::vector<size_t> &train = nn.levelTrainIdx();
	std::vector<size_t> &weights = nn.levelWeightsIdx();
	std::vector<size_t> &usage = nn.levelUsageIdx();

	if (UT_IS(train.size(), 4)
	|| UT_IS(weights.size(), 4)
	|| UT_IS(usage.size(), 4))
		return;

	UT_IS(train[0], 0);
	UT_IS(train[1], 1 * 2);
	UT_IS(train[2], (1 + 6) * 2);
	UT_IS(train[3], (1 + 6 + 2) * 2);

	UT_IS(weights[0], 0); 
	UT_IS(weights[1], 0); // level 0 doesn't have any weights
	UT_IS(weights[2], (1 + 1) * 6); 
	UT_IS(weights[3], (1 + 1) * 6 +  (6 + 1) * 2); 

	UT_IS(usage[0], 0);
	UT_IS(usage[1], 1);
	UT_IS(usage[2], 1 + 6);
	UT_IS(usage[3], 1 + 6 + 2);

	// SBXXX test failures
}

UTESTCASE construct_options(Utest *utest)
{
	FloatNeuralNet::LevelSizeVector levels = {1, 6, 2};
	FloatNeuralNet::Options options;
	options.reluLeakiness_ *= 10;
	TestFloatNn nn(levels, FloatNeuralNet::LEAKY_RELU, &options);

	FloatNeuralNet::Options opt2 = nn.getOptions();
	UT_IS(opt2.reluLeakiness_, options.reluLeakiness_);
}

UTESTCASE setGetWeights(Utest *utest)
{
	FloatNeuralNet::LevelSizeVector levels = {1, 6, 2};
	TestFloatNn nn(levels, FloatNeuralNet::RELU);
	UT_NOERROR(nn.getErrors());

	// Set the levels of weights.

	FloatNeuralNet::ValueVector level1(paraLevel1, paraLevel1 + PARA_LEVEL_1_SZ);
	UT_NOERROR(nn.setLevel(1, level1));

	FloatNeuralNet::ValueVector level2(paraLevel2,  paraLevel2 + PARA_LEVEL_2_SZ);
	UT_NOERROR(nn.setLevel(2, level2));

	// Read the weights back.

	FloatNeuralNet::ValueVector readvals;

	UT_NOERROR(nn.getLevel(1, readvals));
	UT_IS(readvals.size(), level1.size());
	UT_ASSERT(!memcmp(readvals.data(), paraLevel1, sizeof(paraLevel1)));

	// SBXXX test failures
}

UTESTCASE computeForTraining(Utest *utest)
{
	Erref err;

	// Use the "intelligently designed" model.

	FloatNeuralNet::LevelSizeVector levels = {1, 6, 2};
	TestFloatNn nn(levels, FloatNeuralNet::RELU);
	UT_NOERROR(nn.getErrors());

	// Set the levels of weights.

	FloatNeuralNet::ValueVector level1(paraLevel1, paraLevel1 + PARA_LEVEL_1_SZ);
	UT_NOERROR(nn.setLevel(1, level1));

	FloatNeuralNet::ValueVector level2(paraLevel2,  paraLevel2 + PARA_LEVEL_2_SZ);
	UT_NOERROR(nn.setLevel(2, level2));

	// Run the computation.
	FloatNeuralNet::ValueVector input(1);
	FloatNeuralNet::ValueVector output;

	// Indexes of the activated results in the output vector.
	size_t r0_idx = 1 * 2 + 6 * 2 + 2;
	size_t r1_idx = 1 * 2 + 6 * 2 + 3;

	// The center point, 0.

	input[0] = 0.;
	UT_NOERROR(nn.computeX(input, output));
	if (UT_IS(output.size(), 1 * 2 + 6 * 2 + 2 * 2))
		return;  // don't read the values

	UT_NEAR(output[r0_idx], 0.,  eps);
	UT_NEAR(output[r1_idx], 0.,  eps);

	// Upper boundary.

	input[0] = 1.;
	UT_NOERROR(nn.computeX(input, output));
	if (UT_IS(output.size(), 1 * 2 + 6 * 2 + 2 * 2))
		return;  // don't read the values

	UT_NEAR(output[r0_idx], 1.,  eps);
	UT_NEAR(output[r1_idx], 0.,  eps);

	// Lower boundary.

	input[0] = -1.;
	UT_NOERROR(nn.computeX(input, output));
	if (UT_IS(output.size(), 1 * 2 + 6 * 2 + 2 * 2))
		return;  // don't read the values

	UT_NEAR(output[r0_idx], 0.,  eps);
	UT_NEAR(output[r1_idx], 1.,  eps);

	// At 0.5.

	input[0] = 0.5;
	UT_NOERROR(nn.computeX(input, output));
	if (UT_IS(output.size(), 1 * 2 + 6 * 2 + 2 * 2))
		return;  // don't read the values

	UT_NEAR(output[r0_idx], 5./18.,  eps);
	UT_NEAR(output[r1_idx], 0.,  eps);

	// At -0.5.

	input[0] = -0.5;
	UT_NOERROR(nn.computeX(input, output));
	if (UT_IS(output.size(), 1 * 2 + 6 * 2 + 2 * 2))
		return;  // don't read the values

	UT_NEAR(output[r0_idx], 0.,  eps);
	UT_NEAR(output[r1_idx], 5./18.,  eps);

	// SBXXX test failures
	// SBXXX test compute() and computation of highest
}

UTESTCASE reclaim(Utest *utest)
{
	Erref err;

	FloatNeuralNet::LevelSizeVector levels = {1, 2, 2};
	TestFloatNn nn(levels, FloatNeuralNet::RELU);
	UT_NOERROR(nn.getErrors());

	// Set the weights to values that can't be created during randomization.

	constexpr size_t level1_sz = (1 + 1) * 2;
	FloatNeuralNet::ValueVector level1(level1_sz);
	for (size_t i = 0; i < level1_sz; i++)
		level1[i] = 100.;
	UT_NOERROR(nn.setLevel(1, level1));

	constexpr size_t level2_sz = (2 + 1) * 2;
	FloatNeuralNet::ValueVector level2(level2_sz);
	for (size_t i = 0; i < level2_sz; i++)
		level2[i] = 100.;
	UT_NOERROR(nn.setLevel(2, level2));

	// Do a computation to see that the produced value will be as expected.
	FloatNeuralNet::ValueVector ins(1);
	FloatNeuralNet::ValueVector outs(2);
	size_t highest;

	ins[0] = 1.;
	UT_NOERROR(nn.compute(ins, outs, highest));
	UT_NEAR(outs[0], 100. + 2. * 100. * 200.,  eps);
	UT_NEAR(outs[1], 100. + 2. * 100. * 200.,  eps);

	// Set the usages such that the first neuron in each level will get reclaimed
	// (except for the last level that never gets reclaimed).
	nn.usageAt(1, 1) = 2;
	nn.usageAt(2, 0) = 1;

	size_t reclaim_count = 0;
	UT_NOERROR(nn.reclaim(1, reclaim_count));

	UT_IS(reclaim_count, 1);

	// Weights flipped on reclaim.
	UT_NEAR(nn.weightAt(1, 0, 0), -100., eps);
	UT_NEAR(nn.weightAt(1, 0, 1), -100., eps);
	UT_NEAR(nn.weightAt(1, 1, 0), 100., eps);
	UT_NEAR(nn.weightAt(1, 1, 1), 100., eps);

	// Gets reset on reclaiming of the input neuron.
	UT_NEAR(nn.weightAt(2, 0, 0), 0., eps);
	UT_NEAR(nn.weightAt(2, 0, 1), 100., eps);
	UT_NEAR(nn.weightAt(2, 0, 2), 100., eps);
	// Gets reset on reclaiming of the input neuron.
	UT_NEAR(nn.weightAt(2, 1, 0), 0., eps);
	UT_NEAR(nn.weightAt(2, 1, 1), 100., eps);
	UT_NEAR(nn.weightAt(2, 1, 2), 100., eps);

	// SBXXX test failures
}

