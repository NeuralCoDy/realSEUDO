//
// (C) Copyright 2022 Sergey A. Babkin.
// This file is a part of Triceps.
// See the file COPYRIGHT for the copyright notice and license information
//
//
// Demo of the floating-point simple NN with Leaky ReLU function.

#include <utest/Utest.h>
#include <nn/test/TestFloatNn.h>

static long random_seed = time(NULL);

UTESTCASE train_leaky_relu_from_random(Utest *utest)
{
	Erref err;
	bool debug = true;  // enable extra printouts

	// Train from random initial values.

	// Do just the positive side, one result value.
	FloatNeuralNet::LevelSizeVector levels = {1, 3, 1};
	FloatNeuralNet::Options options;
	// options.autoRate_ = true;
	TestFloatNn nn(levels, FloatNeuralNet::LEAKY_RELU, &options);
	UT_NOERROR(nn.getErrors());

	// Do the training.
	if (true) {
		long seed = random_seed;
		// seed = 1665622913; // a bad one
		seed = 1665633373; // a good one
		printf("Seed: %ld\n", seed);
		srand48(seed);
		nn.randomize();
	} else {
		// Try some previously known example.
		// --- L1
		// L1 0 [used 0]: ( + 0:0.129584 + B:0.815727)
		// L1 1 [used 0]: ( + 0:0.715262 + B:0.307513)
		// L1 2 [used 0]: ( + 0:-0.559153 + B:-0.110741)
		// --- L2
		// L2 0 [used 0]: ( + 0:-0.518741 + 1:-0.415665 + 2:0.312564 + B:-0.574592)
		FloatNeuralNet::ValueVector level1({
			// Input to neurons responsible for the positive side.
			0.129584,  // 0 -> 0
			0.715262,  // 0 -> 1
			-0.559153,  // 0 -> 2
			// Bias to neurons responsible for the positive side.
			0.815727,  // bias -> 0
			0.307513,  // bias -> 1
			-0.110741,  // bias -> 2
		});
		UT_NOERROR(nn.setLevel(1, level1));

		FloatNeuralNet::ValueVector level2({-0.518741, -0.415665, 0.312564, -0.574592});
		UT_NOERROR(nn.setLevel(2, level2));
	}

	if (debug) {
		nn.printSimpleDump(utest);
	}

	int nSteps = 100;
	double rate = 0.1 / nSteps;

	// nn.trainSquareFunction(utest, /*nPasses*/ 1000, /*printEvery*/ 100, nSteps, /*trainRate*/ rate, /*applyRate*/ 0., /*reclaim*/ true);
	nn.trainSquareFunction(utest, /*nPasses*/ 1000, /*printEvery*/ 100, nSteps, /*trainRate*/ 0., /*applyRate*/ rate, /*reclaim*/ true);

	// Print the new values and compute their stats.
	printf("--- After training:\n");
	nn.printSquareFunction(utest, nullptr, nullptr);
	nn.printSquareStats(utest, nSteps);

	nn.printSimpleDump(utest);
}


