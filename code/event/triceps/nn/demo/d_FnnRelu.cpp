//
// (C) Copyright 2022 Sergey A. Babkin.
// This file is a part of Triceps.
// See the file COPYRIGHT for the copyright notice and license information
//
//
// Demo of the floating-point simple NN with ReLU function.

#include <utest/Utest.h>
#include <nn/test/TestFloatNn.h>

static long random_seed = time(NULL);

UTESTCASE train_relu_from_random(Utest *utest)
{
	Erref err;
	bool debug = true;  // enable extra printouts

	// Train from random initial values.

	// Do just the positive side, one result value.
	FloatNeuralNet::LevelSizeVector levels = {1, 3, 1};
	FloatNeuralNet::Options options;
	// options.autoRate_ = true;
	// options.weightSaturation_ = 100.;
	TestFloatNn nn(levels, FloatNeuralNet::RELU, &options);
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

	// Examples of bad initial weights:
	//
	// --- L1
	// L1 0 [used 0]: ( + 0:0.129584 + B:0.815727)
	// L1 1 [used 0]: ( + 0:0.715262 + B:0.307513)
	// L1 2 [used 0]: ( + 0:-0.559153 + B:-0.110741)
	// --- L2
	// L2 0 [used 0]: ( + 0:-0.518741 + 1:-0.415665 + 2:0.312564 + B:-0.574592)
	//   With applyGradient() it converges real badly and keep reclaiming a neuron on every pass:
	//   Mean error: 0.393264, mean squared error: 0.512311, gradient: last 8.6531 all 8.65311
	//   --- L1
	//   L1 0 [used 2]: ( + 0:0.246844 + B:-0.201367)
	//   L1 1 [used 3]: ( + 0:0.067972 + B:-0.049803)
	//   L1 2 [used 3]: ( + 0:0.054557 + B:-0.043072)
	//   --- L2
	//   L2 0 [used 0]: ( + 0:-0.001343 + 1:0.000163 + 2:0.001570 + B:-0.043259)
	//   But without applyGradient(), not so bad, reclaims 2 naurons on pass 0 and then works:
	//   Mean error: 0.0216786, mean squared error: 0.0268784, gradient: last 0.0507443 all 0.0894848
	//
	// --- L1
	// L1 0 [used 0]: ( + 0:0.143875 + B:-0.407469)
	// L1 1 [used 0]: ( + 0:-0.250101 + B:0.781502)
	// L1 2 [used 0]: ( + 0:0.633543 + B:-0.565700)
	// --- L2
	// L2 0 [used 0]: ( + 0:0.452139 + 1:0.100083 + 2:0.166219 + B:-0.249058)
	//
	// no reclaims!
	// --- L1
	// L1 0 [used 0]: ( + 0:-0.841933 + B:0.538026)
	// L1 1 [used 0]: ( + 0:0.449234 + B:0.369901)
	// L1 2 [used 0]: ( + 0:0.270552 + B:0.463445)
	// --- L2
	// L2 0 [used 0]: ( + 0:-0.161000 + 1:-0.375530 + 2:0.338396 + B:0.492619)
	//
	// --- L1
	// L1 0 [used 0]: ( + 0:-0.111512 + B:0.848896)
	// L1 1 [used 0]: ( + 0:-0.153214 + B:0.426140)
	// L1 2 [used 0]: ( + 0:-0.734883 + B:-0.259729)
	// --- L2
	// L2 0 [used 0]: ( + 0:-0.553393 + 1:-0.399681 + 2:0.214344 + B:0.426774)
	//
	// Gradient goes to almost 0 but the error doesn't:
	//
	// Mean error: 0.0763637, mean squared error: 0.0883176, gradient: last 3.29123e-06 all 4.76226e-06
	// --- L1
	// L1 0 [used 11]: ( + 0:0.619363 + B:0.478925)
	// L1 1 [used 11]: ( + 0:0.974418 + B:0.548424)
	// L1 2 [used 11]: ( + 0:0.704373 + B:0.717618)
	// --- L2
	// L2 0 [used 9]: ( + 0:0.229605 + 1:0.741949 + 2:0.191405 + B:-0.804220)
	//
	// Joined gradient
	// Before:
	// --- L1
	// L1 0 [used 0]: ( + 0:0.727602 + B:-0.311867)
	// L1 1 [used 0]: ( + 0:0.756054 + B:-0.991187)
	// L1 2 [used 0]: ( + 0:-0.744358 + B:0.774400)
	// --- L2
	// L2 0 [used 0]: ( + 0:-0.349429 + 1:0.531338 + 2:-0.119818 + B:0.320743)
	// After:
	// Mean error: 0.0763636, mean squared error: 0.0883176, gradient: last 3.0747e-06 all 4.65981e-06
	// --- L1
	// L1 0 [used 11]: ( + 0:-0.532397 + B:0.532601)
	// L1 1 [used 11]: ( + 0:-0.933158 + B:0.933956)
	// L1 2 [used 11]: ( + 0:-0.871085 + B:0.872817)
	// --- L2
	// L2 0 [used 9]: ( + 0:-0.098874 + 1:-0.436746 + 2:-0.619694 + B:0.851442)
	// 
	// Seed: 1665622913
	// Before:
	// --- L1
	// L1 0 [used 0]: ( + 0:-0.680445 + B:-0.960084)
	// L1 1 [used 0]: ( + 0:-0.504175 + B:0.176647)
	// L1 2 [used 0]: ( + 0:-0.870858 + B:-0.983848)
	// --- L2
	// L2 0 [used 0]: ( + 0:-0.368900 + 1:-0.569019 + 2:0.511528 + B:0.096956)
	// After:
	// Mean error: 0.0763636, mean squared error: 0.0883176, gradient: last 1.74948e-06 all 2.24888e-06
	// --- L1
	// L1 0 [used 11]: ( + 0:0.749692 + B:0.924955)
	// L1 1 [used 11]: ( + 0:-0.619237 + B:0.620064)
	// L1 2 [used 11]: ( + 0:0.970499 + B:0.933295)
	// --- L2
	// L2 0 [used 9]: ( + 0:0.196443 + 1:-0.895972 + 2:0.306965 + B:-0.062630)
	//
	// Seed: 1666464657
	// Basic RELU reclaims successfully, LEAKY_RELU gets stuck.
	// --- L1
	// L1 0 : ( + 0:0.461910 + B:-0.493564)
	// L1 1 : ( + 0:-0.956627 + B:-0.834016)
	// L1 2 : ( + 0:-0.582727 + B:-0.126361)
	// --- L2
	// L2 0 : ( + 0:-0.153497 + 1:-0.570985 + 2:0.348785 + B:0.345654)

}

