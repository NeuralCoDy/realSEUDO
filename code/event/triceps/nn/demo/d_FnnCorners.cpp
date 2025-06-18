//
// (C) Copyright 2022 Sergey A. Babkin.
// This file is a part of Triceps.
// See the file COPYRIGHT for the copyright notice and license information
//
//
// Demo of the floating-point simple NN with CORNERS (V-shaped) function.

#include <utest/Utest.h>
#include <nn/test/TestFloatNn.h>

static long random_seed = time(NULL);

// The usual square function, but a single neuron with CORNER activation.
UTESTCASE train_corners1_from_random(Utest *utest)
{
	Erref err;
	bool debug = true;  // enable extra printouts

	// Train from random initial values.

	// Do just the positive side, one result value.
	FloatNeuralNet::LevelSizeVector levels = {1, 1};

	FloatNeuralNet::Options options;
	// A higher saturation allows a greater flexibility in a single neuron
	// (for example, saturation at 1 doesn't allow a single neuron to
	// simulate the square function any better than a plain ReLU)
	// but increases the risk of overshooting on gradient steps and divergence.
	options.weightSaturation_ = 1e10;
	options.autoRate_ = true;
	// options.scaleRatePerLayer_ = true;
	options.enableWeightFloor_ = true;
	// Also see the adjustment in trainSquareFunction()
	options.trainingRateScale_ = 1;
	// options.momentum_ = true;

	TestFloatNn nn(levels, FloatNeuralNet::CORNER, &options);
	UT_NOERROR(nn.getErrors());

	// Do the training.

	long seed = random_seed;
	// seed = 1665633373;
	// seed = 1666547471;
	printf("Seed: %ld\n", seed);
	srand48(seed);
	nn.randomize();

	if (debug) {
		nn.printSimpleDump(utest);
	}

	int nSteps = 10;
	double rate = 0.1 / nSteps;
	int nPasses = 10000;
	int printEvery = nPasses / 10;

	nn.trainSquareFunction(utest, nPasses, printEvery, nSteps, /*trainRate*/ 0., /*applyRate*/ rate, /*reclaim*/ true);
	// nn.trainSquareFunction(utest, nPasses, printEvery, nSteps, /*trainRate*/ rate, /*applyRate*/ 0., /*reclaim*/ true);
	// nn.trainSquareFunction(utest, nPasses, printEvery, nSteps, /*trainRate*/ rate/2., /*applyRate*/ rate/2., /*reclaim*/ true);

	// Print the new values and compute their stats.
	printf("--- After training:\n");
	nn.printSquareFunction(utest, nullptr, nullptr);
	nn.printSquareStats(utest, nSteps);

	nn.printSimpleDump(utest);

	// These examples include multi-neuron geometries.
	//
	// bad seed 1666479695
	// L1 0 : ( + 0:0.858077 + B:0.464669 [C 0.010000:+ 0.000000:1.000000])
	//
	// Bad Seed: 1666493206
	// (has a hard time digging out of the right side)
	// L1 0 : ( + 0:-0.403937 + B:-0.236159 [C 0.010000:+ 0.000000:1.000000])
	// -> L1 0 : ( + 0:-1.477942 + B:0.586464 [C -1.072212:+ 0.029667:0.091867])
	// --- After training:
	//  0.0 ->   0.083544 (true   0.000000) d=0.083544
	//  0.1 ->   0.069966 (true   0.010000) d=-0.013577
	//  0.2 ->   0.056389 (true   0.040000) d=-0.013577
	//  0.3 ->   0.042811 (true   0.090000) d=-0.013577
	//  0.4 ->   0.034719 (true   0.160000) d=-0.008092
	//  0.5 ->   0.193186 (true   0.250000) d=0.158467
	//  0.6 ->   0.351653 (true   0.360000) d=0.158467
	//  0.7 ->   0.510119 (true   0.490000) d=0.158467
	//  0.8 ->   0.668586 (true   0.640000) d=0.158467
	//  0.9 ->   0.827053 (true   0.810000) d=0.158467
	//  1.0 ->   0.985519 (true   1.000000) d=0.158467
	//    Mean error: 0.0434334, mean squared error: 0.0554207, gradient: last 0.216917 all 0.216917
	// 
	// Bad Seed: 1666494916
	// (two neurons fighting each other? but goes OK with a higher rate)
	// --- L1
	// L1 0 : ( + 0:-0.274431 + B:0.671716 [C 0.010000:+ 0.000000:1.000000])
	// L1 1 : ( + 0:-0.506710 + B:0.206302 [C 0.010000:+ 0.000000:1.000000])
	// --- L2
	// L2 0 : ( + 0:0.209602 + 1:-0.133366 + B:-0.142273 [C 0.010000:+ 0.000000:1.000000])
	// --- After training:
	//   0.0 ->   0.349995 (true   0.000000) d=0.349995
	//   0.1 ->   0.349554 (true   0.010000) d=-0.000441
	//   0.2 ->   0.349113 (true   0.040000) d=-0.000441
	//   0.3 ->   0.349122 (true   0.090000) d=0.000009
	//   0.4 ->   0.349121 (true   0.160000) d=-0.000001
	//   0.5 ->   0.349517 (true   0.250000) d=0.000396
	//   0.6 ->   0.349912 (true   0.360000) d=0.000396
	//   0.7 ->   0.350308 (true   0.490000) d=0.000396
	//   0.8 ->   0.350704 (true   0.640000) d=0.000396
	//   0.9 ->   0.351099 (true   0.810000) d=0.000396
	//   1.0 ->   0.351495 (true   1.000000) d=0.000396
	// Mean error: 0.281173, mean squared error: 0.327611, gradient: last 0.0940402 all 0.164202
	// --- L1
	// L1 0 : ( + 0:-0.347261 + B:0.507169 [C 0.010000:+ -0.076627:0.978157])
	// L1 1 : ( + 0:-0.497355 + B:0.146000 [C -0.005975:+ 0.067455:0.995958])
	// --- L2
	// L2 0 : ( + 0:0.081586 + 1:-0.119136 + B:-0.017523 [C -0.140963:+ 0.348775:0.999761])
	//
	// Good Seed: 1666545938 (10K steps)
	// --- L1
	// L1 0 : ( + 0:-0.138724 + B:0.616802 [C 0.010000:+ 0.000000:1.000000])
	// L1 1 : ( + 0:0.799690 + B:-0.349348 [C 0.010000:+ 0.000000:1.000000])
	// --- L2
	// L2 0 : ( + 0:0.490667 + 1:0.137396 + B:0.078366 [C 0.010000:+ 0.000000:1.000000])
	// --- After training:
	//   0.0 ->  -0.003561 (true   0.000000) d=-0.003561
	//   0.1 ->   0.017130 (true   0.010000) d=0.020691
	//   0.2 ->   0.037822 (true   0.040000) d=0.020692
	//   0.3 ->   0.083755 (true   0.090000) d=0.045934
	//   0.4 ->   0.167521 (true   0.160000) d=0.083766
	//   0.5 ->   0.251286 (true   0.250000) d=0.083765
	//   0.6 ->   0.350626 (true   0.360000) d=0.099340
	//   0.7 ->   0.494989 (true   0.490000) d=0.144363
	//   0.8 ->   0.639353 (true   0.640000) d=0.144364
	//   0.9 ->   0.813323 (true   0.810000) d=0.173970
	//   1.0 ->   0.997725 (true   1.000000) d=0.184401
	// Mean error: 0.00441186, mean squared error: 0.00517764, gradient: last 0.0224426 all 0.0232175
	// --- L1
	// L1 0 : ( + 0:1.144403 + B:-0.657229 [C 0.372638:+ 2.015183:0.667570])
	// L1 1 : ( + 0:0.407222 + B:-0.336388 [C 0.140642:+ 1.418304:0.922680])
	// --- L2
	// L2 0 : ( + 0:1.971001 + 1:1.380176 + B:-5.620495 [C 0.225012:+ 0.050232:0.910911])
	// [above is with applyGradient, without it:
	//   Mean error: 0.00582674, mean squared error: 0.00714143, gradient: last 0.0370658 all 0.0421325
	// mixed:
	//   Mean error: 0.00435438, mean squared error: 0.00513681, gradient: last 0.0226888 all 0.0247726
	// ]
	//
	// Bad Seed: 1666547164 (gets stuck around 0.02 no matter what)
	// --- L1
	// L1 0 : ( + 0:-0.852189 + B:-0.781530 [C 0.010000:+ 0.000000:1.000000])
	// L1 1 : ( + 0:0.489927 + B:-0.838039 [C 0.010000:+ 0.000000:1.000000])
	// --- L2
	// L2 0 : ( + 0:-0.266901 + 1:-0.288694 + B:0.529845 [C 0.010000:+ 0.000000:1.000000])
	// --- After training:
	//   0.0 ->  -0.034955 (true   0.000000) d=-0.034955
	//   0.1 ->   0.015714 (true   0.010000) d=0.050669
	//   0.2 ->   0.066383 (true   0.040000) d=0.050669
	//   0.3 ->   0.117052 (true   0.090000) d=0.050669
	//   0.4 ->   0.167722 (true   0.160000) d=0.050669
	//   0.5 ->   0.218391 (true   0.250000) d=0.050669
	//   0.6 ->   0.335606 (true   0.360000) d=0.117215
	//   0.7 ->   0.497530 (true   0.490000) d=0.161924
	//   0.8 ->   0.659229 (true   0.640000) d=0.161698
	//   0.9 ->   0.820927 (true   0.810000) d=0.161699
	//   1.0 ->   0.982626 (true   1.000000) d=0.161698
	// Mean error: 0.0193537, mean squared error: 0.0217087, gradient: last 0.0226936 all 0.0389386
	// --- L1
	// L1 0 : ( + 0:-1.683254 + B:1.010224 [C 0.551624:+ -1.487493:1.016556])
	// L1 1 : ( + 0:0.536227 + B:-0.547802 [C 0.106298:+ -0.626385:1.000000])
	// --- L2
	// L2 0 : ( + 0:-1.616243 + 1:-0.695802 + B:-2.783998 [C 0.185878:+ 0.255627:1.106721])
	//
	// Bad Seed: 1666547471 (dissolves into NaN around pass 9000)
	// --- L1
	// L1 0 : ( + 0:-0.246833 + B:0.683488 [C 0.010000:+ 0.000000:1.000000])
	// L1 1 : ( + 0:-0.953383 + B:-0.991244 [C 0.010000:+ 0.000000:1.000000])
	// --- L2
	// L2 0 : ( + 0:0.349361 + 1:0.237200 + B:0.639784 [C 0.010000:+ 0.000000:1.000000])
	// Works OK with a smaller rate but doesn't converge well; rate 0.02 and 90K steps:
	// --- After training:
	//   0.0 ->   0.222871 (true   0.000000) d=0.222871
	//   0.1 ->   0.211408 (true   0.010000) d=-0.011463
	//   0.2 ->   0.199945 (true   0.040000) d=-0.011464
	//   0.3 ->   0.188482 (true   0.090000) d=-0.011463
	//   0.4 ->   0.177019 (true   0.160000) d=-0.011463
	//   0.5 ->   0.166671 (true   0.250000) d=-0.010347
	//   0.6 ->   0.321623 (true   0.360000) d=0.154952
	//   0.7 ->   0.468576 (true   0.490000) d=0.146953
	//   0.8 ->   0.615534 (true   0.640000) d=0.146958
	//   0.9 ->   0.294555 (true   0.810000) d=-0.320980
	//   1.0 ->   1.187220 (true   1.000000) d=0.892665
	//     Mean error: 0.142726, mean squared error: 0.199093, gradient: last 7.15263 all 16.5377
	// Finally starts converging with step 0.01. In gradient-only mode it actually does converge
	// OK to 0.0035 in the first 3K steps before it starts diverging.
	//
	// Bad Seed: 1666548800 (can't pull out of 0 when it tries to flip direction)
	// --- L1
	// L1 0 : ( + 0:0.486449 + B:-0.232074 [C 0.010000:+ 0.000000:1.000000])
	// --- L2
	// L2 0 : ( + 0:-0.607038 + B:0.427622 [C 0.010000:+ 0.000000:1.000000])
	// --- After training:
	//   0.0 ->   0.355051 (true   0.000000) d=0.355051
	//   0.1 ->   0.355053 (true   0.010000) d=0.000002
	//   0.2 ->   0.355052 (true   0.040000) d=-0.000001
	//   0.3 ->   0.355051 (true   0.090000) d=-0.000001
	//   0.4 ->   0.355050 (true   0.160000) d=-0.000001
	//   0.5 ->   0.355048 (true   0.250000) d=-0.000001
	//   0.6 ->   0.355047 (true   0.360000) d=-0.000001
	//   0.7 ->   0.355046 (true   0.490000) d=-0.000001
	//   0.8 ->   0.355045 (true   0.640000) d=-0.000001
	//   0.9 ->   0.355044 (true   0.810000) d=-0.000001
	//   1.0 ->   0.355042 (true   1.000000) d=-0.000001
	// Mean error: 0.28228, mean squared error: 0.328371, gradient: last 0.111097 all 0.142883
	// --- L1
	// L1 0 : ( + 0:-0.000294 + B:0.000029 [C -0.624450:+ -0.340627:0.970878])
	// --- L2
	// L2 0 : ( + 0:-0.887208 + B:-0.302327 [C 0.075367:+ 0.355062:1.189192])
	//
	// Bad Seed: 1669514145 (doesn't converge well with autoRate auto-reduction, does OK without auto-reduction)
	// --- L1
	// L1 0 : ( + 0:0.976031 + B:-0.865050 [C 0.010000:+ 0.000000:1.000000])
	// L1 1 : ( + 0:-0.573728 + B:-0.157718 [C 0.010000:+ 0.000000:1.000000])
	// --- L2
	// L2 0 : ( + 0:-0.546677 + 1:-0.384530 + B:-0.081090 [C 0.010000:+ 0.000000:1.000000])
	// Mean error: 0.254679, mean squared error: 0.293236, gradient: last 1.0726 all 1.28142
	//
	// Good Seed: 1669514145 (converges decently with auto-rate, with or without reduction
	// but better with reduction).
	// --- L1
	// L1 0 : ( + 0:-0.472334 + B:0.372453 [C 0.010000:+ 0.000000:1.000000])
	// L1 1 : ( + 0:0.662083 + B:-0.892575 [C 0.010000:+ 0.000000:1.000000])
	// --- L2
	// L2 0 : ( + 0:-0.300421 + 1:0.453664 + B:0.325481 [C 0.010000:+ 0.000000:1.000000])

}

UTESTCASE train_corners1_xor(Utest *utest)
{
	Erref err;
	bool debug = true;  // enable extra printouts

	// Train from random initial values.

	// Do just the positive side, one result value.
	FloatNeuralNet::LevelSizeVector levels = {2, 1};

	FloatNeuralNet::Options options;
	// A higher saturation allows a greater flexibility in a single neuron
	// but increases the risk of overshooting on gradient steps and divergence.
	// Interestingly, the XOR function can be simulated with weights not
	// exceeding 1, and generally ends up with weights very close to 1, but
	// limiting the values to 1 produces a very poor result, and even limiting
	// to 1.1 doesn't allow the values to fully converge to the right result,
	// so I suppose being able to have higher values as intermediate results
	// is important.
	options.weightSaturation_ = 2.;

	TestFloatNn nn(levels, FloatNeuralNet::CORNER, &options);
	UT_NOERROR(nn.getErrors());

	// Do the training.

	long seed = random_seed;
	// seed = 1667553859;
	printf("Seed: %ld\n", seed);
	srand48(seed);
	nn.randomize();

	if (debug) {
		nn.printSimpleDump(utest);
	}

	double rate = 0.1; // will be divided by nSteps
	int nPasses = 1000;
	int printEvery = nPasses / 10;

	nn.trainXorFunction(utest, nPasses, printEvery, /*trainRate*/ 0., /*applyRate*/ rate, /*reclaim*/ true);
	// nn.trainXorFunction(utest, nPasses, printEvery, /*trainRate*/ rate, /*applyRate*/ 0., /*reclaim*/ true);
	// nn.trainXorFunction(utest, nPasses, printEvery, /*trainRate*/ rate/2., /*applyRate*/ rate/2., /*reclaim*/ true);

	// Print the new values and compute their stats.
	printf("--- After training:\n");
	nn.printXorFunction(utest, nullptr, nullptr);

	nn.printSimpleDump(utest);

	// Bad seed 1667553859 : similar to 1667554167 had issues with the left side
	// getting driven towards 0, but this time not exactly, so it just converged very slowly.
	// --- L1
	// L1 0 : ( + 0:-0.182412 + 1:-0.180136 + B:0.168660 [C 0.010000:+ 0.000000:1.000000])
	//
	// Bad seed 1667554167: 
	// The training values happen to drive the left side of the CORNER to 0 on the
	// very first pass, and the initial breaking point happens to put everything on
	// the left side, so if the output derivative is mixed into the computation of 
	// the breaking point (which is now fixed to not be done), everything gets stuck at
	// L1 0 : ( + 0:0.222516 + 1:-0.162416 + B:-0.600964 [C 0.000000:+ 0.000000:1.000000])
}
