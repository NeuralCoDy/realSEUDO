//
// (C) Copyright 2022 Sergey A. Babkin.
// This file is a part of Triceps.
// See the file COPYRIGHT for the copyright notice and license information
//
//
// Test wrapper for the floating-point simple NN implementation.

#include <string.h>
#include <cmath>
#include <time.h>
#include <memory>
#include <nn/FloatNeuralNet.h>

static const float eps = 1e-5; // precision for float comparisons

class TestFloatNn : public FloatNeuralNet
{
public:
	// Inherit the constructors.
	using FloatNeuralNet::FloatNeuralNet;

	Erref computeX(const ValueVector &inputs, ValueVector &traindata)
	{
		return computeForTraining(inputs, traindata);
	}

	std::vector<size_t> &levelTrainIdx()
	{
		return level_train_idx_;
	}

	std::vector<size_t> &levelWeightsIdx()
	{
		return level_weights_idx_;
	}

	// Find the weight that connects an input neuron in a previous level 
	// to a neuron in the given level.
	// 
	// @param level - Level of neuron to examine
	// @param neuron - index neuron to examine within level
	// @param input - index of input neuron within previous level to examine,
	//   can be equal to the number of neurons in the previous level to access
	//   the bias value (one that always thas the constant of 1 as input)
	Value &weightAt(size_t level, size_t neuron, size_t input)
	{
		if (level <= 0 ||  level >= level_usage_idx_.size()) {
			fprintf(stderr, "weightAt: Invalid level %zd\n", level);
			abort();
		}

		size_t idx = level_weights_idx_[level] + input * levels_[level] +  neuron;
		if (idx >= level_weights_idx_[level + 1]) {
			fprintf(stderr, "weightAt: Invalid index %zd from (%zd, %zd), limit %zd\n", idx, neuron, input, level_weights_idx_[level + 1]);
			abort();
		}

		// fprintf(stderr, "DEBUG (%zd, %zd, %zd) weights[%zd] = %f\n", level, neuron, input, idx, weights_[idx]);
		return weights_[idx];
	}

	std::vector<size_t> &levelUsageIdx()
	{
		return level_usage_idx_;
	}

	std::vector<uint32_t> &usage()
	{
		return usage_;
	}


	uint32_t &usageAt(size_t level, size_t neuron)
	{
		if (level <= 0 ||  level >= level_usage_idx_.size()) {
			fprintf(stderr, "usageAt: Invalid level %zd\n", level);
			abort();
		}

		size_t idx = level_usage_idx_[level] + neuron;
		if (idx >= level_usage_idx_[level + 1]) {
			fprintf(stderr, "usageAt: Invalid index %zd from (%zd), limit %zd\n", idx, neuron, level_usage_idx_[level + 1]);
			abort();
		}

		return usage_[idx];
	}

	// Do the training for a commonly reused example: a square function in range [0, 1].
	// It can handle the NNs with both 1 and 2 outputs.
	//
	// @param utest - for error reporting
	// @param nPasses - number of training passes
	// @param printEvery - print the stats every so many passes
	// @param nSteps - number of steps for every pass, the range [0, 1] will be divided into
	//   this many intervals (so, with 0 included, it's technically one more step)
	// @param trainRate - rate for train()
	// @param applyRate - rate for applyGradient()
	// @param doReclaim - whether to reclaim the unused neurons
	void trainSquareFunction(Utest *utest, int nPasses, int printEvery, int nSteps, Value trainRate, Value applyRate, bool doReclaim)
	{
		ValueVector ins(1);
		ValueVector outs(levels_.back(), 0.);

		Value stepSize = 1./nSteps;

		for (int pass = 0; pass < nPasses; pass++) {
			startTrainingPass();
			for (int step = 0; step <= nSteps; step++) {
				ins[0] = stepSize * step;
				FloatNeuralNet::Value trainval = ins[0] * ins[0];
				outs[0] = trainval;
				if (UT_NOERROR(train(ins, outs, trainRate)))
					return;
			}
			endTrainingPass();

			if (applyRate != 0.) {
				bool undone;
				applyGradient(applyRate, &undone);
				if (undone) {
					double meansq_err_nn = getSquaredError();
					double grad_all, grad_last;
					UT_NOERROR(getLastGradientNorm2(grad_last));
					UT_NOERROR(getAllGradientNorm2(grad_all));
					printf("      pass %4d, meansq err %f, gradient last %f all %f\n", pass, meansq_err_nn, grad_last, grad_all);

					pass -= 2;
					continue;
				}
			}

			if ((pass + 1) % printEvery == 0) {
				double meansq_err_nn = getSquaredError();
				double grad_all, grad_last;
				UT_NOERROR(getLastGradientNorm2(grad_last));
				UT_NOERROR(getAllGradientNorm2(grad_all));
				printf("      pass %4d, meansq err %f, gradient last %f all %f\n", pass, meansq_err_nn, grad_last, grad_all);
				printSimpleDump(utest);
				// printSquareFunction(utest, nullptr, nullptr);
			}

			// Auto-adjust auto-detected training scale.
			if ((pass + 1) % 1000 == 0) {
				// options_.trainingRateScale_ *= 0.3;
			}

			if (doReclaim) {
				size_t reclaimed = 0;
				if (UT_NOERROR(reclaim(0, reclaimed)))
					return;
				if (reclaimed > 0) {
					printf("reclaimed %zd neurons on pass %d\n", reclaimed, pass);
					// printSimpleDump(utest);
				}
			}
		}
	}

	// For a square function, do a single round of traiing with rate 0
	// to compute the stats, and print them.
	//
	// @param utest - for error reporting
	// @param nSteps - number of steps for every pass, the range [0, 1] will be divided into
	//   this many intervals (so, with 0 included, it's technically one more step)
	void printSquareStats(Utest *utest, int nSteps)
	{
		trainSquareFunction(utest, /*nPasses*/ 1, /*printEvery*/ 100, nSteps, /*trainRate*/ 0., /*applyRate*/ 0., /*reclaim*/ false);
		double mean_err_nn = getAbsError();
		double meansq_err_nn = getSquaredError();
		double grad_all, grad_last;
		UT_NOERROR(getLastGradientNorm2(grad_last));
		UT_NOERROR(getAllGradientNorm2(grad_all));
		printf("    Mean error: %g, mean squared error: %g, gradient: last %g all %g\n",
			mean_err_nn, meansq_err_nn, grad_last, grad_all);
	}

	// For a square function, print the function on 10 intervals,
	// and compute the errors.
	//
	// @param utest - for error reporting
	// @param meanErr - place to return the mean error, or nullptr
	// @param meanSqErr - place to return the mean square error, or nullptr
	void printSquareFunction(Utest *utest, double *meanErr, double *meanSqErr)
	{
		ValueVector ins(1);
		ValueVector outs(levels_.back(), 0.);
		size_t highest;
		double mean_err = 0., meansq_err = 0.;
		Value lastout;
		{
			// To compute the difference at 0, compute the value at -0.1.
			ins[0] = -0.1;
			UT_NOERROR(compute(ins, outs, highest));
			lastout = outs[0];
		}
		for (int step = 0; step <= 10; step++) {
			ins[0] = 0.1 * step;
			FloatNeuralNet::Value trainval = ins[0] * ins[0];
			UT_NOERROR(compute(ins, outs, highest));
			printf("  %.1f -> %10f (true %10f) d=%f\n", ins[0], outs[0], trainval, outs[0]-lastout);
			lastout = outs[0];
			double err = outs[0] - trainval;
			mean_err += abs(err);
			meansq_err += err * err;;
		}
		mean_err /= 11.;
		meansq_err = sqrt(meansq_err / 11.);

		if (meanErr != nullptr)
			*meanErr = mean_err;
		if (meanSqErr != nullptr)
			*meanSqErr = meansq_err;
	}

	// Print a simple dump of the weights.
	//
	// @param utest - for error reporting
	void printSimpleDump(Utest *utest)
	{
		Erref err = simpleDump();
		UT_ASSERT(!err.isEmpty());
		printf("%s", err->print("    ").c_str());
	}

	// Do the training for a XOR function, with 2 inputs and 1 output.
	//
	// @param utest - for error reporting
	// @param nPasses - number of training passes
	// @param printEvery - print the stats every so many passes
	// @param trainRate - rate for train(), it will be divided by the number of examples in a pass
	// @param applyRate - rate for applyGradient(), it will be divided by the number of examples in a pass
	// @param doReclaim - whether to reclaim the unused neurons
	void trainXorFunction(Utest *utest, int nPasses, int printEvery, Value trainRate, Value applyRate, bool doReclaim)
	{
		ValueVector ins(2);
		ValueVector outs(1);

		int nSteps = 4;
		trainRate /= nSteps;
		applyRate /= nSteps;

		for (int pass = 0; pass < nPasses; pass++) {
			startTrainingPass();
			for (int a = -1; a <= 1; a += 2) {
				for (int b = -1; b <= 1; b += 2) {
					ins[0] = (Value) a;
					ins[1] = (Value) b;
					outs[0] = (a == b? -1. : 1.);
					if (UT_NOERROR(train(ins, outs, trainRate)))
						return;
				}
			}
			endTrainingPass();

			if (applyRate != 0.) {
				bool undone;
				applyGradient(applyRate, &undone);
				if (undone) {
					pass -= 2;
					continue;
				}
			}

			if ((pass + 1) % printEvery == 0) {
				double meansq_err_nn = getSquaredError();
				double grad_all, grad_last;
				UT_NOERROR(getLastGradientNorm2(grad_last));
				UT_NOERROR(getAllGradientNorm2(grad_all));
				printf("      pass %4d, meansq err %f, gradient last %f all %f\n", pass, meansq_err_nn, grad_last, grad_all);
				// printSimpleDump(utest);
			}

			if (doReclaim) {
				size_t reclaimed = 0;
				if (UT_NOERROR(reclaim(0, reclaimed)))
					return;
				if (reclaimed > 0) {
					printf("reclaimed %zd neurons on pass %d\n", reclaimed, pass);
					// printSimpleDump(utest);
				}
			}
		}
	}

	// For a XOR function, print the function and compute the errors.
	//
	// @param utest - for error reporting
	// @param meanErr - place to return the mean error, or nullptr
	// @param meanSqErr - place to return the mean square error, or nullptr
	void printXorFunction(Utest *utest, double *meanErr, double *meanSqErr)
	{
		ValueVector ins(2);
		ValueVector outs(1);
		size_t highest;
		double mean_err = 0., meansq_err = 0.;
		int nSteps = 4;

		for (int a = -1; a <= 1; a += 2) {
			for (int b = -1; b <= 1; b += 2) {
				ins[0] = (Value) a;
				ins[1] = (Value) b;
				Value expected = (a == b? -1. : 1.);
				if (UT_NOERROR(compute(ins, outs, highest)))
					return;
				printf("  %.1f ^ %.1f -> %10f (true %10f)\n", ins[0], ins[1], outs[0], expected);
				double err = outs[0] - expected;
				mean_err += abs(err);
				meansq_err += err * err;;
			}
		}
		mean_err /= nSteps;
		meansq_err = sqrt(meansq_err / nSteps);
		printf("    Mean error: %g, mean squared error: %g\n", mean_err, meansq_err);

		if (meanErr != nullptr)
			*meanErr = mean_err;
		if (meanSqErr != nullptr)
			*meanSqErr = meansq_err;
	}

	// ---------------------------------------------------------------
	// This is generic training on any set of inputs/outputs and test data.

	// A single element of training data.
	struct Training {
		// Allocate vectors with pre-sizing and setting the
		// values of outputs to -1.
		Training(size_t nIn, size_t nOut)
			: input(nIn), output(nOut, -1.)
		{ }

		FloatNeuralNet::ValueVector input;
		FloatNeuralNet::ValueVector output;
		// For the "choose one" classifiers, this provides the index
		// of the label (i.e. usually the value "1" in the output)
		// in a direct form.
		int label = -1;
		// How many copies of this record are in the training set
		size_t count = 1;
		// How many copies of this record are in the training set
		// after adjustment to ensure that all the training cases
		// produce the correct result for a classifier.
		FloatNeuralNet::Value effectiveCount = 1.;
	};

	typedef vector< std::shared_ptr<Training>  > TrainingVector;

	// Do the training for an arbitrary training set, and verify on a test set.
	// The structure of NN inputs and outputs must match.
	// The assumption is that the classification goes into non-overlapping
	// classes, so the highest produced output determines the class.
	//
	// @param utest - for error reporting
	// @param ckpFileName - if not empty, name of checkpoint file that gets written on
	//    every printing
	// @param trainData - training data, going through which will constitute a single pass
	// @param testData - a separate set of test data for verification (or if don't care,
	//   can use the same as trainData)
	// @param nPasses - number of training passes
	// @param printEvery - print the stats every so many passes
	// @param testEvery - test on every so many passes
	// @param trainRate - rate for train(), it will be divided by the number of examples in a pass
	// @param applyRate - rate for applyGradient(), it will be divided by the number of examples in a pass
	// @param doReclaim - whether to reclaim the unused neurons
	// @param batchSize - this many records get averaged for each training, the combination of
	//   records get somewhat shuffled on each pass
	void trainClassifier(Utest *utest, 
		const std::string &ckpFileName,
		const TrainingVector &trainData,
		const TrainingVector &testData,
		int nPasses, int printEvery, int testEvery, Value trainRate, Value applyRate,
		bool doReclaim, int batchSize)
	{
		trainRate /= trainData.size();
		applyRate /= trainData.size();

		for (int pass = 0; pass < nPasses; pass++) {
			size_t nextpass = getTrainingPass() + 1;
			startTrainingPass();
			for (size_t elem = 0; elem < trainData.size(); ++elem) {
				if (batchSize <= 1) {
					if (UT_NOERROR(train(trainData[elem]->input, trainData[elem]->output, trainRate,
							trainData[elem]->count, &trainData[elem]->effectiveCount)))
						return;
				} else {
					Training t = *trainData[elem];
					for (int b = 1; b < batchSize; b++) {
						// The picking of elements in a batch depends on the pass.
						size_t belem = (elem + (1 + b) * nextpass) % trainData.size();
						Training &tb = *trainData[belem];
						for (size_t i = 0; i < t.input.size(); i++) {
							t.input[i] += tb.input[i];
						}
						for (size_t i = 0; i < t.output.size(); i++) {
							t.output[i] += tb.output[i];
						}
					}
					for (size_t i = 0; i < t.input.size(); i++) {
						t.input[i] /= batchSize;
					}
					for (size_t i = 0; i < t.output.size(); i++) {
						t.output[i] /= batchSize;
					}
					if (UT_NOERROR(train(t.input, t.output, trainRate)))
						return;
				}
			}
			endTrainingPass();

			if (applyRate != 0.) {
				bool undone;
				applyGradient(applyRate, &undone);
				if (undone) {
					pass -= 2;
					continue;
				}
			}

			if ((pass + 1) % printEvery == 0) {
				double meansq_err_nn = getSquaredError();
				double eff_meansq_err_nn = getEffectiveSquaredError();
				double grad_all, grad_last;
				if (options_.totalGradient_) {
					UT_NOERROR(getLastGradientNorm2(grad_last));
					UT_NOERROR(getAllGradientNorm2(grad_all));
				} else {
					grad_all = 0.;
					grad_last = 0.;
				}
				printf("      pass %4d [%4d], meansq err %f/%f, gradient last %f all %f; est ert %f below %f\n",
					pass, (int)getTrainingPass(), meansq_err_nn, eff_meansq_err_nn, grad_last, grad_all,
					(double)getPassIncorrect() / (double)(getPassIncorrect() + getPassCorrect()),
					(double)getPassNotAbove() / (double)(getPassNotAbove() + getPassAbove())
					);
				// printSimpleDump(utest);

				if (!ckpFileName.empty()) {
					if (UT_NOERROR(checkpoint(ckpFileName))) {
						return;
					}
				}
			}

			if ((pass + 1) % testEvery == 0) {
				{
					size_t correct = 0, incorrect = 0;
					size_t above = 0, below = 0;
					size_t highest;
					ValueVector outs;
					double meansq_sum = 0.;
					size_t meansq_count = 0;
					for (size_t elem = 0; elem < testData.size(); ++elem) {
						if (UT_NOERROR(compute(testData[elem]->input, outs, highest)))
							return;
						if (testData[elem]->label != highest) {
							++incorrect;
							++below;
						} else {
							++correct;
							if (outs[highest] <= 0.) {
								++below;
							} else {
								++above;
							}
						}
						for (size_t i = 0; i < outs.size(); i++) {
							double v = testData[elem]->output[i] - outs[i];
							meansq_sum += v * v;
							++meansq_count;
						}
					}
					printf("         p %4d, test: meansq err %f, error rate %f,  below %f\n",
						pass, sqrt(meansq_sum/meansq_count), (double)incorrect / (correct + incorrect),
						(double)below / (above + below));
				}

				if ((pass + 1) % (testEvery * 10) == 0) {
					size_t correct = 0, incorrect = 0;
					size_t above = 0, below = 0;
					size_t highest;
					ValueVector outs;
					double meansq_sum = 0.;
					size_t meansq_count = 0;
					for (size_t elem = 0; elem < trainData.size(); ++elem) {
						if (UT_NOERROR(compute(trainData[elem]->input, outs, highest)))
							return;
						if (trainData[elem]->label != highest) {
							++incorrect;
							++below;
						} else {
							++correct;
							if (outs[highest] <= 0.) {
								++below;
							} else {
								++above;
							}
						}
						for (size_t i = 0; i < outs.size(); i++) {
							double v = trainData[elem]->output[i] - outs[i];
							meansq_sum += v * v;
							++meansq_count;
						}
					}
					printf("         p %4d, train: meansq err %f, error rate %f,  below %f\n",
						pass, sqrt(meansq_sum/meansq_count), (double)incorrect / (correct + incorrect),
						(double)below / (above + below));
				}
			}

			if (doReclaim) {
				size_t reclaimed = 0;
				if (UT_NOERROR(reclaim(0, reclaimed)))
					return;
				if (reclaimed > 0) {
					printf("reclaimed %zd neurons on pass %d\n", reclaimed, pass);
					// printSimpleDump(utest);
				}
			}
		}
	}
};
