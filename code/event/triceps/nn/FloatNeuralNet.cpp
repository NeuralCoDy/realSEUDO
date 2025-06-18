// (C) Copyright 2022 Sergey A. Babkin.
// This file is a part of Triceps.
// See the file COPYRIGHT for the copyright notice and license information
//
//
// A simple neural network implementation with float arithmetic.

#include <common/Strprintf.h>
#include <nn/FloatNeuralNet.h>
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

namespace TRICEPS_NS {

/////////////////////////////////// FloatNeuralNet ////////////////////////////////////////////

FloatNeuralNet::FloatNeuralNet(const LevelSizeVector &levels, ActivationFunction activation,
		Options *options) :
	activation_(activation),
	levels_(levels), level_train_idx_(levels.size() + 1),
	level_weights_idx_(levels.size() + 1),
	level_corners_idx_(levels.size() + 1),
	level_usage_idx_(levels.size() + 1)
{
	if (levels.size() < 2) {
		errors_.f("The level count %zd is invalid, minimum is 2", levels.size());
		return;
	}

	if (levels[0] <= 0) {
		errors_.f("Invalid size %" PRId32 " of level 0.", levels[0]);
		return;
	}

	if (options != nullptr) {
		options_ = *options;
	}
	reconcileOptions();

	level_weights_idx_[0] = 0;
	level_weights_idx_[1] = 0; // level 0 contains no neurons.

	level_corners_idx_[0] = 0;
	level_corners_idx_[1] = 0; // level 0 contains no neurons.

	for (size_t i = 1; i < levels.size(); i++) {
		if (levels[i] <= 0) {
			errors_.f("Invalid size %" PRId32 " of level %zd.", levels[0], i);
			return;
		}
		
		// Each neuron has a weight for each neuron of the previous level, plus
		// 1 extra weight as bias (it can be seen as receiving the pseudo-input
		// of constant 1).
		LevelSize sz = (size_t)levels[i] * (levels[i - 1] + 1);
		level_weights_idx_[i + 1] = level_weights_idx_[i] + sz;
		if  (sz < 0 || level_weights_idx_[i + 1] < level_weights_idx_[i]) {
			errors_.f("The requested size of weight table is so large that it overflowed.");
			return;
		}

		if (activation_ == CORNER)
			level_corners_idx_[i + 1] = level_corners_idx_[i] + CW_SIZE * levels[i];
		else
			level_corners_idx_[i + 1] = 0;
	}

	level_train_idx_[0] = 0;
	level_usage_idx_[0] = 0;
	for (size_t i = 0; i < levels.size(); i++) {
		// All the checks were already done with level_weights_idx_.

		// For each neuron we store the raw sum of weights and the activated value.
		LevelSize sz = (size_t)levels[i] * 2;
		level_train_idx_[i + 1] = level_train_idx_[i] + sz;

		if (activation == RELU) {
			level_usage_idx_[i + 1] = level_usage_idx_[i] + levels[i];
		} else {
			level_usage_idx_[i + 1] = 0;
		}
	}

	try {
		weights_.assign(weightsVectorSize(), 0.);
		corners_.assign(cornersVectorSize(), 0.);
		usage_.assign(usageVectorSize(), 0);
		if (options_.momentum_ || options_.autoRate2_) {
			mdiff_weights_ = weights_;
			mdiff_corners_ = corners_;
		}
	} catch (Exception &e) {
		errors_.f("The requested size of weight table %zd doesn't fit into memory.", level_weights_idx_.back());
		return;
	}
	
	if (activation_ == CORNER) {
		// By default, initialize the corner weights to make the corners
		// resemble Leaky ReLU.
		for (int i = 0; i < corners_.size(); i+= CW_SIZE) {
			corners_[i + CW_LEFT] = options_.reluLeakiness_;
			corners_[i + CW_OFFSET] = 0.;
			corners_[i + CW_RIGHT] = 1.;
		}
	}
}


// Generation of a nicer random value. Since the values near 0 have a lot of
// trouble with getting out of this proximity in training, make sure to
// initialize with the values that are not too small. Returns a value in
// the range: (-absmax, -absmin] union [absmin, absmax).
static inline FloatNeuralNet::Value random_value(FloatNeuralNet::Value absmin, FloatNeuralNet::Value absmax)
{
	// drand48() returns a value in range [0, 1) but we want the negative
	// values too, so shift it to [-1, 1). Then scale them to the limit.
	FloatNeuralNet::Value val = (drand48() * 2. - 1.) * (absmax - absmin);

	// Then increase the absolute value by absmin.
	if (val < 0)
		val -= absmin;
	else
		val += absmin;
	return val;
}

// Pick a bias value. This is just a convenient way to keep the computation
// in one place.
//
// @param absmax - absolute maximum value for the "normal" random weights.
//   This function is free to use or not use it.
static inline FloatNeuralNet::Value pick_bias_value(FloatNeuralNet::Value absmax)
{
	// https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/
	// recommends a fixed value of 0.1, but for all I can tell, this is a half-assed
	// way to pre-reclaim the neurons that always produce a negative value. With
	// the explicit reclaiming logic, a random value seems to work better.
	// 
	// return  FloatNeuralNet::DEFAULT_BIAS;
	return random_value(absmax * 0.1, absmax);
}

void
FloatNeuralNet::randomize()
{
	for (size_t level = 1; level < levels_.size(); level++) {
		size_t levsz = (size_t)levels_[level];
		size_t prevsz = levels_[level - 1];

		// Position of weights
		size_t weightpos = level_weights_idx_[level];

		// The input weights get scaled per the number of inputs of neuron.
		// See https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/
		const Value limit = 1. / sqrt(prevsz);

		size_t max = prevsz * levsz;
		for (size_t idx = 0; idx < max; idx++) {
			weights_[weightpos + idx] = random_value(0.1 * limit, limit);
			// printf("DEBUG   L %zd rand[%zd] = %f\n", level, weightpos + idx, weights_[weightpos + idx]);
		}

		// Now fill in the bias.
		weightpos += max;
		for (size_t idx = 0; idx < levsz; idx++) {
			weights_[weightpos + idx] = pick_bias_value(limit);
			// printf("DEBUG   L %zd rand[%zd] = %f\n", level, weightpos + idx, weights_[weightpos + idx]);
		}
	}
}

Erref
FloatNeuralNet::setLevel(size_t level, const ValueVector &coeff)
{
	Erref err;
	if (errors_.hasError()) {
		err.fAppend(errors_, "NN failed to initialize:");
		return err;
	}

	if (level > lastLevel()) {
		err.f("No level %zd in the NN, supported levels are 0..%zd.", level, lastLevel());
		return err;
	}

	size_t levelsz = level_weights_idx_[level + 1] - level_weights_idx_[level];

	if (coeff.size() != levelsz) {
		err.f("Invalid data size for level %zd, must contain %zd elements, contains %zd.", level, levelsz, coeff.size());
		return err;
	}

	auto srcit = coeff.begin();
	auto dstit = weights_.begin() + level_weights_idx_[level];
	while (srcit != coeff.end()) {
		*dstit++ = *srcit++;
	}
	return err;
}

Erref
FloatNeuralNet::getLevel(size_t level, ValueVector &coeff) const
{
	Erref err;
	if (errors_.hasError()) {
		err.fAppend(errors_, "NN failed to initialize:");
		return err;
	}

	if (level > lastLevel()) {
		err.f("No level %zd in the NN, supported levels are 0..%zd.", level, lastLevel());
		return err;
	}

	size_t levelsz = level_weights_idx_[level + 1] - level_weights_idx_[level];

	coeff.resize(levelsz);

	auto srcit = weights_.begin() + level_weights_idx_[level];
	auto dstit = coeff.begin();
	while (dstit != coeff.end()) {
		*dstit++ = *srcit++;
	}
	return err;
}

// ----------------------- RELU ------------------------------

// The ReLU activation function.
static inline FloatNeuralNet::Value
relu(FloatNeuralNet::Value val)
{
	if (val < 0.)
		return 0.;
	else
		// Don't limit the value to 1, because that would prevent the
		// automatic scaling of the values to fit into the [0, 1] range,
		// because the derivative above 1 would be 0 and would not allow
		// to do a proper backpropagation.
		return val;
}

// Apply the ReLU activation function.
// @param lastLevel - flag: this is the last level, just copy the values
// @param inactiv - the inactivated values
// @param activ - place to put the activated values
static void
activateRelu(bool lastLevel, FloatNeuralNet::ValueSubVector inactiv, FloatNeuralNet::ValueSubVector activ)
{
	if (lastLevel) {
		memmove(activ.data_, inactiv.data_, inactiv.size() * sizeof(*activ.data_));
	} else  {
		for (size_t cur = 0; cur < inactiv.size(); cur++) {
			activ[cur] = relu(inactiv[cur]);
			// fprintf(stderr, "DEBUG   data[level, %zd] = %f <- relu(data[%zd, %zd] = %f)\n", cur, activ[cur], level, cur, inactiv[cur]);
		}
	}
}

// The derivative of ReLU activation function at a point.
static inline FloatNeuralNet::Value
relu_der(FloatNeuralNet::Value val)
{
	if (val < 0.)
		return 0.;
	else
		return 1.;
}

// Compute activated derivatives for the outpus of a level, for ReLU.
// @param level - level whose output derivatives are being computed.
// @param inactiv - inactivated outputs of the level.
// @param der - place to put the computed derivatives, will be resized
//   as needed.
static void
computeReluDerivatives(size_t level, FloatNeuralNet::ValueSubVector inactiv, FloatNeuralNet::ValueVector &der)
{
	der.resize(inactiv.size());
	for (size_t i = 0; i < inactiv.size(); i++) {
		der[i] = relu_der(inactiv[i]);
	}
}

// ----------------------- LEAKY_RELU ------------------------

// The Leaky ReLU activation function.
// @param val - inactivated value
// @param left_der - derivative for the left side
static inline FloatNeuralNet::Value
leaky_relu(FloatNeuralNet::Value val, FloatNeuralNet::Value left_der)
{
	if (val < 0.)
		return val * left_der;
	else
		// Don't limit the value to 1, because that would prevent the
		// automatic scaling of the values to fit into the [0, 1] range,
		// because the derivative above 1 would be 0 and would not allow
		// to do a proper backpropagation.
		return val;
}

// Apply the Leaky ReLU activation function.
// @param lastLevel - flag: this is the last level, just copy the values
// @param inactiv - the inactivated values
// @param activ - place to put the activated values
// @param left_der - derivative for the left side
static void
activateLeakyRelu(bool lastLevel, FloatNeuralNet::ValueSubVector inactiv, FloatNeuralNet::ValueSubVector activ, FloatNeuralNet::Value left_der)
{
	if (lastLevel) {
		memmove(activ.data_, inactiv.data_, inactiv.size() * sizeof(*activ.data_));
	} else  {
		for (size_t cur = 0; cur < inactiv.size(); cur++) {
			activ[cur] = leaky_relu(inactiv[cur], left_der);
			// fprintf(stderr, "DEBUG   data[level, %zd] = %f <- lkrelu(data[%zd, %zd] = %f)\n", cur, activ[cur], level, cur, inactiv[cur]);
		}
	}
}

// The derivative of Leaky ReLU activation function at a point.
// @param val - inactivated value
// @param left_der - derivative for the left side
static inline FloatNeuralNet::Value
leaky_relu_der(FloatNeuralNet::Value val, FloatNeuralNet::Value left_der)
{
	if (val < 0.)
		return left_der;
	else
		return 1.;
}

// Compute activated derivatives for the outpus of a level, for LEAKY_RELU.
// @param level - level whose output derivatives are being computed.
// @param inactiv - inactivated outputs of the level.
// @param der - place to put the computed derivatives, will be resized
//   as needed.
// @param left_der - derivative for the left side
static void
computeLeakyReluDerivatives(size_t level, FloatNeuralNet::ValueSubVector inactiv,
	FloatNeuralNet::ValueVector &der, FloatNeuralNet::Value left_der)
{
	der.resize(inactiv.size());
	for (size_t i = 0; i < inactiv.size(); i++) {
		der[i] = leaky_relu_der(inactiv[i], left_der);
	}
}

// ----------------------- CORNER ----------------------------

void
FloatNeuralNet::activateCorner(size_t level, ValueSubVector inactiv, ValueSubVector activ)
{
	ValueSubMatrix corners = cornersOfLevel(level);
	for (size_t i = 0; i < inactiv.size(); i++) {
		Value v = inactiv[i];
		if (v < 0.) {
			v *= corners.at(i, CW_LEFT);
		} else if (v > 0.) {
			v *= corners.at(i, CW_RIGHT);
		}
		v += corners.at(i, CW_OFFSET);
		activ[i] = v;
		// fprintf(stderr, "DEBUG   data[%zd, %zd] = %f <- corner(%f)\n", level, i, v, inactiv[i]);
	}
}

void
FloatNeuralNet::backpropCorners(size_t level, ValueSubVector inactiv,
	ValueVector &der, const ValueVector &sigma, Value rate, bool debug)
{
	// debug = true;
	bool do_total_gradient = (totalCornersGradient_.size() == cornersVectorSize());

	der.resize(inactiv.size());
	ValueSubMatrix corners = cornersOfLevel(level);
	ValueSubMatrix cgrad = cornersGradientOfLevel(level);

	for (size_t i = 0; i < inactiv.size(); i++) {
		Value input = inactiv[i];

		if (debug) printf("DEBUG    level %zd[%zd] corner input=%f sigma=%f slope_gradient=%f rate=%f\n",
			level, i, input, sigma[i], input * sigma[i], rate);

		if (input < 0.) {
			auto w = &corners.at(i, CW_LEFT);
			der[i] = *w;

#if 1 // {
			// Disable this part to keep the left side at fixed coefficient of leaky ReLU.
			Value gradient = input * sigma[i];
			if (do_total_gradient) {
				cgrad.at(i, CW_LEFT) += gradient;
			}

			if (rate != 0.) {
				Value newwt = *w - rate * gradient;
				saturate(newwt, *w);

				if (debug) printf("DEBUG    level %zd[%zd] corner left grad=%f weihgt=%f->%f\n",
					level, i, gradient, *w, newwt);

				*w = newwt;
			}
#endif // }
		} else if (input > 0.) {
			auto w = &corners.at(i, CW_RIGHT);
			der[i] = *w;

#if 1 // {
			// Disable this part to keep the right side at fixed coefficient of 1.
			// In theory it shouldn't matter, as the coefficients for the
			// linear regression for the basic neuron would be adjusted instead.
			// But as it turns out, in reality it does matter in case if the
			// sign of the right side should be going negative. Then the coefficients
			// don't flip their sign easily, and everything gets stuck in a bad position.
			// So disabling this part makes things a lot worse.

			Value gradient = input * sigma[i];
			if (do_total_gradient) {
				cgrad.at(i, CW_RIGHT) += gradient;
			}

			if (rate != 0.) {
				Value newwt = *w - rate * gradient;
				saturate(newwt, *w);

				if (debug) printf("DEBUG    level %zd[%zd] corner right grad=%f weihgt=%f->%f\n",
					level, i, gradient, *w, newwt);

				*w = newwt;
			}
#endif // }
		} else {
			// At the corner pick the average derivative.
			der[i] = (corners.at(i, CW_LEFT) + corners.at(i, CW_RIGHT)) / 2.;
			// Don't touch the multipliers.
		}

		// Update the offset
#if 1 
		// Disable this part to avoid any shift, allowing to reproduce Leaky ReLU
		// but with a different computation of the breaking point.
		{
			auto *w = &corners.at(i, CW_OFFSET);

			// Offset has the pseudo-input of 1.
			Value gradient = sigma[i];
			if (do_total_gradient) {
				cgrad.at(i, CW_OFFSET) += gradient;
			}

			if (rate != 0.) {
				Value newwt = *w - rate * gradient;
				saturate(newwt, *w);

				if (debug) printf("DEBUG    level %zd[%zd] corner offset grad=%f weihgt=%f->%f\n",
					level, i, gradient, *w, newwt);

				*w = newwt;
			}
		}
#else
		// make the error on the unused variables shut up
		if (do_total_gradient) {
			cgrad.at(i, CW_OFFSET) += 0.;
		}
#endif
	}
}

// -----------------------------------------------------------

void
FloatNeuralNet::backpropLevel(size_t level, ValueVector &s_cur, const ValueVector &s_next,
	const ValueVector &der, ValueSubVector prev_activ, ValueSubVector inactiv,
	Value rate, bool debug)
{
	// Since the offsets don't benefit from further backpropagation,
	// their rate can be boosted by this factor.
	static const Value boost_offsets = 1.;

	bool do_total_gradient = (totalGradient_.size() == weightsVectorSize());

	size_t levsz = (size_t)levels_[level];

	s_cur.resize(prev_activ.size());

	// Position of weights, that go in the right order for this loop.
	ValueSubMatrix lweights = weightsOfLevel(level);

	ValueSubMatrix lgrads;
	if (do_total_gradient) {
		lgrads = gradientOfLevel(level);
	}

	for (size_t prev = 0; prev < prev_activ.size(); prev++)  {
		Value input = prev_activ[prev];

		// Sum of weights for computing sigma of this input.
		Value wsigma = 0.;

		ValueSubVector subweights = lweights.rowAsVector(prev);
		for (size_t i = 0; i < levsz; i++)  {
			// Use the original weight!
			wsigma += subweights[i] * s_next[i] * der[i];

			Value gradient = input * s_next[i] * der[i];
			if (do_total_gradient) {
				lgrads.at(prev, i) += gradient;
			}

			if (rate != 0.) {
				Value newwt = subweights[i] - rate * gradient;
				saturate(newwt, subweights[i]);

				if (debug) printf("DEBUG    level %zd[%zd][p=%zd] grad=%f weihgt=%f->%f\n",
					level, i, prev, gradient, subweights[i], newwt);

				subweights[i] = newwt;
			}
		}

		s_cur[prev] = wsigma;
	}

	// Then go the weights for the biases that have the "hardcoded
	// pseudo-input" 1.
	ValueSubVector subweights = lweights.rowAsVector(prev_activ.size());
	for (size_t i = 0; i < levsz; i++)  {
		// Since the input is 1, gradient is s_next[i].
		Value gradient = s_next[i] * der[i];
		if (activation_ == CORNER) {
			// This attempts to balance between two things:
			// (1) propagating the gradient through;
			// (2) getting the breaking point into a position where its own
			//     balance exists between the training cases on the left and
			//     right sides of it. The major idea is that we have the data
			//     points falling far from the model line, we might be able to
			//     fit them better if the line had a higher slope, and to do
			//     a higher slope without going well above or below all the points,
			//     we have to move the breaking point closer to them. So whichever
			//     side has a greater divergence, we move the breaking point that
			//     way (and the gradient is opposite to the movement direction).
			//     Here abs(s_next[i]) is used as an indicator of that distance
			//     of the point from the produced line. I've tried it in two ways,
			//     one mixes this with the traditional signal
			//     and just uses a fixed multiplier for abs(s_next[i])
			//     (and just in case if the s_next is too weak, it's augmented
			//     by a constant), another one takes this weight proportional
			//     to the output, which gives the greater leverage to the points
			//     farther out. The second one is probably better, because a
			//     divergence farther out calls for a greater move that way,
			//     and when the move of the breaking point causes a point to move
			//     to another side, it should cause only a small change,
			//     and that seems to work better in practice.
			//
			// It kind of works empirically, but I haven't figured out the theory
			// for it yet. And when there are many neurons, they take a while
			// to sort things out between themsevles, and the optimization surface
			// is not smooth, once in a while the error drops but gradient grows.
			// This needs more work to figure out. Maybe it just needs an
			// auto-adjustable step. But more likely, the problem is that moving
			// the breaking point sideways is not coordinated with the vertical
			// moves in the activation function, so a small improvement in
			// horizontal position can cause a large deficiency in the vertical
			// position, which then takes a while to catch up to and improve.
			//
			// This balance thing is probably related to the way how boosting
			// tries on each step to pick a partial function that takes the most
			// discriminating property, and after applying this function, the
			// property becomes balanced.
			//
			// Either way, we don't apply der[i], because (1) it should not
			// affect the weights, and (2) if the derivative goes to 0, it would
			// nullify the weights and that would be wrong.
#if 0
			if (inactiv[i] > 0.)
				gradient = 0.6 * s_next[i] + 0.4 * abs(s_next[i]) + 0.0001;
			else if (inactiv[i] < 0.)
				gradient = 0.6 * s_next[i] - 0.4 * abs(s_next[i]) - 0.0001;
#else
			gradient = inactiv[i] * abs(s_next[i]);
#endif
		}
		if (do_total_gradient) {
			lgrads.at(prev_activ.size(), i) += gradient;
		}

		if (rate != 0.) {
			Value newwt = subweights[i] - boost_offsets * rate * gradient;
			saturate(newwt, subweights[i]);

			if (debug) printf("DEBUG    level %zd[%zd][bias] grad=%f weihgt=%f->%f\n",
				level, i, gradient, subweights[i], newwt);

			subweights[i] = newwt;
		}
	}
}

// Find the highest element in a vector if it's unanimously highest. This logic
// is used to decide if a classifier training case has produced the correct result.
//
// @param vec - vector to examine
// @return - index of the highest element; if there is more than one element
//   with the same value, returns -1.
static int findHighest(const FloatNeuralNet::ValueSubVector vec) {
	FloatNeuralNet::Value highval = vec[0];
	int highest = 0;
	bool unique = true;

	for (size_t i = 1; i < vec.size(); i++) {
		FloatNeuralNet::Value val = vec[i];
		if (val > highval) {
			highval = val;
			highest = i;
			unique = true;
		} else if (val == highval) {
			unique = false;
		}
	}

	if (unique)
		return highest;
	else
		return -1;
}

Erref
FloatNeuralNet::train(const ValueVector &inputs, const ValueVector &outputs, Value rate,
	size_t count, Value *effectiveCount)
{
	Erref err;

	// Should become an argument?
	static constexpr bool debug = false;

	if (debug) printf("DEBUG --- train()\n");

	if (count < 1) {
		err.f("The count argument must not be 0.");
		return err;
	}
	if (effectiveCount != nullptr && *effectiveCount < 1) {
		err.f("The value at effectiveCount argument must not be 0.");
		return err;
	}

	// For auto-rate computations, only applyGradient() does the moves.
	if (options_.autoRate_)
		rate = 0.;

	if (options_.momentum_) {
		if (rate != 0.) {
			err.f("Option momentum_ may not be used with non-0 rate in train().");
			return err;
		}
	}

	// Compute the NN on the inputs.

	ValueVector traindata;
	err.fAppend(computeForTraining(inputs, traindata), "When computing a training example:");
	if (err.hasError()) {
		return err;
	}

	// Allows to pretend that the case was processed the effective count of times.
	Value gradientScale = (effectiveCount == nullptr? count : *effectiveCount);

	if (options_.isClassifier_) {
		// Check if the result is correct
		ValueSubVector actualOutput = trainDataActivated(traindata, lastLevel());
		int actualResult = findHighest(actualOutput);
		ValueSubVector expectedOutput(const_cast<Value *>(outputs.data()), 0, outputs.size());
		int expectedResult = findHighest(expectedOutput);
		if (actualResult == expectedResult) {
			correctCases_ += count;
			if (actualOutput[actualResult] > 0.) {
				aboveCases_ += count;
			} else {
				notAboveCases_ += count;
			}

			if (effectiveCount != nullptr
			&& *effectiveCount > count) {
				Value c = *effectiveCount - count * options_.effecttiveDecreaseRate_;
				if (c < count)
					c = count;
				gradientScale = *effectiveCount = c;
			}
		} else {
			incorrectCases_ += count;
			notAboveCases_ += count;

			if (trainingPass_ > options_.minPassEffectiveAdj_
			&& effectiveCount != nullptr
			&& *effectiveCount < count * options_.maxMultiplier_) {
				gradientScale = (*effectiveCount += count);
			}
		}
	}

	// Prepare the backpropagation data storage.

	// Sensitivity of the next and current level' input.
	ValueVector s_next; // sigma^{L+1} in the whitepaper
	ValueVector s_cur; // sigma^{L} in the whitepaper
	// Start from the last level back.
	size_t level = lastLevel();

	// Derivatives of activation function will be placed here.
	ValueVector der;

	// Will start by computing the error, which also happens to be the
	// sensitivity of the "level just past the last real level".

	// Last level's result in the training data.
	ValueSubVector prev_activ = trainDataActivated(traindata, level);

	s_cur.resize(prev_activ.size());

	if (outputs.size() != prev_activ.size()) {
		err.f("Invalid training output data size %zd, expected %zd", outputs.size(), prev_activ.size());
		return err;
	}

	// Now do the backpropagation.

	// Start from "behind" the last level, it's special because it's where the
	// comparison to the expected outputs happens.
	for (size_t i = 0; i < prev_activ.size(); i++) {
		// For classic activation functions, the last layer just copies data
		// instead of activation.
		Value out = prev_activ[i];

		// The constant 2 comes from the derivative: if e is the error
		// defined as e = (I - t)^2, t is the true value, and c is the signal
		// after the activation function, and I is the signal before the
		// activation function:
		//
		//   sigma = de/dc = d( (c-t)^2 )/dc = 2(c - t)
		//
		// (there is normally no activation function after the last level)
		// which translates to
		//
		//   s_cur[i] = 2 * (out - outputs[i]);
		//
		// which then gets backpropagated through
		//
		//   sigma_{i} = A'(I) * sigma_{i+1}
		//
		double err = out - outputs[i];
		s_cur[i] = 2. * err * gradientScale;

		if (debug) printf("DEBUG level %zd[%zd] v=%f sigma=%f\n", level, i, out, s_cur[i]);

		errorAbsSum_ += abs(err) * count;
		errorSqSum_ += err * err * count;
		errorEffectiveSqSum_ += err * err * gradientScale;
	}

	cases_ += count;
	effectiveCases_ += gradientScale;

	errorCases_ += prev_activ.size() * count;
	errorEffectiveCases_ += prev_activ.size() * gradientScale;

	// Level 1 has a special computation of derivatives (which are always 1,
	// because level 0 is the inputs that don't pass through any activation
	// function) but is otherwise the same as the others.
	for (level = lastLevel(); level > 0; level--) {
		s_cur.swap(s_next); // Next level's s_cur becomes s_next.

		ValueSubVector inactiv = trainDataInactivated(traindata, level);
		// Previous level's results in the training data, AKA inputs of this level.
		prev_activ = trainDataActivated(traindata, level - 1);

		// Only the CORNER activation works after the last level.
		if (level < lastLevel() || activation_ == CORNER) {
			switch (activation_) {
			case RELU:
				computeReluDerivatives(level, inactiv, der);
				{
					UsageSubVector lusage = usageOfLevel(level);
					for (size_t i = 0; i < inactiv.size(); i++) {
						if (inactiv[i] >= 0.)
							lusage[i]++;
					}
				}
				break;
			case LEAKY_RELU:
				computeLeakyReluDerivatives(level, inactiv, der, options_.reluLeakiness_);
				break;
			case CORNER:
				backpropCorners(level, inactiv, der, s_next, rate, debug);
				break;
			default:
				err.f("Invalid activation function %d.", activation_);
				return err;
			}
		} else {
			// Otherwise make the derivative an identity.
			der.assign(inactiv.size(), 1.);
		}

		FloatNeuralNet::backpropLevel(level, s_cur, s_next, der, prev_activ, inactiv, rate, debug);

		if (options_.autoRate_ && level == lastLevel()) {
			// Collect the information about inputs to last layer.
			for (size_t i = 0; i < prev_activ.size(); i++)  {
				cp_linputs_[i] += prev_activ[i];
				cp_linputs_grads_[i] += s_cur[i];
			}
		}
	}

	return err;
}

Erref
FloatNeuralNet::applyGradient(Value rate, bool *undone)
{
	Erref err;
	if (errors_.hasError()) {
		err.fAppend(errors_, "NN failed to initialize:");
		return err;
	}
	if (totalGradient_.size() != weightsVectorSize()) {
		if (options_.totalGradient_)
			err.f("Must complete a training pass before applying the gradient.");
		else
			err.f("Total gradient computation is disabled in the options.");
		return err;
	}

	if (undone != nullptr) {
		*undone = false;
	}

	if (options_.autoRate_) {
		if (options_.trainingRate_ <= 0) {
			// Do an initial guess.
			// TODO: if dividing by cases_ later, don't do it here.
			options_.trainingRate_ = 0.001 / cases_;
		}

		// Don't drive the rate lower than this.
		// TODO: if dividing by cases_ later, don't do it here.
		double min_rate = 0.0001 / cases_;

		if (autoRateRedo_) {
			// autoRateRedo_ means that the rate was computed on the previous pass,
			// and this pass is a redo of pre-previous pass with a new rate, so
			// just apply it.
			autoRateRedo_ = false;
			autoRateAfterRedo_ = true;
			printf("DEBUG this is a redo of the last pass\n");
		} else {
			if (!lp_lweights_grads_.empty() && !autoRateAfterRedo_) {
				// If we have done a last pass and saved values from it, we can compute 1/L.

				double vsum = 0.; // sum of squared weight values differences
				double gsum = 0.; // sum of squared weight gradients differences

				size_t level = lastLevel();
				size_t prevsz = levels_[level-1];

				// last layer inputs & their gradients
				for (size_t i = 0; i < prevsz; i++) {
					double v = cp_linputs_[i] - lp_linputs_[i];
					vsum += v * v;
					double g = cp_linputs_grads_[i] - lp_linputs_grads_[i];
					gsum += g * g;
				}

				// last layer weights & their gradients
				ValueSubVector cp_lwt = weightsOfLevel(level).asVector();
				ValueSubVector cp_lgr = gradientOfLevel(level).asVector();
				ValueSubVector lp_lwt = lpWeightsOfLevel(level).asVector();
				// Gradients for last pass last layer are in lp_lweights_grads_.
				for (size_t i = 0; i < cp_lwt.size(); i++) {
					double v = cp_lwt[i] - lp_lwt[i];
					vsum += v * v;
					double g = cp_lgr[i] - lp_lweights_grads_[i];
					gsum += g * g;
				}

				// last layer corner weights & their gradients
				if (activation_ == CORNER) {
					ValueSubVector cp_lcor_wt = cornersOfLevel(level).asVector();
					ValueSubVector cp_lcor_gr = cornersGradientOfLevel(level).asVector();
					// Last pass weitght are in lp_corners_, gradients in lp_lcorners_grads_.
					for (size_t i = 0; i < cp_lcor_wt.size(); i++) {
						double v = cp_lcor_wt[i] - lp_corners_[i];
						vsum += v * v;
						double g = cp_lcor_gr[i] - lp_lcorners_grads_[i];
						gsum += g * g;
					}
				}

				if (gsum == 0.) {
					printf("DEBUG gradient difference is 0? Leaving the rate unchanged, dx=%g\n", sqrt(vsum));
				} else if (vsum == 0.) {
					printf("DEBUG 1/L is 0? Leaving the rate unchanged, dgrad = %g\n", sqrt(gsum));
				} else {
					// 1/L represents the maximum step, and to be safe, keep it a few times lower
					// by multiplying to trainingRateScale_.
					double invL = sqrt(vsum / gsum) * options_.trainingRateScale_;
					printf("DEBUG 1/L = %g, old rate = %g, dx = %g, dgrad = %g\n", invL, options_.trainingRate_, sqrt(vsum), sqrt(gsum));

					// To reduce the number of redoings, keep the watermarks that
					// grow the rate low.
					const double low_watermark = 0.1 * 2./3.;
					const double high_watermark = 0.1 * 5./6.;
					const double very_high_watermark = 15./16.;
					const double max_growth_rate = 2.;

					if (options_.trainingRate_ < invL * low_watermark) {
						// Gradually grow the rate, leaving a safety margin.
						options_.trainingRate_ *= max_growth_rate;
						if (options_.trainingRate_ > invL * low_watermark) {
							options_.trainingRate_ = invL * low_watermark;
						}
						printf("DEBUG bump the rate up to %g\n", options_.trainingRate_);
					} else if (options_.trainingRate_ > min_rate && options_.trainingRate_ > invL * high_watermark) {
						if (options_.trainingRate_ > invL * very_high_watermark) {
							// Revert the last step, retry it with the new smaller rate.
							autoRateRedo_ = true;
						}
						// Reduce to a safe rate.
						options_.trainingRate_ = invL * low_watermark;
						if (options_.trainingRate_ < min_rate)
							options_.trainingRate_ = min_rate;
						printf("DEBUG bump the rate down to %g\n", options_.trainingRate_);

						if (autoRateRedo_) {
							printf("DEBUG restored weights for redo\n");
							// And restore the weights, to redo with a smaller rate.
							// There is no need to restore the rest of values because
							// they're not affected by a redo, and the full set of gradients
							// will be recomputed on the redo pass.
							weights_ = lp_weights_;
							corners_ = lp_corners_;

							// Go back by 2 passes, since the current pass got never completed and
							// the previous pass got undone.
							trainingPass_ -= 2;
							*undone = true;
							// Don't mess up the restored weights, return and let the
							// redo pass deal with them.
							return err;
						}
					}
				}
			}
			autoRateAfterRedo_ = false;

			// Save current pass values to become last pass.
			size_t level = lastLevel();

			swap(cp_linputs_, lp_linputs_);
			swap(cp_linputs_grads_, lp_linputs_grads_);

			lp_weights_ = weights_;

			ValueSubVector cp_lgr = gradientOfLevel(level).asVector();
			lp_lweights_grads_.resize(cp_lgr.size());
			for (size_t i = 0; i < cp_lgr.size(); i++) {
				lp_lweights_grads_[i] = cp_lgr[i];
			}

			if (activation_ == CORNER) {
				lp_corners_ = corners_;

				ValueSubVector cp_lcor_gr = cornersGradientOfLevel(level).asVector();
				lp_lcorners_grads_.resize(cp_lcor_gr.size());
				for (size_t i = 0; i < cp_lcor_gr.size(); i++) {
					lp_lcorners_grads_[i] = cp_lcor_gr[i];
				}
			}
		}

		// TODO: maybe divide by cases_ ?
		rate = options_.trainingRate_;
		printf("DEBUG effective rate = %g\n", rate);
	} else {
		if (cases_ != effectiveCases_) {
			// Compute the size by which the gradients got moved relatively
			// to by how much would they be moved without enhancing the
			// incorrect cases.
			Value newrate = rate * cases_ / effectiveCases_;
			printf("DEBUG effective rate = %g (adjusted by effective cases from %g)\n", newrate, rate);
			rate = newrate;
		}

		if (options_.autoRate2_
		&& lastTotalGradient_.size() == weights_.size()
		&& totalGradient_.size() == weights_.size()) {
			// Compute the stats of where the gradient has changed sign.

			if (true) {
				// SBXXX This is a starting point of computing gradients by level,
				// so far purely informational, hacked from the overall computation.

				// Level 0 represenst inputs, it has no gradients.
				for (size_t level = 1; level < levels_.size(); level++) {
					double sumChgBefore = 0.;
					double sumChgAfter = 0.;
					double sumUnchgBefore = 0.;
					double sumUnchgAfter = 0.;
					size_t nchg = 0;
					size_t nzero = 0;
					size_t nunchg = 0;

					ValueSubVector grads = gradientOfLevel(level).asVector();
					ValueSubVector lastgrads = ValueSubVector(lastTotalGradient_,
						level_weights_idx_[level], (levels_[level-1] + 1) * levels_[level]);
					const size_t ndim = grads.size();

					for (size_t i = 0; i < ndim; i++) {
						Value vb = lastgrads[i];
						Value va = grads[i];
						if (vb * va < 0) {
							++nchg;
							sumChgBefore += vb * vb;
							sumChgAfter += va * va;
						} else if (vb == 0.) {
							++nzero;
						} else {
							++nunchg;
							sumUnchgBefore += vb * vb;
							sumUnchgAfter += va * va;
						}
					}

					if (activation_ == CORNER) {
						// Include gradients from the activation function weights.
						ValueSubVector grads = cornersGradientOfLevel(level).asVector();
						ValueSubVector lastgrads = ValueSubVector(lastTotalCornersGradient_,
							level_corners_idx_[level], levels_[level] * CW_SIZE);
						const size_t ndim = grads.size();

						for (size_t i = 0; i < ndim; i++) {
							Value vb = lastgrads[i];
							Value va = grads[i];
							if (vb * va < 0) {
								++nchg;
								sumChgBefore += vb * vb;
								sumChgAfter += va * va;
							} else if (vb == 0.) {
								++nzero;
							} else {
								++nunchg;
								sumUnchgBefore += vb * vb;
								sumUnchgAfter += va * va;
							}
						}
					}
					
					double xchg = (nchg == 0? 1 : nchg);
					double xunchg = (nunchg == 0? 1 : nunchg);

					// mean squared gradients;
					// note that the "after" items haven't been adjusted for the
					// saturation yet
					double msqChgBefore = 
						sqrt(sumChgBefore / xchg);
					double msqChgAfter = 
						sqrt(sumChgAfter / xchg);
					double msqUnchgBefore = 
						sqrt(sumUnchgBefore / xunchg);
					double msqUnchgAfter = 
						sqrt(sumUnchgAfter / xunchg);
					printf("DEBUG    L%zd gradient changed sign in %zd of %zd, zero %zd; msq chg %g/%g unchg %g/%g\n",
						level, nchg, nchg + nunchg, nzero,
						msqChgBefore, msqChgAfter, msqUnchgBefore, msqUnchgAfter
						);
				}

			}

			double sumChgBefore = 0.;
			double sumChgAfter = 0.;
			double sumUnchgBefore = 0.;
			double sumUnchgAfter = 0.;
			size_t nchg = 0;
			size_t nzero = 0;

			{
				size_t end = totalGradient_.size();
				for (size_t i = 0; i < end; i++) {
					Value vb = lastTotalGradient_[i];
					Value va = totalGradient_[i];
					if (vb * va < 0) {
						++nchg;
						sumChgBefore += vb * vb;
						sumChgAfter += va * va;
					} else if (vb == 0.) {
						++nzero;
					} else {
						sumUnchgBefore += vb * vb;
						sumUnchgAfter += va * va;
					}
				}
			}
			
			if (activation_ == CORNER) {
				size_t end = totalCornersGradient_.size();
				for (size_t i = 0; i < end; i++) {
					Value vb = lastTotalCornersGradient_[i];
					Value va = totalCornersGradient_[i];
					if (vb * va < 0) {
						++nchg;
						sumChgBefore += vb * vb;
						sumChgAfter += va * va;
					} else if (vb == 0.) {
						++nzero;
					} else {
						sumUnchgBefore += vb * vb;
						sumUnchgAfter += va * va;
					}
				}
			}

			double xchg = (nchg == 0? 1 : nchg);
			double xunchg = totalGradient_.size() + totalCornersGradient_.size() - nchg - nzero;
			if (xunchg == 0.)
				xunchg = 1.;

			// mean squared gradients;
			// note that the "after" items haven't been adjusted for the
			// saturation yet
			double msqChgBefore = 
				sqrt(sumChgBefore / xchg);
			double msqChgAfter = 
				sqrt(sumChgAfter / xchg);
			double msqUnchgBefore = 
				sqrt(sumUnchgBefore / xunchg);
			double msqUnchgAfter = 
				sqrt(sumUnchgAfter / xunchg);
			printf("DEBUG gradient changed sign in %zd of %zd, zero %zd; msq chg %g/%g unchg %g/%g\n",
				nchg, totalGradient_.size() + totalCornersGradient_.size() - nzero, nzero,
				msqChgBefore, msqChgAfter, msqUnchgBefore, msqUnchgAfter
				);

			// Alternatives:
			// Implementation (1)
			// if (msqChgAfter > msqUnchgAfter * 1e-1 && msqChgAfter > 0.9 * msqChgBefore
			// && xchg > 0.05 * (xchg + xunchg))
			// Implementation (2)
			// if (msqChgAfter > msqChgBefore
			// && (xchg > 0.30 * (xchg + xunchg) || sumChgAfter > 0.5 * sumUnchgBefore))
			// Implementation (3)
			if (msqChgAfter > msqChgBefore
			&& msqChgAfter * xchg > 2. *  msqUnchgBefore * xunchg) 
			{
				if (msqChgAfter > 2. * msqChgBefore || xchg > 0.7 * (xchg + xunchg)) {
					// TODO: also redo the last pass if the ratio is too high?
					autoRate2Adj_ *= 0.2;
				} else {
					autoRate2Adj_ *= 0.67;
				}
				printf("DEBUG bumped rate adj Down to %g\n", autoRate2Adj_);
			} else if (msqChgAfter > msqChgBefore && msqChgAfter > msqUnchgBefore * 3) {
				autoRate2Adj_ *= 0.8;
				printf("DEBUG bumped rate adj down to %g\n", autoRate2Adj_);
			} else {
				autoRate2Adj_ *= 1.2;
				printf("DEBUG bumped rate adj   up to %g\n", autoRate2Adj_);
			}

			Value newrate = rate * autoRate2Adj_;
			printf("DEBUG effective rate = %g (adjusted by autoRate2 from %g)\n", newrate, rate);
			rate = newrate;
		} else {
			printf("DEBUG effective rate = %g\n", rate);
		}
	}

	const bool momentum = options_.momentum_
		&& mdiff_weights_.size() == weights_.size();

	if (options_.scaleRatePerLayer_) {
		// Update the weights, scaling the rate per the number of
		// outputs in each level.
		for (size_t level = 1; level <= lastLevel(); level++) {
			// The effecive rate for this level.
			Value lrate = rate;
			// Adjust per number of outputs of each neuron.
			if (level != lastLevel()) {
				lrate /= levels_[level + 1];
			}
				
			{
				size_t mw_offset = level_weights_idx_[level]; // same as in weightsOfLevel()
				ValueSubVector w = weightsOfLevel(level).asVector();
				ValueSubVector g = gradientOfLevel(level).asVector();
				const size_t end = w.size();
				
				for (size_t i = 0; i < end; i++) {
					Value wt = w[i];
					Value newwt = wt - lrate * g[i];
					if (saturate(newwt, wt)) {
						// Gradient is trying to push beyond the allowed area, ignore it
						g[i] = 0.; // might interact with options_.enableWeightFloor_
					}
					if (momentum)
						mdiff_weights_[mw_offset + i] += newwt - wt;
					w[i] = newwt;
				}
			}

			if (activation_ == CORNER) {
				size_t mw_offset = level_corners_idx_[level]; // same as in cornersOfLevel()
				ValueSubVector w = cornersOfLevel(level).asVector();
				ValueSubVector g = cornersGradientOfLevel(level).asVector();

				const size_t end = w.size();
				
				for (size_t i = 0; i < end; i++) {
					Value wt = w[i];
					Value newwt = wt - lrate * g[i];
					if (saturate(newwt, wt)) {
						// Gradient is trying to push beyond the allowed area, ignore it
						g[i] = 0.; // might interact with options_.enableWeightFloor_
					}
					if (momentum)
						mdiff_corners_[mw_offset + i] += newwt - wt;
					w[i] = newwt;
				}
			}
		}
	} else {
		// For a sort-of-randomization of weights (so that if we start with the
		// same weights, they won't stay the same), tweak the gradients by a small
		// amount.
		size_t tweakPeriod = levels_[0] * 3;
		Value tweakStep = options_.tweakRate_ / tweakPeriod;

		// Update the weights.
		{
			size_t tweaker = trainingPass_; // start from a different point on each pass
			size_t end = totalGradient_.size();
			for (size_t i = 0; i < end; i++) {
				Value wt = weights_[i];
				Value g = totalGradient_[i];
				Value newwt = wt - rate * (g - g * tweaker * tweakStep);
				if (saturate(newwt, wt)) {
					// Gradient is trying to push beyond the allowed area, ignore it
					totalGradient_[i] = 0.; // might interact with options_.enableWeightFloor_
				}
				if (momentum)
					mdiff_weights_[i] += newwt - wt;
				weights_[i] = newwt;
				tweaker = (tweaker + 1) % tweakPeriod;
			}
		}

		if (activation_ == CORNER) {
			size_t tweaker = trainingPass_; // start from a different point on each pass
			size_t end = totalCornersGradient_.size();
			for (size_t i = 0; i < end; i++) {
				Value wt = corners_[i];
				Value g = totalCornersGradient_[i];
				Value newwt = wt - rate * (g - g * tweaker * tweakStep);
				if (saturate(newwt, wt)) {
					// Gradient is trying to push beyond the allowed area, ignore it
					totalCornersGradient_[i] = 0.; // might interact with options_.enableWeightFloor_
				}
				if (momentum)
					mdiff_corners_[i] += newwt - wt;
				corners_[i] = newwt;
				tweaker = (tweaker + 1) % tweakPeriod;
			}
		}
	}

	if (momentum && options_.momentumFastBrake_
	&& lastTotalGradient_.size() == weights_.size()
	&& totalGradient_.size() == weights_.size()) {
		// Kill the momentum for the dimensions that have swapped the
		// sign of gradient. This drastically reduces the "circling of
		// the drain" behavior of the momentum methods.

		size_t nchg = 0;

		{
			size_t end = totalGradient_.size();
			for (size_t i = 0; i < end; i++) {
				Value vb = lastTotalGradient_[i];
				Value va = totalGradient_[i];
				if (vb * va < 0) {
					mdiff_weights_[i] = 0.;
				}
			}
		}
		
		if (activation_ == CORNER) {
			size_t end = totalCornersGradient_.size();
			for (size_t i = 0; i < end; i++) {
				Value vb = lastTotalCornersGradient_[i];
				Value va = totalCornersGradient_[i];
				if (vb * va < 0) {
					mdiff_corners_[i] = 0.;
				}
			}
		}

		// With autoRate2_ the message would be already logged.
		if (!options_.autoRate2_) {
			printf("DEBUG gradient changed sign in %zd of %zd\n",
				nchg, totalGradient_.size() + totalCornersGradient_.size());
		}
	}

	// Preserve the last gradient for comparisons.
	lastTotalGradient_.swap(totalGradient_);
	lastTotalCornersGradient_.swap(totalCornersGradient_);

	return err;
}

Erref
FloatNeuralNet::computeForTraining(const ValueVector &inputs, ValueVector &traindata)
{
	Erref err;
	if (errors_.hasError()) {
		err.fAppend(errors_, "NN failed to initialize:");
		return err;
	}

	if (inputs.size() != (size_t)levels_[0]) {
		err.f("Invalid input data size for level 0, must contain %d elements, contains %zd.", levels_[0], inputs.size());
		return err;
	}

	traindata.resize(trainDataSize());

	// The level 0 values go directly into activated entries.
	ValueSubVector activ = trainDataActivated(traindata, 0);
	// fprintf(stderr, "DEBUG level 0\n");
	for (size_t i = 0; i < activ.size(); i++) {
		activ[i] = inputs[i];
		// fprintf(stderr, "DEBUG   activ[0, %zd] = %f\n", i, activ[i]);
	}

	for (size_t level = 1; level < levels_.size(); level++) {
		// activated values of the previous level
		ValueSubVector old_activ = activ;
		activ = trainDataActivated(traindata, level);

		ValueSubVector inactiv = trainDataInactivated(traindata, level);

		// fprintf(stderr, "DEBUG level %zd, sz=%zd, oldsz=%zd\n", level, activ.size(), old_activ.size());

		// Reset all the outputs to 0.
		for (size_t cur = 0; cur < inactiv.size(); cur++) {
			inactiv[cur] = 0.;
		}
		
		// Each row of weights corresponds to one input from previous layer.
		ValueSubMatrix lweights = weightsOfLevel(level);

		for (size_t old = 0; old < old_activ.size(); old++) {
			Value oldval = old_activ[old];
			ValueSubVector subweights = lweights.rowAsVector(old);
			for (size_t cur = 0; cur < inactiv.size(); cur++) {
				inactiv[cur] += oldval * subweights[cur];
				// fprintf(stderr, "DEBUG   data[%zd, %zd] = %f <- += %f * %f\n", level, cur, inactiv[cur], oldval, subweights[cur]);
			}
		}

		// Add the bias weights.
		ValueSubVector subweights = lweights.rowAsVector(old_activ.size());
		for (size_t cur = 0; cur < inactiv.size(); cur++) {
			// The bias "input" is always 1.
			inactiv[cur] += subweights[cur];
			// fprintf(stderr, "DEBUG   data[%zd, %zd] = %f <- += %f\n", level, cur, inactiv[cur], subweights[cur]);
		}

		// Now do the activation.
		switch (activation_) {
		case RELU:
			activateRelu(level == lastLevel(), inactiv, activ);
			break;
		case LEAKY_RELU:
			activateLeakyRelu(level == lastLevel(), inactiv, activ, options_.reluLeakiness_);
			break;
		case CORNER:
			activateCorner(level, inactiv, activ);
			break;
		default:
			err.f("Invalid activation function %d.", activation_);
			return err;
		}
	}

	return err;
}

Erref
FloatNeuralNet::compute(const ValueVector &inputs, ValueVector &outputs, size_t &highest)
{
	Erref err;
	ValueVector traindata;

	// TODO: implement in a more efficient way, without keeping all the data
	// from all the levels.
	err = computeForTraining(inputs, traindata);
	if (err.hasError()) {
		return err;
	}

	size_t level = lastLevel();

	// For classic activation functions, the last layer just copies data
	// instead of activation.
	ValueSubVector result = trainDataActivated(traindata, level);

	Value highval = result[0];
	highest = 0;

	outputs.resize(result.size());
	for (size_t i = 0; i < result.size(); i++) {
		Value val = result[i];
		outputs[i] = val;
		if (val > highval) {
			highval = val;
			highest = i;
		}
	}

	return err;
}

Erref
FloatNeuralNet::reclaim(size_t max_unused, size_t &num_reclaimed)
{
	Erref err;
	if (errors_.hasError()) {
		err.fAppend(errors_, "NN failed to initialize:");
		return err;
	}

	// Reclaim gets called at the very end of the pass, after applyGradient(),
	// so (autoRateRedo_ == true) means that the previous pass is getting
	// redone and any reclaim stats computed in this pass are invalid.
	if (activation_ != RELU || autoRateRedo_) {
		num_reclaimed = 0;
		return err; // nothing to reclaim
	}

	size_t reclaim_count = 0;

	// The last level is excluded from reclaiming.
	for (size_t level = 1; level < lastLevel(); level++) {
		UsageSubVector lusage = usageOfLevel(level);

		ValueSubMatrix lweights = weightsOfLevel(level);

		const size_t prevsz = (size_t)levels_[level - 1];

		for (size_t i = 0; i < lusage.size(); i++) {
			// fprintf(stderr, "DEBUG       level %zd,  neuron %zd, use was %zd\n", level, i, (size_t)lusage[i]);
			if (lusage[i] > max_unused)
				continue;

			// fprintf(stderr, "DEBUG reclaim level %zd,  neuron %zd, use was %zd\n", level, i, (size_t)lusage[i]);

			++reclaim_count;

			for (size_t j = 0; j < prevsz; j++) {
				lweights.at(j, i) = -lweights.at(j, i);
				// fprintf(stderr, "DEBUG     weights[%zd][%zd] = %f\n", j, i, lweights.at(j, i));
			}

			// Reset the bias value.
			lweights.at(prevsz, i) = -lweights.at(prevsz, i);
			// fprintf(stderr, "DEBUG     weights[%zd][%zd] = %f\n", prevsz, i, lweights.at(prevsz, i));

			if (level != lastLevel()) {
				// Set the weights that connect this neuron to the next level to 0,
				// to not upset the current function. Later they will get pushed in the
				// optimal direction by the gradient.
				ValueSubMatrix nlweights = weightsOfLevel(level + 1);
				ValueSubVector subweights = nlweights.rowAsVector(i);

				const size_t nextsz = (size_t)levels_[level + 1];
				for (size_t j = 0; j < nextsz; j++) {
					subweights[j] = 0.;
					// fprintf(stderr, "DEBUG     weights[%zd][%zd] = %f\n", i, j, subweights[j]);
				}
			}
		}
	}
		
	if (reclaim_count > 0) {
		// Make the auto-rate restart slowly.
		options_.trainingRate_ = -1;
		lp_lweights_grads_.clear();
	}
	

	num_reclaimed = reclaim_count;
	return err;
}

Erref
FloatNeuralNet::simpleDump()
{
	Erref err(new Errors);
	if (errors_.hasError()) {
		err.fAppend(errors_, "NN failed to initialize:");
		return err;
	}

	bool do_usage = (activation_ == RELU);
	bool do_corners = (activation_ == CORNER);

	for (size_t level = 1; level < levels_.size(); level++) {
		const size_t levsz = levels_[level];
		const size_t prevsz = levels_[level - 1];

		UsageSubVector lusage = usageOfLevel(level);
		ValueSubMatrix lweights = weightsOfLevel(level);
		ValueSubMatrix lcorners = cornersOfLevel(level);

		err->appendMsg(false, strprintf("--- L%zd", level));

		for (size_t i = 0; i < levsz; i++) {
			string s;
			
			if (do_usage)
				s = strprintf("L%zd %zd [used %d]: (", level, i, lusage[i]);
			else
				s = strprintf("L%zd %zd : (", level, i);

			for (size_t j = 0; j < prevsz; j++) {
				s += strprintf(" + %zd:%f", j, lweights.at(j, i));
			}
			// The bias.
			s += strprintf(" + B:%f", lweights.at(prevsz, i));

			if (do_corners) {
				s += strprintf(" [C %f:+ %f:%f]",
					lcorners.at(i, CW_LEFT),
					lcorners.at(i, CW_OFFSET),
					lcorners.at(i, CW_RIGHT));
			}

			s.push_back(')');
			err->appendMsg(false, s);
		}
	}

	return err;
}

void
FloatNeuralNet::startTrainingPass()
{
	errorAbsSum_ = 0.;
	errorSqSum_ = 0.;
	errorEffectiveSqSum_ = 0.;
	cases_ = 0;
	effectiveCases_ = 0;
	errorCases_ = 0;
	errorEffectiveCases_ = 0;
	correctCases_ = 0;
	incorrectCases_ = 0;
	aboveCases_ = 0;
	notAboveCases_ = 0;

	if (options_.totalGradient_) {
		totalGradient_.assign(weightsVectorSize(), 0.);
		totalCornersGradient_.assign(cornersVectorSize(), 0.);
		if (options_.autoRate_) {
			cp_linputs_.assign(levels_[lastLevel() - 1], 0.);
			cp_linputs_grads_.assign(levels_[lastLevel() - 1], 0.);
		} else {
			cp_linputs_.clear();
			cp_linputs_grads_.clear();
		}
	} else {
		totalGradient_.clear();
		totalCornersGradient_.clear();
		cp_linputs_.clear();
		cp_linputs_grads_.clear();
	}

	memset(usage_.data(), 0, sizeof(usage_[0]) * usage_.size());

	// Apply the momentum
	if (options_.momentum_
	&& mdiff_weights_.size() == weights_.size()) {
		double nu = 1.;

		if (options_.momentumFixedNu_) {
			// Don't gradually slow down, because w'll be making A LOT of steps.
			// The killing of inertia in dimensions that switch sign acts as a brake
			// instead.
			if (trainingPass_ < 2) {
				// The classic FISTA algorithm keeps nu at 0 after the first step,
				// probably that with the initial point being random, the first descent
				// step tends to be a large step in a random direction towards the "groove"
				// that doesn't represent well the direction of the "groove" itself,
				// so it gets excluded from the momentum.
				nu = 0.;
			}
		} else {
			double t_next = (1. + sqrt(1. + t_ * t_ * 4.)) / 2.;
			// momentum coefficient
			// (it's really eta in the FISTA paper but I wrote it down as nu,
			// so now nu it is)
			nu = (t_ - 1.) / t_next;

			// The classic FISTA algorithm doesn't update t on step 0.  The reason is
			// probably that with the initial point being random, the first descent
			// step tends to be a large step in a random direction towards the "groove"
			// that doesn't represent well the direction of the "groove" itself,
			// so it gets excluded from the momentum.
			if (trainingPass_ != 0) {
				// TODO this doesn't work well with undo's, t doesn't return,
				// but perhaps it doesn't matter that much
				t_ = t_next;
			} else {
				t_ = 1.;
			}
		}

		// printf("DEBUG nu=%f\n", nu);

		// Apply the momentum.
		{
			size_t end = totalGradient_.size();
			for (size_t i = 0; i < end; i++) {
				Value wt = weights_[i];
				Value newwt = wt + nu * mdiff_weights_[i];
				if (saturate(newwt, wt))
					mdiff_weights_[i] = newwt - wt;
				weights_[i] = newwt;
			}
		}

		if (activation_ == CORNER) {
			size_t end = totalCornersGradient_.size();
			for (size_t i = 0; i < end; i++) {
				Value wt = corners_[i];
				Value newwt = wt + nu * mdiff_corners_[i];
				if (saturate(newwt, wt))
					mdiff_corners_[i] = newwt - wt;
				corners_[i] = newwt;
			}
		}
	}

}

void
FloatNeuralNet::endTrainingPass()
{
	if (totalGradient_.size() == weightsVectorSize()) {
		double allsum = 0.;
		total_gradient_norm2_.resize(levels_.size());

		// This re-weights the gradient to match the original number of
		// cases rather than the effective one.
		double effectiveScale = 1.;
		if (cases_ != 0) {
			effectiveScale  = (double) cases_ / effectiveCases_;
		}

		printf("DEBUG msq grad ");

		// Level 0 represenst inputs, it has no gradients.
		for (size_t level = 1; level < levels_.size(); level++) {
			size_t lcount = 0;
			ValueSubMatrix lgrads = gradientOfLevel(level);
			// Number of dimensions in gradient.
			const size_t ndim = level_weights_idx_[level + 1] - level_weights_idx_[level];

			double sum = 0.;
			for (size_t i = 0; i < ndim; i++) {
				double v = lgrads.data_[i];
				v = v * v;
				sum += v;
				allsum += v;
				lcount++;
			}

			if (activation_ == CORNER) {
				// Include gradients from the activation function weights.
				ValueSubMatrix lcorners = cornersGradientOfLevel(level);
				size_t limit = levels_[level] * CW_SIZE;
				for (size_t i = 0; i < limit; i++) {
					double v = lcorners.data_[i];
					v = v * v;
					sum += v;
					allsum += v;
					lcount++;
				}
			}
			
			total_gradient_norm2_[level] = sqrt(sum) * effectiveScale;

			printf("L%zd: %g/%g, ", level, sqrt(sum/lcount), sqrt(sum/lcount) * effectiveScale); // "DEBUG"
		}

		printf("\n"); // "DEBUG"

		toal_all_gradient_norm2_ = sqrt(allsum) * effectiveScale;
	}
	++trainingPass_;
}

double
FloatNeuralNet::getAbsError() const
{
	return errorAbsSum_ / errorCases_;
}

double
FloatNeuralNet::getSquaredError() const
{
	return sqrt(errorSqSum_ / errorCases_);
}

double
FloatNeuralNet::getEffectiveSquaredError() const
{
	return sqrt(errorEffectiveSqSum_ / errorEffectiveCases_);
}

Erref
FloatNeuralNet::getLevelGradientNorm2(size_t layer, double &result) const
{
	Erref err;

	if ((totalGradient_.size() != weightsVectorSize() && lastTotalGradient_.size() != weightsVectorSize())
	|| total_gradient_norm2_.size() != levels_.size()) {
		if (options_.totalGradient_)
			err.f("Must complete a training pass before getting the gradient.");
		else
			err.f("Total gradient computation is disabled in the options.");
		return err;
	}
	if (layer < 1 || layer >= total_gradient_norm2_.size()) {
		err.f("Invalid layer %zd, max layer is %zd.", layer, total_gradient_norm2_.size() - 1);
		return err;
	}
	result = total_gradient_norm2_[layer];
	return err;
}

Erref
FloatNeuralNet::getLastGradientNorm2(double &result) const
{
	return getLevelGradientNorm2(lastLevel(), result);
}

Erref
FloatNeuralNet::getAllGradientNorm2(double &result) const
{
	Erref err;

	if (totalGradient_.size() != weightsVectorSize() && lastTotalGradient_.size() != weightsVectorSize()) {
		if (options_.totalGradient_)
			err.f("Must complete a training pass before getting the gradient.");
		else
			err.f("Total gradient computation is disabled in the options.");
		return err;
	}
	result = toal_all_gradient_norm2_;
	return err;
}

void
FloatNeuralNet::reconcileOptions()
{
	if (options_.autoRate_ || options_.momentum_)
		options_.totalGradient_ = true;
}

Erref FloatNeuralNet::checkpoint(const std::string &fname)
{
	Erref err;
	std::string tmpname = fname + ".tmp";
	FILE *f = fopen(tmpname.c_str(), "w");
	if (f == NULL) {
		err.f("Failed to open the file '%s' for writing: %s.", tmpname.c_str(), strerror(errno));
		return err;
	}

	bool logfail = false; // Need to log a failure
	
	do {
		static const char magic[sizeof(uint32_t) + 1] = "FNNC";
		if (fwrite(&magic, sizeof(uint32_t), 1, f) != 1) {
			logfail = true;
			break;
		}

		static const uint32_t version = 1;
		if (fwrite(&version, sizeof(version), 1, f) != 1) {
			logfail = true;
			break;
		}

		if (fwrite(&trainingPass_, sizeof(trainingPass_), 1, f) != 1) {
			logfail = true;
			break;
		}

		if (fwrite(weights_.data(), sizeof(weights_[0]), weights_.size(), f) != weights_.size()) {
			logfail = true;
			break;
		}
		
		if (activation_ == CORNER) {
			if (fwrite(corners_.data(), sizeof(corners_[0]), corners_.size(), f) != corners_.size()) {
				logfail = true;
				break;
			}
		}
	} while(false);

	if (logfail) {
		err.f("Failed to write to the file '%s': %s.", tmpname.c_str(), strerror(errno));
	}

	if (fclose(f) < 0 && !logfail) {
		err.f("Failed to write to the file '%s': %s.", tmpname.c_str(), strerror(errno));
	}
	if (err.hasError())
		return err;

	if (rename(tmpname.c_str(), fname.c_str()) < 0) {
		err.f("Failed to move the file '%s' to '%s': %s.", tmpname.c_str(), fname.c_str(), strerror(errno));
	}

	return err;
}

Erref FloatNeuralNet::uncheckpoint(const std::string &fname)
{
	Erref err;
	FILE *f = fopen(fname.c_str(), "r");
	if (f == NULL) {
		err.f("Failed to open the file '%s' for reading: %s.", fname.c_str(), strerror(errno));
		return err;
	}

	do {
		static const char magic[sizeof(uint32_t) + 1] = "FNNC";
		char got_magic[sizeof(uint32_t) + 1];
		
		if (fread(&got_magic, sizeof(uint32_t), 1, f) != 1) {
			err.f("Failed to read magic from the file '%s': %s.",
				fname.c_str(), errno < 0? strerror(errno) : "File is too short");
			break;
		}
		if (memcmp(magic, got_magic, sizeof(int32_t)) != 0) {
			err.f("Wrong magic cookie in file '%s'.", fname.c_str());
			break;
		}

		uint32_t version;
		if (fread(&version, sizeof(version), 1, f) != 1) {
			err.f("Failed to read version from the file '%s': %s.",
				fname.c_str(), errno < 0? strerror(errno) : "File is too short");
			break;
		}
		if (version != 1) {
			err.f("Unsupported version %d in file '%s'.", version, fname.c_str());
			break;
		}

		if (fread(&trainingPass_, sizeof(trainingPass_), 1, f) != 1) {
			err.f("Failed to read training pass number from the file '%s': %s.",
				fname.c_str(), errno < 0? strerror(errno) : "File is too short");
			break;
		}

		if (fread(weights_.data(), sizeof(weights_[0]), weights_.size(), f) != weights_.size()) {
			err.f("Failed to read weights from the file '%s': %s.",
				fname.c_str(), errno < 0? strerror(errno) : "File is too short");
			break;
		}
		
		if (activation_ == CORNER) {
			if (fread(corners_.data(), sizeof(corners_[0]), corners_.size(), f) != corners_.size()) {
				err.f("Failed to read corner weights from the file '%s': %s.",
					fname.c_str(), errno < 0? strerror(errno) : "File is too short");
				break;
			}
		}
	} while(false);

	if (fclose(f) < 0) {
		err.f("Failed to close the file '%s': %s.", fname.c_str(), strerror(errno));
	}

	if (err.hasError())
		return err;

	return err;
}

}; // TRICEPS_NS
