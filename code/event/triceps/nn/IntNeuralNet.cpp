//
// (C) Copyright 2022 Sergey A. Babkin.
// This file is a part of Triceps.
// See the file COPYRIGHT for the copyright notice and license information
//
//
// A simple neural network implementation with integer arithmetic.
// INCOMPLETE YET.

#include <nn/IntNeuralNet.h>
#include <inttypes.h>

namespace TRICEPS_NS {

/////////////////////////////////// IntNeuralNet ////////////////////////////////////////////

IntNeuralNet::IntNeuralNet(const LevelSizeVector &levels, int bytes_per_entry, ActivationFunction activation) :
	levels_(levels), bpe_(bytes_per_entry), level_idx_(levels.size() + 1)
{
	if (levels.size() < 2) {
		errors_.f("The level count %zd is invalid, minimum is 2", levels.size());
		return;
	}
	if (bpe_ != 2) {
		errors_.f("The entry size of %d bytes is not supported, only support 2.", bpe_);
		return;
	}

	if (levels[0] <= 0) {
		errors_.f("Invalid size %" PRId32 " of level 0.", levels[0]);
		return;
	}
	level_idx_[0] = 0;
	level_idx_[1] = 0; // level 0 contains no neurons.
	for (size_t i = 1; i < levels.size(); i++) {
		if (levels[i] <= 0) {
			errors_.f("Invalid size %" PRId32 " of level %zd.", levels[0], i);
			return;
		}
		
		// Each neuron has a coefficient for each neuron of the previous level.
		LevelSize sz = (size_t)levels[i] * levels[i - 1];
		level_idx_[i + 1] = level_idx_[i] + sz;
		if  (sz < 0 || level_idx_[i + 1] < level_idx_[i]) {
			errors_.f("The requested size of coefficent table is so large that it overflowed.");
			return;
		}
	}

	try {
		coeff1_.resize(level_idx_.back(), 0);
	} catch (Exception &e) {
		errors_.f("The requested size of coefficent table %zd doesn't fit into memory", level_idx_.back());
		return;
	}
}

void IntNeuralNet::randomize()
{
}

Erref IntNeuralNet::setLevel(size_t level, const ValueVector &coeff)
{
	Erref err;
	err.f("Not implemented yet");
	return err;
}

Erref IntNeuralNet::getLevel(size_t level, ValueVector &coeff) const
{
	Erref err;
	err.f("Not implemented yet");
	return err;
}

Erref IntNeuralNet::train(const ValueVector &inputs, const ValueVector &outputs)
{
	Erref err;
	err.f("Not implemented yet");
	return err;
}

Erref IntNeuralNet::compute(bool activate, const ValueVector &inputs,
	ValueVector &outputs, size_t &highest)
{
	Erref err;
	err.f("Not implemented yet");
	return err;
}

}; // TRICEPS_NS
