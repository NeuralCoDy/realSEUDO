//
// (C) Copyright 2022 Sergey A. Babkin.
// This file is a part of Triceps.
// See the file COPYRIGHT for the copyright notice and license information
//
//
// A simple neural network implementation with integer arithmetic.
// INCOMPLETE YET.

#ifndef __Triceps_IntNeuralNet_h__
#define __Triceps_IntNeuralNet_h__

#include <mem/Mtarget.h>
#include <common/Common.h>
#include <vector>

namespace TRICEPS_NS {

// A basic neural network logic. It doesn't know about the rows or anything,
// it just deals with values. It doesn't do any smart accelerations, just
// a basic computation.
//
// It's really a single-threaded object, requiring external synchronization.
class IntNeuralNet : public Mtarget
{
public:
	// Vector containint the size of each level, 1 element per level.
	// Signed int on purpose, to catch the negative numbers that might otherwise
	// be unwittingly converted to very large unsigned numbers.
	typedef int32_t LevelSize;
	typedef vector<LevelSize> LevelSizeVector;

	// The supported activation function types for neurons.
	typedef enum {
		RELU,
	} ActivationFunction;

	// Values used to represent the inputs and outputs of the neural network.
	// They are an integer representation of floating-point values in
	// range [-1, 1]. VALUE_1 represents 1, and -VALUE_1 represents -1.
	// (Note that internally a shorter representation can be used).
	typedef uint16_t Value;
	typedef vector<Value> ValueVector;

	// There are 2 basic ways to represent the maximum:
	// One is to use INT16_MAX. The trouble though is that 1 is then
	// represented as 0x7FFF, and when we multiply two of them, we
	// can't just shift the result of multiplication, or it would be
	// deteriorating little by little, we would have to multiply it
	// by 0x10002 to adjust it before shifting.
	// Another is to use 0x4000. This loses 1 bit of value but then the
	// result of multiplication can be simply shifted.
	static constexpr Value VALUE_1 = 0x4000;

	static Value doubleToValue(double val) {
		return (Value)(val * INT16_MAX);
	}

	static double valueToDouble(Value val) {
		return (double)val / (double)INT16_MAX;
	}

	// Constructs with coefficients initialized to 0.
	// Don't forget to use randomize() or setLevel() for a complete initialization.
	//
	// The level 0 is just inputs as they come in, there are no neurons at level 0,
	// so a minimum NN contains 2 levels. The network is fully connected, each neuron
	// is connected to all the data from the previous level.
	//
	// @param levels - sizes of each level (0 is the input level), must have at least
	//   2 levels. Be careful with large sizes, they consume a quadratic amount
	//   of memory.
	// @param bytes_per_entry - size of the integer coefficients in bytes,
	//   1 byte should be sufficient for execution, 2 for training. So far only 2
	//   is supported.
	// @param activation - type of activation function used for all neurons.
	IntNeuralNet(const LevelSizeVector &levels, int bytes_per_entry, ActivationFunction activation);

	// Get back the initialization values.
	const LevelSizeVector &getLevels() const
	{
		return levels_;
	}
	int getBytesPerEntry() const
	{
		return bpe_;
	}
	ActivationFunction getActivation() const
	{
		return RELU;
	}

	// Get the errors from construction. If there is an error, the object is unusable.
	Erref getErrors() const
	{
		return errors_;
	}

	// Set random values into the coefficients. It uses the system random number
	// generator. To use some other generator, use storeLevel().
	void randomize();

	// Store the coefficients for one level into the NN state.
	//
	// @param level - NN level to store to (must be valid).
	// @param coeff - coefficients to store, the vector size must match the size of the level.
	Erref setLevel(size_t level, const ValueVector &coeff);

	// Get the coefficients for one level into the NN state.
	//
	// @param level - NN level to get (must be valid).
	// @param coeff - for returning coefficients, will be resized as needed.
	Erref getLevel(size_t level, ValueVector &coeff) const;

	// Train by backpropagation.
	//
	// @param inputs - the inputs for a training example. The size of the vector
	//   must match the size of the 0th level of the network.
	// @param outputs - the ground truth of the outputs. The size of the vector
	//   must match the size of the last level of the network.
	Erref train(const ValueVector &inputs, const ValueVector &outputs);

	// Compute the result for one set of inputs.
	//
	// @param activate - if true, the returned values will be passed through
	//   the activation function, if false they will be the raw linear regression
	//   values of the top level.
	// @param inputs - the inputs for the computation. The size of the vector
	//   must match the size of the 0th level of the network.
	// @param outputs - the produced outputs. The size of the vector
	//   returned will match the size of the last level of the network,
	//   auto-resized if necessary. The activate argument determines if the
	//   values returned will be passed through the activation function.
	// @param highest - place to return the index of the highest output
	//   (which will essentially determine the classification).
	Erref compute(bool activate, const ValueVector &inputs,
		ValueVector &outputs, size_t &highest);

protected:
	// Errors from construction. If contains an error, the object is unusable.
	Erref errors_;
	// Sizes of the levels.
	LevelSizeVector levels_;
	// Bytes per entry of the network.
	int bpe_;
	// Indexes of the start points of the levels in the storage vector. The
	// level 0 is empty, since it contains no neurons, and the reat are
	// essentially cumulative sums of values from the levels vector multiplied by the
	// number of inputs (so the size ends up quadratic). Ather the last level,
	// it contain 1 more value that points past the end of the last level.
	std::vector<size_t> level_idx_;
	// Storage of the 2-byte coefficients.
	std::vector<int16_t> coeff1_;

private:
	// Those might be worth implementing later.
	IntNeuralNet(const IntNeuralNet &t);
	void operator=(const IntNeuralNet &t);
};

}; // TRICEPS_NS

#endif // __Triceps_IntNeuralNet_h__
