//
// (C) Copyright 2022 Sergey A. Babkin.
// This file is a part of Triceps.
// See the file COPYRIGHT for the copyright notice and license information
//
//
// A simple neural network implementation with float arithmetic.

#ifndef __Triceps_FloatNeuralNet_h__
#define __Triceps_FloatNeuralNet_h__

#include <mem/Mtarget.h>
#include <common/Common.h>
#include <string>
#include <vector>
#include <stdio.h>

namespace TRICEPS_NS {

// A basic neural network logic. It doesn't know about the rows or anything,
// it just deals with values. It doesn't do any smart accelerations, just
// a basic computation. This is the simplest version to implement.
//
// It's really a single-threaded object, requiring external synchronization.
class FloatNeuralNet : public Mtarget
{
public:
	// Vector containing the size of each level, 1 element per level.
	// Signed int on purpose, to catch the negative numbers that might otherwise
	// be unwittingly converted to very large unsigned numbers.
	typedef int32_t LevelSize;
	typedef vector<LevelSize> LevelSizeVector;

	// The supported activation function types for neurons.
	typedef enum {
		// Has gradient 0 on the left from 0, gradient 1 on the right.
		RELU,
		// Has gradient Options.reluLeakiness_ on the left from 0, gradient 1 on the right.
		LEAKY_RELU,
		// V-shaped, a corner of two straight lines joined at a point.
		// Not exactly an activation function in the traditional way.
		// It has its own set of weights determining the gradients of the
		// lines and position of the joining point that do get adjusted by
		// backpropagation, so it's more of a very specialized neuron.
		// SBXXX add link to description in blog, or even copy it to a README file
		CORNER,
	} ActivationFunction;

	// Values used to represent the inputs and outputs of the neural network.
	typedef float Value;
	typedef vector<Value> ValueVector;

	template <typename T>
	struct SubVector;

	// A 2-dimensional matrix stored inside a fgragment of a vector.
	template <typename T>
	struct SubMatrix
	{
		SubMatrix()
			: data_(nullptr), rowsize_(0)
		{ }

		// @param vector - vector that contains this submatrix, must be held
		//    by the caller for the duration of submatrix's life
		// @param offset - offset of submatrix's first element relative to the vector
		// @param rowcount - count of rows in submatrix
		// @param rowsize - size of row of the submatrix
		SubMatrix(vector<T> &vector, size_t offset, size_t rowcount, size_t rowsize)
			: data_(vector.data() + offset), rowcount_(rowcount), rowsize_(rowsize)
		{ }

		// @param data - the first element of data
		// @param offset - offset of subvector's first element relative to the vector
		// @param rowcount - count of rows in submatrix
		// @param rowsize - size of row of the submatrix
		SubMatrix(T *data, size_t offset, size_t rowcount, size_t rowsize)
			: data_(data + offset), rowcount_(rowcount), rowsize_(rowsize)
		{ }

		// Refer to a value (row, col), like a 2-dimensional array.
		T &at(size_t row, size_t col) const
		{
			return data_[row * rowsize_ + col];
		}

		// Get one row as a vector.
		SubVector<T> rowAsVector(size_t row) const;

		// Get the whole data set as one vector.
		SubVector<T> asVector() const;

		T *data_;
		size_t rowcount_;
		size_t rowsize_;
	};
	typedef SubMatrix<Value> ValueSubMatrix;

	// A nested vector stored inside a fgragment of a vector.
	template <typename T>
	struct SubVector
	{
		SubVector()
			: data_(nullptr), size_(0)
		{ }

		// @param vector - vector that contains this subvector, must be held
		//    by the caller for the duration of subvector's life
		// @param offset - offset of subvector's first element relative to the vector
		// @param sz - size of the vector
		SubVector(vector<T> &vector, size_t offset, size_t sz)
			: data_(vector.data() + offset), size_(sz)
		{ }

		SubVector(SubVector &subvector, size_t offset, size_t sz)
			: data_(subvector.data_ + offset), size_(sz)
		{ }

		SubVector(T *data, size_t offset, size_t sz)
			: data_(data + offset), size_(sz)
		{ }

		SubVector(SubMatrix<T> &submatrix, size_t row, size_t col, size_t sz)
			: data_(&submatrix.at(row, col)), size_(sz)
		{ }

		T &operator[](size_t idx) const
		{
			return data_[idx];
		}

		size_t size() const
		{
			return size_;
		}

		T *data_;
		size_t size_;
	};
	typedef SubVector<Value> ValueSubVector;

	struct Options {
		// Compute a cumulative total gradient on the whole training pass.
		bool totalGradient_ = true;

		// Automatically compute the training rate and apply it. Requires
		// totalGradient_=true and will force it automatically.
		// Also, setting scaleRatePerLayer_=true is recommended.
		// See the other related options below.
		bool autoRate_ = false;

		// The second attempt at automatic rate computation.
		bool autoRate2_ = false;

		// -- Options related to autoRate_.

		// Scale the training rate per each layer, dividing it by the number
		// of outputs of this layer. This should make the use of a higher
		// rate more safe.
		bool scaleRatePerLayer_ = false;

		// The auto-detected training rate will be multiplied by this number.
		// This is needed because the auto-detection is not very good at the
		// moment. From experience, should be 0.1 or lower. This value can
		// also be gradually lowered during training to achieve the more fine
		// detail.
		Value trainingRateScale_ = 1e-2;

		// The automatically computed rate will be kept here. This is the current
		// guess. A negative value means to take an automatic first guess.
		Value trainingRate_ = -1.;

		// --

		// Weights will be limited to this absolute value on top.
		// Going much beyond 1 may create a runaway growth of weights, so larger
		// values should be used with care, and prefreeably with autoRate_==true.
		Value weightSaturation_ = 1.;

		// Enables the handling of weightFloor_.
		bool enableWeightFloor_ = false;

		// Weights will be limited to this absolute value on the bottom.
		// Enabled by enableWeightFloor_. Whenever a value tries to cross
		// this boundary on the way towards 0, it will stop at it, and then if
		// it continues to move towards 0 on the next pass, will be
		// "teleported" to the other side from 0.
		Value weightFloor_ = 1e-4;

		// Used only with LEAKY_RELU. The gradient for its left side.
		Value reluLeakiness_ = 0.01;

		// Enables the (mono-)classifier training mode where getting the
		// right outcome show the highest value is more important than
		// hitting the specific output values. The gradients of the training
		// cases that show the incorrect outcome get their effective count
		// automatically grown. The effective training rate gets
		// auto-adjusted in applyGradient() for this growth, so no need to
		// mess with it manually. (Except that the auto-adjustment doesn't 
		// work currently with autoRate_ = true).
		bool isClassifier_ = false;

		// The maximum value by which the effective count can be higher than the
		// real count for the cases that got misclassified in the classifier
		// training mode (isClassifier_ = true). When the effective count
		// reaches this multiplier, it stops increasing.
		size_t maxMultiplier_ = 100;

		// In the classifier training mode (isClassifier_ = true), the effective
		// case counts start getting adjusted only after this many passes.
		// This allows things to settle roughly, so that the case counts
		// won't be messed up too much at random.
		size_t minPassEffectiveAdj_ = 50;

		// In the classifier training mode (isClassifier_ = true), the
		// effective count of the cases that are correct gets gradually
		// adjusted down until it reaches the real count.  This is the rate of
		// adjustment down relative to adjustment up.  It should be < 1 to
		// provide a hysteresis.
		Value effecttiveDecreaseRate_ = 0.3;

		// Enables the momentum descent, based on the FISTA algorithm from
		// https://people.rennes.inria.fr/Cedric.Herzet/Cedric.Herzet/Sparse_Seminar/Entrees/2012/11/12_A_Fast_Iterative_Shrinkage-Thresholding_Algorithmfor_Linear_Inverse_Problems_(A._Beck,_M._Teboulle)_files/Breck_2009.pdf
		// Works only with applyGradient(), a non-0 rate in train() would 
		// cause problems.
		// When both momentum_ = true and isClassifier_ = true, the effective
		// training rate auto-adjustment gets disabled because it tends to hit
		// too high.
		bool momentum_ = false;

		// If momentum descent is enabled, kill the momentum in the dimensions where
		// the gradient has changed sign.
		bool momentumFastBrake_ = true;

		// If the momentum descent is enabled, don't gradually reduce the
		// Fista nu coefficient (it's really eta in the FISTA paper but I wrote it down as nu,
		// so now nu it is). This works well when momentumFastBrake_=true.
		bool momentumFixedNu_ = true;

		// The gradients are slightly tweaked in a deterministic but pseudo-random-ish
		// fashion to make the identical neurons to diverge. This sets the
		// fraction of the maximum tweak to the gradient.
		Value tweakRate_ = 0.01;
	};

	// The fiexed bias that gets assigned during randomization. This may change
	// in the future, and a random bias might become used.
	// The whole idea of fixed bias and the value come from 
	// https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/
	static constexpr Value DEFAULT_BIAS = 0.1;

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
	// @param activation - type of activation function used for all neurons.
	// @param options - can be used to specify the options up front (some of them
	//   may get overridden if the activation function hardcodes things differently).
	FloatNeuralNet(const LevelSizeVector &levels, ActivationFunction activation, Options *options = nullptr);

	// Get back the initialization values.
	const LevelSizeVector &getLevels() const
	{
		return levels_;
	}
	// Only one type supported for now.
	ActivationFunction getActivation() const
	{
		return activation_;
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

	// Get the coefficients for one level from the NN state.
	//
	// @param level - NN level to get (must be valid).
	// @param coeff - for returning coefficients, will be resized as needed.
	Erref getLevel(size_t level, ValueVector &coeff) const;

	// SBXXX add methods to get and set CORNER coefficients by level

	// Mark the start of one pass of training data. Resets the stats to be
	// collected for the pass.
	void startTrainingPass();

	// Mark the end of one pass of training data. Summarizes the stats for the
	// pass and makes them readable.
	void endTrainingPass();

	// Train by backpropagation.
	//
	// The logic is based on
	// https://www.researchgate.net/publication/266396438_A_Gentle_Introduction_to_Backpropagation
	// and see also my (arguably better) explanation of reasons behind it in
	// https://babkin-cep.blogspot.com/2022/10/optimization-4-backpropagation-for-last.html
	// https://babkin-cep.blogspot.com/2022/10/optimization-5-backpropagation-further.html
	// https://babkin-cep.blogspot.com/2022/10/optimization-6-neural-networks.html
	//
	// @param inputs - the inputs for a training example. The size of the vector
	//   must match the size of the 0th level of the network.
	// @param outputs - the ground truth of the outputs. The size of the vector
	//   must match the size of the last level of the network.
	// @param rate - training rate, the fraction of gradient that gets updated
	//   in the weights. 0.01 should be a good value.
	// @param count - the number of times this record is duplicated in the training
	//   set. Together with effectiveCount, affects how many times will this record
	//   be counted in getPassSize() and how its gradient will be scaled.
	//   Must be > 0.
	// @param effectiveCount - if not NULL, gives the location of the effective
	//   value of count. In this case all the effects on getPassSize() and gradient
	//   scaling are determined by the value at effectiveCount, while count is used
	//   as a base for auto-adjusting the effectiveCount when
	//   Options.isClassifier_ == true. Value must be > 0.
	Erref train(const ValueVector &inputs, const ValueVector &outputs, Value rate,
		size_t count = 1, Value *effectiveCount = nullptr);

	// Train by applying the gradient computed form backpropagation. This allows
	// to take into account gradients from multiple training examples together,
	// instead of applying them separately. Option total_gradient must be enabled.
	// The usage pattern is:
	//
	// nn.startTrainingPass();
	// for (int i = 0; i < num_cases; i++) {
	//   nn.train(inputs[i], ouputs[i], 0.); // rate 0 only computes the gradient
	// }
	// nn.endTrainingPass();
	// nn.applyGradient(rate, &undone);
	//
	// The gradient is the sum of gradients from all cases, so the rate kind of
	// automatically scales with the number of cases, but they also partially
	// counteract each other, so a somewhat higher rate may be safe.
	//
	// It's also possible to use non-0 rate in both train() and applyGradient(),
	// splitting the rate between the two.
	//
	// @param rate - training rate, ignored if the option autoRate_ is used.
	// @param undone - if not NULL, the flag indicating the auto-rate's undo
	//    decision is returned there. If option autoRate_ == false, always returns
	//    false. If true is returned, for the correct accounting of passes, 2
	//    has to be subtracted from the count of done passes, since it means
	//    that the current pass got aborted, and the next will be a redo of the
	//    previous pass.
	Erref applyGradient(Value rate, bool *undone = nullptr);

	// Compute the result for one set of inputs. The top level has no
	// activation function after it (see 
	// https://babkin-cep.blogspot.com/2022/10/optimization-4-backpropagation-for-last.html
	// for a description of why), so the returned values represent
	// the raw linear regression values of the top level.
	//
	// @param inputs - the inputs for the computation. The size of the vector
	//   must match the size of the 0th level of the network.
	// @param outputs - the produced outputs. The size of the vector
	//   returned will match the size of the last level of the network,
	//   auto-resized if necessary. The activate argument determines if the
	//   values returned will be passed through the activation function.
	// @param highest - place to return the index of the highest output
	//   (which will essentially determine the classification).
	Erref compute(const ValueVector &inputs, ValueVector &outputs, size_t &highest);

	// Attempt to reclaim the neurons that have been initialized to
	// random values that never allow them to activate (i.e. to produce
	// a positive value that ReLU would pass through. If the whole set of
	// training data was presented and the neuron never activated, it means
	// that it's dead and cannot be trained. It can be reclaimed by
	// re-randomizing it or by flipping the signs of all the weights,
	// and then continuing the training.
	//
	// The use counts get reset in startTrainingPass().
	//
	// BE CAREFUL NOT TO CALL THE RECLAIMING TWICE, or it will flip the signs back.
	//
	// @param max_unused - the highest number of activations with which the
	//   neuron is still considered unused/dead. 0 is the normal value, but
	//   if there is a desire to reclaim the neurons that are used very rarely,
	//   a higher value can be used (this risks destroying the detection of
	//   rare cases).
	// @param num_reclaimed - returned number of neurons that have been
	//   reclaimed to a new random value.
	Erref reclaim(size_t max_unused, size_t &num_reclaimed);

	// A simple dump of weights for debugging. Can get unwieldy for all but
	// the very small networks. The dump is returned in erref.
	Erref simpleDump();

	// Get the number of completed training passes.
	// This number gets advanced in endTrainingPass() and may go back in
	// appplyGradient() if the auto-rate logic decides to do an undo.
	int64_t getTrainingPass() const
	{
		return trainingPass_;
	}

	// Get the avreage absolute error for the pass. The errors are summed from
	// the start of each case (i.e. before applying training).
	// (Must call endTrainingPass() first to complete the pass).
	double getAbsError() const;

	// Get the avreage squared error for the pass, for the real cases. Note
	// that this returns the SQUARE ROOT of the averaged squared error value.
	// The errors are summed from the start of each case (i.e. before applying
	// training). To get the errors in a stable state, do a "training pass"
	// with rate=0.  (Must call endTrainingPass() first to complete the pass).
	double getSquaredError() const;

	// Just like getSquaredError() but scales the error per the effective cases.
	double getEffectiveSquaredError() const;

	// Get the number of training cases in the last pass, i.e. the sum of
	// of the count arguments from train().
	size_t getPassSize() const
	{
		return cases_;
	}

	// Get the number of effective training cases in the last pass, i.e. the sum
	// of the *effectiveCount arguments from train().
	Value getEffectivePassSize() const
	{
		return effectiveCases_;
	}

	// Get the number of correctly classified cases in the pass (produces a valid
	// value only with options_.isClassifier_ = true).
	size_t getPassCorrect() const
	{
		return correctCases_;
	}

	// Get the number of incorrectly classified cases in the pass (produces a valid
	// value only with options_.isClassifier_ = true).
	size_t getPassIncorrect() const
	{
		return incorrectCases_;
	}

	// Get the number of cases that are both correctly classified and produced the
	// classifier output above 0 in the pass (produces a valid
	// value only with options_.isClassifier_ = true).
	size_t getPassAbove() const
	{
		return aboveCases_;
	}

	// Get the number of cases that are either correctly classified or produced the
	// classifier output <= 0 in the pass (produces a valid
	// value only with options_.isClassifier_ = true).
	size_t getPassNotAbove() const
	{
		return notAboveCases_;
	}

	// Get the "norm 2" (i.e. the Euclidean vector length, computed as square
	// root of the sum of squares of all dimensions) of the layer's
	// gradient, summarized over the pass.  This serves as a measurement of how
	// close we are to the optimim.  The gradients are summed from the start of
	// each case (i.e. before applying training). To get the gradient in a
	// stable state, do a training pass with rate=0.
	// (Must call endTrainingPass() first to complete the pass).
	//
	// Returns an error if the gradient computation was disabled in the options,
	// or if level is invalid.
	//
	// @param level - level number, at least 1 (since level 0 represents inputs
	//   and has no gradient)
	// @param result - the resulting value
	Erref getLevelGradientNorm2(size_t level, double &result) const;

	// Same as getLevelGradientNorm2() for last level.
	Erref getLastGradientNorm2(double &result) const;

	// Get the "norm 2" (i.e. the Euclidean vector length, computed as square
	// root of the sum of squares of all dimensions) of the gradient across all
	// layers, summarized over the pass.  This serves as a measurement of how
	// close we are to the optimim.  The gradients are summed from the start of
	// each case (i.e. before applying training). To get the gradient in a
	// stable state, do a training pass with rate=0.
	// (Must call endTrainingPass() first to complete the pass).
	//
	// Returns an error if the gradient computation was disabled in the options,
	// or if level is invalid.
	Erref getAllGradientNorm2(double &result) const;

	// Get the options.
	Options getOptions() const
	{
		return options_;
	}

	// Set the options. Definitely safe to set right after creation,
	// some options may also be safely changed before calling startTrainingPass().
	// Some options may get overridden if the activation function hardcodes things
	// different
	void setOptions(const Options &opt)
	{
		options_ = opt;
		reconcileOptions();
	}

	// Write the checkpoint to a file.
	// @param fname - file name
	Erref checkpoint(const std::string &fname);

	// Read the checkpoint from a file.
	// @param fname - file name
	Erref uncheckpoint(const std::string &fname);

protected:
	ActivationFunction activation_;

	// Errors from construction. If contains an error, the object is unusable.
	Erref errors_;
	// Sizes of the levels.
	LevelSizeVector levels_;

	// Index of the last level.
	inline size_t lastLevel() const
	{
		return levels_.size() - 1;
	}

	// Options for computation.
	Options options_;

	// --------------

	// Indexes of the start points of the levels in the neuron values vector that
	// is used in training. The vector itself gets allocated for duration of training
	// on one example. It's essentially a cumulative sum of the sizes from the
	// levels vector.
	//
	// Each level gets 2 sets of values: the originally computed
	// sum of weights and the value after the activation function, so each level's
	// size is multiplied by 2 (for the level 0 the original and activated values
	// would be the same).
	//
	// Ather the last level, it contains one more element that points past the
	// end of the last level and gives the total size of the values vector.
	std::vector<size_t> level_train_idx_;

	// Get the size that needs to be allocated for the training data.
	// See the description of last element in level_train_idx_.
	size_t trainDataSize() const
	{
		return level_train_idx_.back();
	}

	// Get the subvector of training data containing the inactivated outputs
	// of the given level.
	ValueSubVector trainDataInactivated(ValueVector &traindata, size_t level) const
	{
		// See the description of level_train_idx_ above.
		return ValueSubVector(traindata, level_train_idx_[level], levels_[level]);
	}

	// Get the subvector of training data containing the activated outputs
	// of the given level.
	ValueSubVector trainDataActivated(ValueVector &traindata, size_t level) const
	{
		// See the description of level_train_idx_ above.
		size_t levsz = levels_[level];
		return ValueSubVector(traindata, level_train_idx_[level] + levsz, levsz);
	}

	// --------------

	// Indexes of the start points of the levels in the coefficients storage
	// vector. The level 0 is empty, since it contains no neurons, and the rest
	// are essentially cumulative sums of sizes from the levels vector plus 1
	// multiplied by the number of inputs (so the size ends up quadratic). The
	// "plus 1" part reserves space for the bias weight that can be seen as
	// always receiving the pseudo-input of the constant 1.
	//
	// Ather the last level, it contains one more element that points past the
	// end of the last level and gives the total size of weights_.
	std::vector<size_t> level_weights_idx_;

	// Size of the weights vector.
	size_t weightsVectorSize() const 
	{
		return level_weights_idx_.back();
	}

	// Storage of the weights.
	// 
	// Data for each level is ordered by input first
	// and by neuron next.  I.e. first go all weigths for the 0th neuron of
	// previous level to each of the neurons of this level, then for 1st neuron
	// of previous level, and so on.
	//
	// The very last input is a pseudo-input that is always considered to be 1,
	// so the wieght for it is the bias value that gets always added to the total.
	ValueVector weights_;

	// Get the weights matrix of a particular level. Since the weights are grouped
	// by neurons of this level, the matrix row size is equal to the number of neurons
	// in this level, and the number of rows is equal to the number of inputs plus 1
	// (also equal to the number of neurons in the previous level plus 1). The "plus 1"
	// part represents the bias weights.
	ValueSubMatrix weightsOfLevel(size_t level)
	{
		return ValueSubMatrix(weights_, level_weights_idx_[level], levels_[level-1] + 1, levels_[level]);
	}

	// --------------

	// Used only with the CORNER activation function.
	// Indexes of the start points of the levels in the storage vector for the
	// weights of CORNER activation function. Each level contains 3 values per
	// each neuron in the level (see enum CornerWeightIdx).
	// The level 0 is empty, since it contains no neurons, and the rest are
	// essentially cumulative sums of sizes.
	//
	// Ather the last level, it contains one more element that points past the
	// end of the last level and gives the total size of corners_.
	std::vector<size_t> level_corners_idx_;

	// Size of the corners vector.
	size_t cornersVectorSize() const 
	{
		return level_corners_idx_.back();
	}

	// Used only with the CORNER activation function.
	// Storage of the weights for the CORNER activation function.
	ValueVector corners_;

	// Indexes of weights for each row of corners matrix.
	enum CornerWeightIdx {
		// Multiplier of the left line, the output of the neuron gets
		// multiplied by this when it's to the left of the corner.
		CW_LEFT,
		// Offset of the corner, the output of the neuron has this value
		// added in either case.
		CW_OFFSET,
		// Multiplier of the right line, the output of the neuron gets
		// multiplied by this when it's to the right of the corner.
		CW_RIGHT,
		// Not an index but the number of real indexes. Always the last value.
		CW_SIZE,
	};

	// Get the CORNER weights matrix of a particular level. Each row contains 3 weights,
	// per a single neuron (see enum CornerWeightIdx).
	ValueSubMatrix cornersOfLevel(size_t level)
	{
		return ValueSubMatrix(corners_, level_corners_idx_[level], levels_[level], CW_SIZE);
	}

	// Gradient values for every weight in corners_, accumulated by summming the gradients from
	// every training case in a training pass. Same indexes and size as in corners_.
	// Gets populated only if options_.totalGradient_=true.
	ValueVector totalCornersGradient_;

	// The gradients from the last pass.
	// TODO: lp_lcorners_grads_ can be made a part of lastTotalCornersGradient_.
	ValueVector lastTotalCornersGradient_;

	// Get the gradient matrix of a particular level, with gradients structured in the
	// same way as corners_.
	ValueSubMatrix cornersGradientOfLevel(size_t level)
	{
		return ValueSubMatrix(totalCornersGradient_, level_corners_idx_[level], levels_[level], CW_SIZE);
	}

	// Apply the CORNER activation function.
	// @param level - level being activated
	// @param inactiv - the inactivated values
	// @param activ - place to put the activated values
	void activateCorner(size_t level, ValueSubVector inactiv, ValueSubVector activ);

	// Compute activated derivatives for the outpus of a level, for CORNER,
	// and update the CORNER weights.
	//
	// @param level - level whose output derivatives are being computed.
	// @param inactiv - inactivated outputs of the level.
	// @param der - place to put the computed derivatives, will be resized
	//   as needed.
	// @param sigma - the "sigma" derivative term, coming from the next layer
	// @param rate - the rate of gradient descent
	// @param debug - print the debuggign information
	void backpropCorners(size_t level, ValueSubVector inactiv,
		ValueVector &der, const ValueVector &sigma, Value rate, bool debug);

	// --------------
	// Usage stats, for neuron reclaiming with plain ReLU.

	// Indexes in the usage_ vector, very similar to level_train_idx_, only storing
	// one value per neuron.
	// Used only for RELU, for the rest filled with 0s.
	std::vector<size_t> level_usage_idx_;

	// Size of the usage vector.
	size_t usageVectorSize() const
	{
		return level_usage_idx_.back();
	}

	typedef uint32_t UsageCount;
	typedef vector<UsageCount> UsageVector;
	typedef SubVector<UsageCount> UsageSubVector;

	// Used only for RELU.
	// Count of how many times the output of the neuron was usable
	// in training, i.e. produced a non-0 value. Indexes for the start of each
	// level are stored in level_usage_idx_[level];
	UsageVector usage_;

	// Get the subvector of usage data containing the data of the given level.
	UsageSubVector usageOfLevel(size_t level)
	{
		// See the description of level_usage_idx_ above.
		return UsageSubVector(usage_, level_usage_idx_[level], levels_[level]);
	}

	// --------------

	// Gradient values for every weight, accumulated by summming the gradients from
	// every training case in a training pass. The indexes of each level are stored
	// in level_weights_idx_[level], just as for weights_.
	// Gets populated only if options_.totalGradient_=true.
	ValueVector totalGradient_;
	
	// The gradients from the last pass.
	// TODO: lp_lweights_grads_ can be made a part of lastTotalGradient_.
	ValueVector lastTotalGradient_;

	// Get the gradient matrix of a particular level, with gradients structured in the
	// same way as the weights. Since the weights and gradients are grouped
	// by neurons of this level, the matrix row size is equal to the number of neurons
	// in this level, and the number of rows is equal to the number of inputs plus 1
	// (also equal to the number of neurons in the previous level plus 1). The "plus 1"
	// part represents the bias weights.
	ValueSubMatrix gradientOfLevel(size_t level)
	{
		return ValueSubMatrix(totalGradient_, level_weights_idx_[level], levels_[level-1] + 1, levels_[level]);
	}

	// --------------
	// Momentum descent.

	// Difference of weights since last pass, representing the momentum.
	// Has the same structure as weights_.
	ValueVector mdiff_weights_;

	// Difference of corner weights since last pass, representing the momentum.
	// Has the same structure as corners_.
	ValueVector mdiff_corners_;

	// The coefficient that determines the scale of the inertia.
    double t_ = 1.;

	// --------------
	// Auto-computation of training rate.
	// These vectors always exist but gain non-0 size only if they are used.
	// 
	// The computation is done in a strange way, broken up between multiple calls:
	//
	// * Since only the last layer is computed as a honest optimization, only
	//   it participates in computation of the training rate. But its inputs are
	//   also included into computation as weights.
	// * Since the only way to compute a gradient is to perform a training pass,
	//   to be able to detect an overshoot and undo it, all the "normal" weights
	//   are saved in applyGradients() before applying the gradients, and used
	//   for computation of L and adjustment of the training rate in the next
	//   applyGradients(). If an overshoot is found, applyGradients() will
	//   revert to the saved weights instead of moving forwards.
	// * Averaging the inputs, weights, and gradients of the last layer over all
	//   training cases is technically incorrect, because we're computing the
	//   function as a sum of errors on all training cases, so we should be taking
	//   the separate squares of all the differences on all the training cases.
	//   But that would require a lot of memory. Since the weights are adjusted
	//   from averages anyway, we use the squares of differences of sums across
	//   all the training cases and hope that it's close enough. (Or should that
	//   be averaging the sum of absolute values?) Since the gradients and inputs
	//   are summed from all the training cases, the weights are also multiplied
	//   by the number of training cases (since they are the same on all cases) for
	//   the computation.
	// * Since the offset weights with CORNER activation are driven by a completely separate
	//   function, its value and gradient could be excluded from the common computation of
	//   the rate and have their own computation (what computation is right for them?).
	//   But at least for now they're included into the common computation.

	// -- Values saved from last pass.

	// This pass is a redo: we've found that the L value assumed in a pass was too
	// low, so the next pass was aborted, options_.trainingRate_ reduced, and
	// now this pass is a redo of the aborted pass.
	bool autoRateRedo_ = false;
	// This is the next pass after redo, don't adjust the weights, to avoid
	// the situation when the rate keeps going down stuck in the same pass.
	bool autoRateAfterRedo_ = false;

	// Inputs to the last layer. These are the same as the activated
	// values of the pass before last, summed from all training cases. Has the
	// same structure as trainDataActivated(*, lastLevel() - 1).
	ValueVector lp_linputs_;

	// Gradients for inputs to the last layer, also summed
	// as usual from all training cases. Saved from s_cur on the last layer.
	ValueVector lp_linputs_grads_;

	// All weights.
	ValueVector lp_weights_;

	// Same as weightsOfLevel() but for the last pass.
	ValueSubMatrix lpWeightsOfLevel(size_t level)
	{
		return ValueSubMatrix(lp_weights_, level_weights_idx_[level], levels_[level-1] + 1, levels_[level]);
	}

	// Gradients of weights of the last layer, summed as usual
	// from all training cases. Has the same structure as gradientOfLevel(lastLevel()).
	// TODO: lp_lweights_grads_ can be made a part of lastTotalGradient_.
	ValueVector lp_lweights_grads_;

	// All weights of the CORNER activation function for the last pass.
	ValueVector lp_corners_;

	// Gradients of weights of the CORNER activation function for last layer.
	// Has the same structure as cornersGradientOfLevel(lastLevel()).
	// TODO: lp_lcorners_grads_ can be made a part of lastTotalCornersGradient_.
	ValueVector lp_lcorners_grads_;

	// -- Values being computed on the current pass. (In addition to the
	// usual weights and gradients).
	
	// Inputs to the last layer. These are the same as the activated
	// values of the pass before last, summed from all training cases. Has the
	// same structure as trainDataActivated(*, lastLevel() - 1).
	ValueVector cp_linputs_;

	// Gradients for inputs to the last layer, also summed
	// as usual from all training cases. Saved from s_cur on the last layer.
	ValueVector cp_linputs_grads_;

	// --------------
	// Auto-computation of training rate, version 2.
	// This version relies on the rate provided from the outside as a baseline
	// and does its own adjustment on top of it.

	// The rate gets multiplied by this before use.
	Value autoRate2Adj_ = 1.;

	// --------------
	// Stats about training.

	// How many training passes have been completed. This number may go back
	// if the auto-rate logic decides to do an undo.
	int64_t trainingPass_ = 0;
	// Running sum of absolute errors (losses) in the pass.
	double errorAbsSum_ = 0.;
	// Running sum of squared errors (losses) in the pass, for real and effective
	// cases.
	double errorSqSum_ = 0.;
	double errorEffectiveSqSum_ = 0.;
	// Number of training cases in the current pass.
	UsageCount cases_ = 0;
	// Effective number of training cases in the current pass, i.e. adjusted
	// to help even the rare and unusual cases to be classified correctly.
	Value effectiveCases_ = 0;
	// Number of values added up in the error sums. Pass size multiplied by the
	// number of outputs. Real and effective versions.
	UsageCount errorCases_ = 0;
	Value errorEffectiveCases_ = 0;
	// In the classifier mode (options_.isClassifier_ = true), counts of the
	// cases that have been classified correctly and incorrectly. These are real
	// cases, not effective.
	UsageCount correctCases_ = 0;
	UsageCount incorrectCases_ = 0;
	// In the classifier mode (options_.isClassifier_ = true), counts of the
	// cases that have been classified both correctly and above 0, and those
	// that are not. These are real cases, not effective.
	UsageCount aboveCases_ = 0;
	UsageCount notAboveCases_ = 0;

	// Gradient across all layers, computed over the pass.
	double toal_all_gradient_norm2_;
	// Gradient by layers, computed over the pass.
	std::vector<double> total_gradient_norm2_;

	// --------------

	// Compute the result for one set of inputs for training, keeping all the
	// intermediate values.
	//
	// @param inputs - the inputs for the computation. The size of the vector
	//   must match the size of the 0th level of the network.
	// @param traindata - the computed values of the network, with the size and
	//   structure as defined in level_train_idx_.
	Erref computeForTraining(const ValueVector &inputs, ValueVector &traindata);

	// Do backpropagation by one level
	// @param level - level of neurons to backpropagate through, must be at least 1
	// @param s_cur - current level's sigma to fill, that will be sent to the previous level
	// @param s_next - sigma values received from the next level (not adjusted for
	//   derivative of this level's activation function)
	// @param der - values of derivative of this level's activation function
	// @param prev_activ - activated inputs (from the current training case)
	// @param rate - the descent rate
	// @param inactiv - in activated result (from the current training case)
	// @param debug - flag: print debugging info
	void backpropLevel(size_t level, ValueVector &s_cur, const ValueVector &s_next,
		const ValueVector &der, ValueSubVector prev_activ, ValueSubVector inactiv,
		Value rate, bool debug);

	// Saturate a weight.
	// @param val - new value of a weight to check for saturation and saturate,
	//    will get updated in-place
	// @param oldval - previous value of the same weight, used to handle the
	//   floor saturation.
	// @return - true if saturated and val has been updated, false if not
	inline bool saturate(Value &val, Value oldval)
	{
		const Value limit = options_.weightSaturation_;
		if (val > limit) {
			val = limit;
			return true;
		} else if (val < -limit) {
			val = -limit;
			return true;
		} else if (options_.enableWeightFloor_) {
			const Value floor = options_.weightFloor_;
			if (oldval > 0) {
				if (val < floor) {
					if (oldval > floor) {
						// stop at the floor limit
						val = floor;
						return true;
					} else {
						// if were already at the floor, teleport
						val = -floor;
						// printf("DEBUG teleported down to %f\n", val);
						return true;
					}
				}
			} else {
				if (val > -floor) {
					if (oldval < -floor) {
						// stop at the floor limit
						val = -floor;
						return true;
					} else {
						// if were already at the floor, teleport
						val = floor;
						// printf("DEBUG teleported up to %f\n", val);
						return true;
					}
				}
			}
		}
		return false;
	}

	// --------------

	// Make sure that the options are self-consistent by force-changing any inconsistent values.
	void reconcileOptions();

private:
	// Those might be worth implementing later.
	FloatNeuralNet(const FloatNeuralNet &t);
	void operator=(const FloatNeuralNet &t);
};

template<typename T>
inline FloatNeuralNet::SubVector<T> FloatNeuralNet::SubMatrix<T>::rowAsVector(size_t row) const
{
	return SubVector<T>(&at(row, 0), 0, rowsize_);
}

template<typename T>
inline FloatNeuralNet::SubVector<T> FloatNeuralNet::SubMatrix<T>::asVector() const
{
	return SubVector<T>(&at(0, 0), 0, rowsize_ * rowcount_);
}

}; // TRICEPS_NS

#endif // __Triceps_FloatNeuralNet_h__
