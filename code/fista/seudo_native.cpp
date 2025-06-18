#include <stdio.h>

#include "mex.h"
#include "matrix.h"
#include "seudo.hpp"
#include "strprintf.hpp"

// This is a workaround for a symbol defined in the new libstac++ 6.30
// but not in matlab's copy of version 6.28. Comment out when Matlab
// catches up.
namespace std
{

void
__throw_bad_array_new_length(void)
{
	throw std::bad_array_new_length();
}

};

// Inputs:
//   double image[ht][wd] - image to match
//   double blob[blobht][blobwd] - blob used to detect spurious lightings
//   double blob_spacing - During computation, place blobs on a grid of this many pixels
//       increase between the points, both vertically and horizontally. The weights for
//       blobs in between will be filled with 0s. If negative, the absolute value is used
//       to compute the pixel count as a fraction of smallest blob dimension. If positive,
//       is the pixel count and must be at least 1 pixel.
//   double rois[ht*wd][nrois] - Regions Of Interest for the cells to match
//   double weights[nrois + ht*wd] - initial weights, if empty, will be initialized to 0s
//   double lambda[nrois + ht*wd] - LASSO lambda values, if empty, will be initialized to 0s
//   double eps - "epsilon" setting the limit for optimization according to stop_mode
//   int max_steps - max number of steps to take
//   int l_mode - a bitmask:
//     bit 0: 0 for dynamic L (as in TFOCS), 1 for static multi-L,
//     bit 1: 0 for classic, 2 for fast brake
//     bit 2: 0 for classic, 4 for fast brake controlling FISTA Nu parameter
//   int stop_mode - 0 for relative norm2 (as in TFOCS), 1 for norm2, 2 for every dimension
//   bool verbose - enable debugging logging
//   int parallel - number of parallel threads to use
// Outputs:
//   double weights[nrois + ht*wd] - weights after optimization
//   (optional) int n_steps - number of steps taken by the optimizer
//   (optional) string log - verbose log
enum InArgIdx {
	InImage,
	InBlob,
	InBlobSpacing,
	InRois,
	InWeights,
	InLambda,
	InEps,
	InMaxSteps,
	InLMode,
	InStopMode,
	InVerbose,
	InParallel,
	InArgCount, // Count of arguments
};
enum OutArgIdx {
	OutWeights,
	OutNSteps,
	OutLog,
	OutArgCount, // Count of arguments
};

// Prevents the leaks of generated strings on errors.
static std::string errorMsg;

// Load a fragment of matrix from Matlab into a FISTA vector.
// @param descr - description for debugging prints, not NULL enables the prints
// @param dst - destination vector, containng a matrix in normal C order
// @param startdst - starting position in the destination where the matrix starts
// @param wd - width of the matrix
// @param ht - height of the matrix
// @param src - source Matlab matrix
// @param startsrc - starting position in the source where the matrix starts
template <typename T>
void loadMatrixFragment(const char *descr, Fista::Vector &dst, int startdst, int wd, int ht, const mxArray *src, int startsrc)
{
	// This may cause a leak but it's a small problem compared to memory corruption
	if (dst.size() < startdst + wd*ht) {
		errorMsg = strprintf("vector (%s) is too small, has %d, needs %d", descr, (int)dst.size(), startdst + wd*ht);
		mexErrMsgIdAndTxt("seudo_native:Internal", errorMsg.c_str());
	}

	T *data = (T*)mxGetData(src);
	for (int y = 0; y < ht; y++) {
		for (int x = 0; x < wd; x++) {
			double v = (double) data[startsrc + x*ht + y]; // Matlab has dimensions in backwards order
			dst[startdst + y*wd + x] = v;
			if (descr != NULL) mexPrintf("%s(%d, %d) = %f\n", descr, y + 1, x + 1, v);
		}
	}
}

// Load a matrix fragment from Matlab into a FISTA vector.
// @param descr - description for debugging prints, not NULL enables the prints
// @param dst - destination vector, containng a matrix in normal C order
// @param startdst - starting position in the destination where the matrix starts
// @param wd - width of the matrix
// @param ht - height of the matrix
// @param src - source Matlab matrix
// @param startsrc - starting position in the source where the matrix starts
void loadNumericMatrixFragment(const char *descr, Fista::Vector &dst, int startdst, int wd, int ht, const mxArray *src, int startsrc)
{
	switch(mxGetClassID(src)) {
	case mxDOUBLE_CLASS:
		loadMatrixFragment<double>(descr, dst, startdst, wd, ht, src, startsrc);
		break;
	case mxSINGLE_CLASS:
		loadMatrixFragment<float>(descr, dst, startdst, wd, ht, src, startsrc);
		break;
	case mxINT8_CLASS:
		loadMatrixFragment<int8_t>(descr, dst, startdst, wd, ht, src, startsrc);
		break;
	case mxUINT8_CLASS:
		loadMatrixFragment<uint8_t>(descr, dst, startdst, wd, ht, src, startsrc);
		break;
	case mxINT16_CLASS:
		loadMatrixFragment<int16_t>(descr, dst, startdst, wd, ht, src, startsrc);
		break;
	case mxUINT16_CLASS:
		loadMatrixFragment<uint16_t>(descr, dst, startdst, wd, ht, src, startsrc);
		break;
	case mxINT32_CLASS:
		loadMatrixFragment<int32_t>(descr, dst, startdst, wd, ht, src, startsrc);
		break;
	case mxUINT32_CLASS:
		loadMatrixFragment<uint32_t>(descr, dst, startdst, wd, ht, src, startsrc);
		break;
	case mxINT64_CLASS:
		loadMatrixFragment<int64_t>(descr, dst, startdst, wd, ht, src, startsrc);
		break;
	case mxUINT64_CLASS:
		loadMatrixFragment<uint64_t>(descr, dst, startdst, wd, ht, src, startsrc);
		break;
	};
}

// Load a matrix from Matlab into a FISTA vector.
// @param descr - description for debugging prints, not NULL enables the prints
// @param dst - destination vector, containng a matrix in normal C order
// @param wd - width of the matrix
// @param ht - height of the matrix
// @param src - source Matlab matrix
void loadNumericMatrix(const char *descr, Fista::Vector &dst, int wd, int ht, const mxArray *src)
{
	loadNumericMatrixFragment(descr, dst, /*startdst*/ 0, wd, ht, src, /*startsrc*/ 0);
}

// Load a scalar from Matlab into a FISTA vector.
// @param descr - description for debugging prints, not NULL enables the prints
// @param src - source Matlab scalar
// @return - the scalr value
double loadScalar(const char *descr, const mxArray *src)
{
	double v = mxGetScalar(src);
	if (descr != NULL) mexPrintf("%s(1, 1) = %f\n", descr, v);
	return v;
}

// Check that the value of an argument is numeric and a matrix, log a Matlab
// error if not (which also exits any current code).
// @param idx - argument index
// @param name - argument name
// @param arg - value to check
void checkNumericMatrix(int idx, const char *name, const mxArray *arg)
{
	if (!mxIsNumeric(arg) || mxIsComplex(arg)) {
		errorMsg = strprintf("argument %d (%s) must contain numbers", idx + 1, name);
		mexErrMsgIdAndTxt("seudo_native:Args", errorMsg.c_str());
	}
	if (mxGetNumberOfDimensions(arg) != 2) {
		errorMsg = strprintf("argument %d (%s) must contain 2 dimensions", idx + 1, name);
		mexErrMsgIdAndTxt("seudo_native:Args", errorMsg.c_str());
	}
}

// Check that the value of an argument is numeric and a scalar, log a Matlab
// error if not (which also exits any current code).
// @param idx - argument index
// @param name - argument name
// @param arg - value to check
void checkNumericScalar(int idx, const char *name, const mxArray *arg)
{
	if (!mxIsNumeric(arg) || mxIsComplex(arg)) {
		errorMsg = strprintf("argument %d (%s) must contain a number", idx + 1, name);
		mexErrMsgIdAndTxt("seudo_native:Args", errorMsg.c_str());
	}
	if (mxGetNumberOfElements(arg) != 1) {
		errorMsg = strprintf("argument %d (%s) must contain a scalar", idx + 1, name);
		mexErrMsgIdAndTxt("seudo_native:Args", errorMsg.c_str());
	}
}

// Check that the value of an argument is logical and a scalar, log a Matlab
// error if not (which also exits any current code).
// @param idx - argument index
// @param name - argument name
// @param arg - value to check
void checkLogicalScalar(int idx, const char *name, const mxArray *arg)
{
	if (!mxIsLogical(arg)) {
		errorMsg = strprintf("argument %d (%s) must contain a logical", idx + 1, name);
		mexErrMsgIdAndTxt("seudo_native:Args", errorMsg.c_str());
	}
	if (mxGetNumberOfElements(arg) != 1) {
		errorMsg = strprintf("argument %d (%s) must contain a scalar", idx + 1, name);
		mexErrMsgIdAndTxt("seudo_native:Args", errorMsg.c_str());
	}
}

static std::shared_ptr<Fista::Seudo>
parseArguments(int nlhs, int nrhs, const mxArray *prhs[])
{
	if (nrhs != InArgCount || nlhs > OutArgCount) {
		mexErrMsgIdAndTxt(
			"seudo_native:Usage",
			"Use: weights[,n_steps[,log]] = seudo_native(image, blob, blob_spacing, rois, weights, lambda, eps, max_steps, l_mode, stop_mode, verbose, parallel)\n"
			" Inputs:\n"
			"   double image[ht][wd] - image to match\n"
			"   double blob[blobht][blobwd] - blob used to detect spurious lightings\n"
			"   double blob_spacing - During computation, place blobs on a grid of this many pixels\n"
			"       increase between the points, both vertically and horizontally. The weights for\n"
			"       blobs in between will be filled with 0s. If negative, the absolute value is used\n"
			"       to compute the pixel count as a fraction of smallest blob dimension. If positive,\n"
			"       is the pixel count and must be at least 1 pixel.\n"
			"   double rois[ht*wd][nrois] - Regions Of Interest for the cells to match\n"
			"   double weights[nrois + ht*wd] - initial weights, if empty, will be initialized to 0s\n"
			"   double lambda[nrois + ht*wd] - LASSO lambda values, if empty, will be initialized to 0s\n"
			"   double eps - 'epsilon' setting the limit for optimization according to stop_mode\n"
			"   int max_steps - max number of steps to take\n"
			"   int l_mode - a bitmask:\n"
			"     bit 0: 0 for dynamic L (as in TFOCS), 1 for static multi-L,\n"
			"     bit 1: 0 for classic, 2 for fast brake\n"
			"     bit 2: 0 for classic, 4 for fast brake controlling FISTA Nu parameter\n"
			"   int stop_mode - 0 for relative norm2 (as in TFOCS), 1 for norm2, 2 for every dimension\n"
			"   bool verbose - enable debugging logging\n"
			"   int parallel - number of parallel threads to use\n"
			" Outputs:\n"
			"   double weights[nrois + ht*wd] - weights after optimization\n"
			"   (optional) int n_steps - number of steps taken by the optimizer\n"
			"   (optional) string log - verbose log"
			);
		return nullptr;
	}

	// Check everything up front before constructing anything, or Matlab will
	// leak stuff on errors.

	// --- image
	checkNumericMatrix(InImage, "image", prhs[InImage]);
	int ht = mxGetM(prhs[InImage]);
	int wd = mxGetN(prhs[InImage]);

	// --- blob
	checkNumericMatrix(InBlob, "blob", prhs[InBlob]);
	int blobht = mxGetM(prhs[InBlob]);
	int blobwd = mxGetN(prhs[InBlob]);

	// --- blob_spacing
	checkNumericScalar(InBlobSpacing, "blob_spacing", prhs[InBlobSpacing]);
	double blobSpacing = loadScalar(NULL, prhs[InBlobSpacing]);
	if (blobSpacing < 0.) {
		blobSpacing = (- blobSpacing) * std::min(blobwd, blobht);
		if (blobSpacing < 1.)
			blobSpacing = 1.;
	} else {
		if (blobSpacing < 1.) {
			errorMsg = strprintf("argument %d (%s) must be at least 1", InBlobSpacing + 1, "blob_spacing");
			mexErrMsgIdAndTxt("seudo_native:Args", errorMsg.c_str());
		}
	}

	// --- rois
	checkNumericMatrix(InRois, "rois", prhs[InRois]);
	int rois_d0 = mxGetM(prhs[InRois]);
	int n_rois = mxGetN(prhs[InRois]);

	if (n_rois > 0 && rois_d0 != wd * ht) {
		errorMsg = strprintf("argument %d (%s) first dimension size must be equal to (wd*ht=%d) of image, got %d",
			InRois + 1, "rois", wd*ht, rois_d0);
		mexErrMsgIdAndTxt("seudo_native:Args", errorMsg.c_str());
	}

	// --- weights
	checkNumericMatrix(InWeights, "weights", prhs[InWeights]);
	int weights_d0 = mxGetM(prhs[InWeights]);
	int weights_d1 = mxGetN(prhs[InWeights]);
	if (weights_d0 != 0 && weights_d0 != 1 && weights_d1 != 1) {
		errorMsg = strprintf("argument %d (%s) must be a unidimensional vector", InWeights + 1, "weights");
		mexErrMsgIdAndTxt("seudo_native:Args", errorMsg.c_str());
	}
	int n_weights = weights_d0 * weights_d1;
	if (n_weights != n_rois + wd*ht && n_weights != 0) {
		errorMsg = strprintf("argument %d (%s) must be empty or contain (nrois + ht*wd = %d) numbers, got %d",
				InWeights + 1, "weights", n_rois + wd*ht, n_weights);
		mexErrMsgIdAndTxt("seudo_native:Args", errorMsg.c_str());
	}

	// --- lambda
	checkNumericMatrix(InLambda, "lambda", prhs[InLambda]);
	int lambda_d0 = mxGetM(prhs[InLambda]);
	int lambda_d1 = mxGetN(prhs[InLambda]);
	if (lambda_d0 != 0 && lambda_d0 != 1 && lambda_d1 != 1) {
		errorMsg = strprintf("argument %d (%s) must be a unidimensional vector", InLambda + 1, "lambda");
		mexErrMsgIdAndTxt("seudo_native:Args", errorMsg.c_str());
	}
	int n_lambda = lambda_d0 * lambda_d1;
	if (n_lambda != n_rois + wd*ht && n_lambda != 0) {
		errorMsg = strprintf("argument %d (%s) must be empty or contain (nrois + ht*wd = %d) numbers, got %d",
				InLambda + 1, "lambda", n_rois + wd*ht, n_lambda);
		mexErrMsgIdAndTxt("seudo_native:Args", errorMsg.c_str());
	}

	// --- eps
	checkNumericScalar(InEps, "eps", prhs[InEps]);

	// --- max_steps
	checkNumericScalar(InMaxSteps, "max_steps", prhs[InMaxSteps]);

	// --- l_mode
	checkNumericScalar(InLMode, "l_mode", prhs[InLMode]);
	int l_mode = loadScalar(NULL, prhs[InLMode]);
	if (l_mode < 0 || l_mode > 7) {
		errorMsg = strprintf("argument %d (%s) must be in range 0-7, got %d", InLMode + 1, "l_mode", l_mode);
		mexErrMsgIdAndTxt("seudo_native:Args", errorMsg.c_str());
	}

	// --- stop_mode
	checkNumericScalar(InStopMode, "stop_mode", prhs[InStopMode]);
	int stop_mode = loadScalar(NULL, prhs[InStopMode]);
	if (stop_mode < 0 || stop_mode > 2) {
		errorMsg = strprintf("argument %d (%s) must be in range 0-2, got %d", InStopMode + 1, "stop_mode", stop_mode);
		mexErrMsgIdAndTxt("seudo_native:Args", errorMsg.c_str());
	}

	// --- verbose
	checkLogicalScalar(InVerbose, "verbose", prhs[InVerbose]);

	// --- parallel
	checkNumericScalar(InParallel, "parallel", prhs[InParallel]);

	// Now get all the values.

	// --- image
	auto seudo = std::make_shared<Fista::Seudo>(wd, ht);
	loadNumericMatrix(NULL, seudo->image_.img_, wd, ht, prhs[InImage]);

	// --- blob
	// The origin of the blob is at its center. When the size is odd, the rounding
	// down after division conveniently points to the central pixel.
	auto blob = std::make_shared<Fista::Sprite>(blobwd, blobht, blobwd/2, blobht/2);
	loadNumericMatrix(NULL, blob->img_, blobwd, blobht, prhs[InBlob]);
	blob->cropWhitespace();
	seudo->blob_ = blob;

	// --- blob_spacing
	seudo->blobSpacing_ = blobSpacing;

	// --- rois
	for (int i = 0; i < n_rois; i++) {
		// ROIs always have the origin in top left corner.
		auto rois = std::make_shared<Fista::Sprite>(wd, ht, 0, 0);
		// The weird ordering of dimensions in both Matlab and SEUDO works
		// out well and makes this simple iteration possible.
		loadNumericMatrixFragment(NULL, rois->img_, 0, wd, ht, prhs[InRois], i * rois_d0);
		rois->cropWhitespace();
		seudo->rois_.emplace_back(rois);
	}

	// --- weights
	if (n_weights == 0) {
		// Auto-fill with 0s.
		seudo->weights_.assign(n_rois + wd*ht, 0.);
	} else {
		seudo->weights_.resize(n_weights);
		loadNumericMatrixFragment(NULL, seudo->weights_, /*startdst*/ 0, /*wd*/ n_rois, /*ht*/ 1, prhs[InWeights], /*startsrc*/ 0);
		loadNumericMatrixFragment(NULL, seudo->weights_, /*startdst*/ n_rois, wd, ht, prhs[InWeights], /*startsrc*/ n_rois);
	}

	// --- lambda
	if (n_lambda == 0) {
		// Auto-fill with 0s.
		seudo->lambda_.assign(n_rois + wd*ht, 0.);
	} else {
		seudo->lambda_.resize(n_lambda);
		loadNumericMatrixFragment(NULL, seudo->lambda_, /*startdst*/ 0, /*wd*/ n_rois, /*ht*/ 1, prhs[InLambda], /*startsrc*/ 0);
		loadNumericMatrixFragment(NULL, seudo->lambda_, /*startdst*/ n_rois, wd, ht, prhs[InLambda], /*startsrc*/ n_rois);
	}

	// --- eps
	seudo->eps_ = loadScalar(NULL, prhs[InEps]);

	// --- max_steps
	seudo->maxSteps_ = loadScalar(NULL, prhs[InMaxSteps]);

	// --- l_mode
	seudo->multiGrad_ = ((l_mode & 1) != 0);
	seudo->fastBrake_ = ((l_mode & 2) != 0);
	seudo->fastBrakeNu_ = ((l_mode & 4) != 0);

	// --- stop_mode
	switch (stop_mode) {
	case 0:
		seudo->stopping_ = Fista::Run::StopEpsNorm2Rel;
		break;
	case 1:
		seudo->stopping_ = Fista::Run::StopEpsNorm2;
		break;
	case 2:
		seudo->stopping_ = Fista::Run::StopEpsEveryDimension;
		break;
	}

	// --- verbose
	seudo->verbose_ = loadScalar(NULL, prhs[InVerbose]);

	// --- parallel
	seudo->setNumThreads(loadScalar(NULL, prhs[InParallel]));

	return seudo;
}

extern "C" {

void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	auto seudo = parseArguments(nlhs, nrhs, prhs);

	seudo->compute();

	if (nlhs > OutWeights) {
		int weights_d0 = mxGetM(prhs[InWeights]);
		int weights_d1 = mxGetN(prhs[InWeights]);

		size_t n_weights = seudo->weights_.size();
		size_t n_rois = seudo->rois_.size();
		int wd = seudo->image_.wd_;
		int ht = seudo->image_.ht_;

		if (weights_d0 == 0) {
			// have no known shape, return same as in SEUDO Matlab code, as a column
			weights_d0 = n_weights;
			weights_d1 = 1;
		}

		plhs[OutWeights] = mxCreateDoubleMatrix(weights_d0, weights_d1, mxREAL);
		double *data = (double *)mxGetData(plhs[OutWeights]);
		for (int i = 0; i < n_rois; i++) {
			data[i] = seudo->weights_[i];
		}
		// Matlab has X and Y dimensions transposed.
		for (int y = 0; y < ht; y++) {
			for (int x = 0; x < wd; x++) {
				data[n_rois + x*ht + y] = seudo->weights_[n_rois + y*wd + x];
			}
		}
	}

	if (nlhs > OutNSteps) {
		plhs[OutNSteps] = mxCreateDoubleScalar(seudo->stepsTaken_);
	}

	if (nlhs > OutLog) {
		plhs[OutLog] = mxCreateString(seudo->log_.c_str());
	}
}

};


#if 0 // {
#include "mexAdapter.hpp"

using namespace matlab::data;
using matlab::mex::ArgumentList;

class MexFunction : public matlab::mex::Function {
public:

	void operator()(ArgumentList outputs, ArgumentList inputs) {
		std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr = getEngine();
		ArrayFactory factory;

		if (false) { // SBXXX test for the cost of the call - do nothing and return immediately
			// Parsing turns out to be the most expensive part.
			std::shared_ptr<Fista::Seudo> seudo = parseArguments(outputs, inputs);

			if (outputs.size() > OutWeights && inputs.size() > InWeights) {
				outputs[OutWeights] = inputs[InWeights];
			}
			if (outputs.size() > OutNSteps) {
				TypedArray<double> doubleArray = factory.createArray<double>({1});
				doubleArray[0] = 0.;
				outputs[OutNSteps] = doubleArray;
			}
			if (outputs.size() > OutLog) {
				outputs[OutLog] = factory.createScalar("");
			}
			return;
		}

		std::shared_ptr<Fista::Seudo> seudo = parseArguments(outputs, inputs);
		if (!seudo)
			return;

		seudo->compute();
		
#if 0
		if (!seudo->log_.empty()) {
			matlabPtr->feval(u"display",
				0,
				std::vector<Array>({ factory.createScalar(seudo->log_.c_str()) }));
		}
#endif

		if (!seudo->error_.empty()) {
			matlabPtr->feval(u"error",
				0,
				std::vector<Array>({ factory.createScalar(seudo->error_.c_str()) }));
			return;
		}

		if (outputs.size() > OutWeights) {
			size_t n_weights = seudo->weights_.size();
			size_t n_rois = seudo->rois_.size();
			int wd = seudo->image_.wd_;
			int ht = seudo->image_.ht_;
			// Preserve the same shape as on input, which is expected by the Matlab code.
			TypedArray<double> doubleArray = factory.createArray<double>(inputs[InWeights].getDimensions());
			for (int i = 0; i < n_rois; i++) {
				doubleArray[i] = seudo->weights_[i];
			}
			// Matlab has X and Y dimensions transposed.
			for (int y = 0; y < ht; y++) {
				for (int x = 0; x < wd; x++) {
					doubleArray[n_rois + x*ht + y] = seudo->weights_[n_rois + y*wd + x];
				}
			}
			outputs[OutWeights] = doubleArray;
		}
		if (outputs.size() > OutNSteps) {
			TypedArray<double> doubleArray = factory.createArray<double>({1});
			doubleArray[0] = (double)seudo->stepsTaken_;
			outputs[OutNSteps] = doubleArray;
		}
		if (outputs.size() > OutLog) {
			outputs[OutLog] = factory.createScalar(seudo->log_.c_str());
		}

	}

};
#endif // }
