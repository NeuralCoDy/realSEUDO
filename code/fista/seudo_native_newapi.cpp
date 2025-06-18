#include <stdio.h>

#include "mex.hpp"
#include "mexAdapter.hpp"
#include "seudo.hpp"
#include "strprintf.hpp"

using namespace matlab::data;
using matlab::mex::ArgumentList;

class MexFunction : public matlab::mex::Function {
public:
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
	//   int l_mode - 0 for dynamic L (as in TFOCS), 1 for static multi-L,
	//       2 for dynamic L + fast brake, 3 for static multi-L + fast brake
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

	std::shared_ptr<Fista::Seudo> parseArguments(ArgumentList outputs, ArgumentList inputs) {
		// Get pointer to engine
		std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr = getEngine();

		// Get array factory
		ArrayFactory factory;

		if (inputs.size() != InArgCount || outputs.size() > OutArgCount) {
			matlabPtr->feval(u"error",
				0,
				std::vector<Array>({ factory.createScalar(
					"Use: weights[,n_steps] = seudo_native(image, blob, rois, weights, lambda, eps, max_steps, verbose)\n"
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
					"   int l_mode - 0 for dynamic L (as in TFOCS), 1 for static multi-L\n"
					"       2 for dynamic L + fast brake, 3 for static multi-L + fast brake\n"
					"   int stop_mode - 0 for relative norm2 (as in TFOCS), 1 for norm2, 2 for every dimension\n"
					"   bool verbose - enable debugging logging\n"
					"   int parallel - number of parallel threads to use\n"
					" Outputs:\n"
					"   double weights[nrois + ht*wd] - weights after optimization\n"
					"   (optional) int n_steps - number of steps taken by the optimizer\n"
					"   (optional) string log - verbose log"
				) }));
			return nullptr;
		}

		// --- image

		if (inputs[InImage].getType() != ArrayType::DOUBLE)
		{
			matlabPtr->feval(u"error",
				0,
				std::vector<Array>({ factory.createScalar("image must contain floating-point numbers") }));
			return nullptr;
		}
		auto inImageDims = inputs[InImage].getDimensions();
		if (inImageDims.size() != 2) {
			matlabPtr->feval(u"error",
				0,
				std::vector<Array>({ factory.createScalar("image must contain 2 dimensions") }));
			return nullptr;
		}

		int ht = inImageDims[0];
		int wd = inImageDims[1];
		auto seudo = std::make_shared<Fista::Seudo>(wd, ht);
		for (int y = 0; y < ht; y++) {
			for (int x = 0; x < wd; x++) {
				seudo->image_.img_[y*wd + x] = inputs[InImage][y][x];
			}
		}

		// --- blob

		if (inputs[InBlob].getType() != ArrayType::DOUBLE)
		{
			matlabPtr->feval(u"error",
				0,
				std::vector<Array>({ factory.createScalar("blob must contain floating-point numbers") }));
			return nullptr;
		}
		auto inBlobDims = inputs[InBlob].getDimensions();
		if (inBlobDims.size() != 2) {
			matlabPtr->feval(u"error",
				0,
				std::vector<Array>({ factory.createScalar("blob must contain 2 dimensions") }));
			return nullptr;
		}

		int blobht = inBlobDims[0];
		int blobwd = inBlobDims[1];

		// The origin of the blob is at its center. When the size is odd, the rounding
		// down after division conveniently points to the central pixel.
		auto blob = std::make_shared<Fista::Sprite>(blobwd, blobht, blobwd/2, blobht/2);
		seudo->blob_ = blob;
		for (int y = 0; y < blobht; y++) {
			for (int x = 0; x < blobwd; x++) {
				blob->img_[y*blobwd + x] = (double)inputs[InBlob][y][x];
			}
		}
		blob->cropWhitespace();

		// --- blob_spacing

		if (inputs[InBlobSpacing].getType() != ArrayType::DOUBLE ||
			inputs[InBlobSpacing].getNumberOfElements() != 1)
		{
			matlabPtr->feval(u"error",
				0,
				std::vector<Array>({ factory.createScalar("blob_spacing must be scalar double") }));
			return nullptr;
		}
		double blobSpacing = inputs[InBlobSpacing][0];
		if (blobSpacing < 0) {
			blobSpacing = (- blobSpacing) * std::min(blobwd, blobht);
			if (blobSpacing < 1.)
				blobSpacing = 1.;
		} else {
			if (blobSpacing < 1.) {
				matlabPtr->feval(u"error",
					0,
					std::vector<Array>({ factory.createScalar("non-negative blob_spacing must be at least 1") }));
				return nullptr;
			}
		}
		seudo->blobSpacing_ = blobSpacing;

		// --- rois

		if (inputs[InRois].getType() != ArrayType::DOUBLE)
		{
			matlabPtr->feval(u"error",
				0,
				std::vector<Array>({ factory.createScalar("rois must contain floating-point numbers") }));
			return nullptr;
		}
		auto inRoisDims = inputs[InRois].getDimensions();
		if (inRoisDims.size() != 2) {
			matlabPtr->feval(u"error",
				0,
				std::vector<Array>({ factory.createScalar("rois must contain 2 dimensions") }));
			return nullptr;
		}

		int rois_d0 = inRoisDims[0];
		int n_rois = inRoisDims[1];

		if (rois_d0 != wd * ht) {
			matlabPtr->feval(u"error",
				0,
				std::vector<Array>({ factory.createScalar("rois 1st dimension size must be equal (wd*ht) of image") }));
			return nullptr;
		}

		// The origin of the blob is at its center. When the size is odd, the rounding
		// down after division conveniently points to the central pixel.
		auto &roisArg = inputs[InRois];
		for (int i = 0; i < n_rois; i++) {
			// ROIs always have the origin in top left corner.
			auto rois = std::make_shared<Fista::Sprite>(wd, ht, 0, 0);
			for (int y = 0; y < ht; y++) {
				for (int x = 0; x < wd; x++) {
					rois->img_[y*wd + x] = roisArg[x*ht + y][i];
				}
			}
			rois->cropWhitespace();
			seudo->rois_.emplace_back(rois);
		}

		// --- weights

		auto inWeightsDims = inputs[InWeights].getDimensions();
		// Everything in Matlab is at least a matrix.
		if (inWeightsDims.size() != 2 || (inWeightsDims[0] != 0 && inWeightsDims[0] != 1 && inWeightsDims[1] != 1)) {
			matlabPtr->feval(u"error",
				0,
				std::vector<Array>({ factory.createScalar("weights must contain 1 dimensions") }));
			return nullptr;
		}

		int n_weights = inputs[InWeights].getNumberOfElements();

		if (n_weights == 0) {
			// Auto-fill with 0s.
			seudo->weights_.assign(n_rois + wd*ht, 0.);
		} else {
			if (inputs[InWeights].getType() != ArrayType::DOUBLE)
			{
				matlabPtr->feval(u"error",
					0,
					std::vector<Array>({ factory.createScalar("weights must contain floating-point numbers") }));
				return nullptr;
			}

			if (n_weights != n_rois + wd*ht) {
				matlabPtr->feval(u"error",
					0,
					std::vector<Array>({ factory.createScalar("weights must contain (nrois + ht*wd) numbers") }));
				return nullptr;
			}

			seudo->weights_.resize(n_weights);
			auto &weightsArg = inputs[InWeights];
			for (int i = 0; i < n_rois; i++) {
				seudo->weights_[i] = weightsArg[i];
			}
			// Matlab has X and Y dimensions transposed.
			for (int y = 0; y < ht; y++) {
				for (int x = 0; x < wd; x++) {
					seudo->weights_[n_rois + y*wd + x] = weightsArg[n_rois + x*ht + y];
				}
			}
		}

		// --- lambda

		auto inLambdaDims = inputs[InLambda].getDimensions();
		// Everything in Matlab is at least a matrix.
		if (inLambdaDims.size() != 2 || (inLambdaDims[0] != 0 && inLambdaDims[0] != 1 && inLambdaDims[1] != 1)) {
			matlabPtr->feval(u"error",
				0,
				std::vector<Array>({ factory.createScalar("lambda must contain 1 dimensions") }));
			return nullptr;
		}

		int n_lambda = inputs[InLambda].getNumberOfElements();

		if (n_lambda == 0) {
			// Auto-fill with 0s.
			seudo->lambda_.assign(n_rois + wd*ht, 0.);
		} else {
			if (inputs[InLambda].getType() != ArrayType::DOUBLE)
			{
				matlabPtr->feval(u"error",
					0,
					std::vector<Array>({ factory.createScalar("lambda must contain floating-point numbers") }));
				return nullptr;
			}

			if (n_lambda != n_rois + wd*ht) {
				matlabPtr->feval(u"error",
					0,
					std::vector<Array>({ factory.createScalar("lambda must contain (nrois + ht*wd) numbers") }));
				return nullptr;
			}

			seudo->lambda_.resize(n_lambda);
			auto &lambdaArg = inputs[InLambda];
			for (int i = 0; i < n_rois; i++) {
				seudo->lambda_[i] = lambdaArg[i];
			}
			// Matlab has X and Y dimensions transposed.
			for (int y = 0; y < ht; y++) {
				for (int x = 0; x < wd; x++) {
					seudo->lambda_[n_rois + y*wd + x] = lambdaArg[n_rois + x*ht + y];
				}
			}
		}

		// --- eps

		if (inputs[InEps].getType() != ArrayType::DOUBLE ||
			inputs[InEps].getNumberOfElements() != 1)
		{
			matlabPtr->feval(u"error",
				0,
				std::vector<Array>({ factory.createScalar("eps must be scalar double") }));
			return nullptr;
		}
		seudo->eps_ = inputs[InEps][0];

		// --- max_steps

		if (inputs[InMaxSteps].getType() != ArrayType::DOUBLE ||
			inputs[InMaxSteps].getNumberOfElements() != 1)
		{
			matlabPtr->feval(u"error",
				0,
				std::vector<Array>({ factory.createScalar("max_steps must be scalar double") }));
			return nullptr;
		}
		seudo->maxSteps_ = inputs[InMaxSteps][0];

		// --- l_mode

		if (inputs[InLMode].getType() != ArrayType::DOUBLE ||
			inputs[InLMode].getNumberOfElements() != 1)
		{
			matlabPtr->feval(u"error",
				0,
				std::vector<Array>({ factory.createScalar("l_mode must be scalar double") }));
			return nullptr;
		}
		switch ((int)inputs[InEps][0]) {
		case 0:
			seudo->multiGrad_ = false;
			seudo->fastBrake_ = false;
			break;
		case 1:
			seudo->multiGrad_ = true;
			seudo->fastBrake_ = false;
			break;
		case 2:
			seudo->multiGrad_ = false;
			seudo->fastBrake_ = true;
			break;
		case 3:
			seudo->multiGrad_ = true;
			seudo->fastBrake_ = true;
			break;
		default:
			matlabPtr->feval(u"error",
				0,
				std::vector<Array>({ factory.createScalar("l_mode must be 0-3") }));
			return nullptr;
			break;
		}

		// --- stop_mode

		if (inputs[InStopMode].getType() != ArrayType::DOUBLE ||
			inputs[InStopMode].getNumberOfElements() != 1)
		{
			matlabPtr->feval(u"error",
				0,
				std::vector<Array>({ factory.createScalar("stop_mode must be scalar double") }));
			return nullptr;
		}
		switch ((int)inputs[InStopMode][0]) {
		case 0:
			seudo->stopping_ = Fista::Run::StopEpsNorm2Rel;
			break;
		case 1:
			seudo->stopping_ = Fista::Run::StopEpsNorm2;
			break;
		case 2:
			seudo->stopping_ = Fista::Run::StopEpsEveryDimension;
			break;
		default:
			matlabPtr->feval(u"error",
				0,
				std::vector<Array>({ factory.createScalar("stop_mode must be 0, 1 or 2") }));
			return nullptr;
			break;
		}

		// --- verbose

		if (inputs[InVerbose].getType() != ArrayType::LOGICAL ||
			inputs[InVerbose].getNumberOfElements() != 1)
		{
			matlabPtr->feval(u"error",
				0,
				std::vector<Array>({ factory.createScalar("verbose must be scalar logical") }));
			return nullptr;
		}
		seudo->verbose_ = inputs[InVerbose][0];

		// --- parallel

		if (inputs[InParallel].getType() != ArrayType::DOUBLE ||
			inputs[InParallel].getNumberOfElements() != 1)
		{
			matlabPtr->feval(u"error",
				0,
				std::vector<Array>({ factory.createScalar("parallel must be scalar double") }));
			return nullptr;
		}
		seudo->setNumThreads(inputs[InParallel][0]);

		return seudo;
	}
};
