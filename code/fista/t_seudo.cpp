// Test of minimizer for SEUDO.

#include <stdio.h>
#include <math.h>
#include "fista.hpp"
#include "fista_gradient.hpp"
#include "seudo.hpp"
#include "draw.hpp"
#include "test.hpp"

// for convenience
inline double pow2(double x)
{
	return x * x;
}

// Makes a blob sort of like in Matlab code.
std::shared_ptr<Fista::Sprite> makeBlob(int radius)
{
	// This is a Gaussian curve, going from 1 in the center to 0.5 at the
	// radius, and 0 beyond the radius.
	int dim = 1 + radius*2;
	auto blob = std::make_shared<Fista::Sprite>(
		/*wd*/ dim, /*ht*/ dim, radius, radius);

	// A special case to avoid division by 0 in this case.
	if (radius == 0) {
		blob->img_[0] = 1.;
		return blob;
	}

	// Radius is also the standard deviation
	double radius2 = (radius * radius);

	for (int y = 0; y < dim; y++) {
		for (int x = 0; x < dim; x++) {
			double dist2 = pow2(x - radius) + pow2(y - radius);
			if (dist2 <= radius2) {
				blob->img_[y*dim + x] = exp( -dist2 / (2. * radius2) );
			}
		}
	}

	return blob;
}

// Bitmaps for the neuron cells
const char *rois8x8[] = {
	"A" 
	"....**.."
	"..*****."
	"********"
	".******."
	"********"
	".*****.."
	"********"
	"..****..",

	"b" 
	"........"
	".*******"
	"..******"
	"..*****."
	".****..."
	"***....."
	"**......"
	"*.......",

	"c" 
	"........"
	"..*....."
	".****..."
	".******."
	".****..."
	"..**...."
	"...*...."
	"........",

	NULL
};

// -------------------------------------------------------------------

class DummyDrawing : public Fista::Drawing
{
public:
	DummyDrawing()
		: Fista::Drawing(/*parallel*/ true)
	{ }

	// from Drawing
	virtual void draw(const Fista::Vector &input, Fista::Drawable &dest, Limits &limits)
	{ }
};

int testThreadInit(const char *tname)
{
	// Test that the threads get partitioned sensibly.
	int wd = 10;
	int ht = 8;
	int nthreads = 10;
	int ninputs = 25;

	auto drawing = std::make_shared<DummyDrawing>();
	Fista::DrawableImg img(wd, ht);
	Fista::Vector lambda(ninputs, 0.);

	auto gradient =
		std::make_shared<Fista::MultiPosDrawScaledGradient>(wd, ht, drawing, /*b*/ img.img_, lambda, nthreads);

	auto &limits1 = gradient->limitsPass1();
	if (limits1.size() != std::min(ht, nthreads)) {
		printf("%s: Limits1 size is wrong: size=%d, ht=%d, nthreads=%d\n",
			tname, (int)limits1.size(), ht, nthreads);
		return 1;
	}
	if (limits1.back().end_dest_y_ != ht) {
		printf("%s: Limits1 ends wrong: end=%d, ht=%d\n",
			tname, limits1.back().end_dest_y_, ht);
		return 1;
	}

	auto &limits2 = gradient->limitsPass2();
	if (limits2.size() != std::min(ninputs, nthreads)) {
		printf("%s: Limits2 size is wrong: size=%d, ninputs=%d, nthreads=%d\n",
			tname, (int)limits2.size(), ninputs, nthreads);
		return 1;
	}
	if (limits2.back().end_input_ != ninputs) {
		printf("%s: Limits2 ends wrong: end=%d, ninputs=%d\n",
			tname, limits2.back().end_input_, ninputs);
		return 1;
	}
	return 0;
}

// -------------------------------------------------------------------

int testShrinkage(const char *tname)
{
	// Test that shrinking and unshrinking by blob spacing works right.

	// Spacing on the blob grid.
	int blobSpacing = 4;

	// Number of ROIs (dummy ones).
	int nrois = 3;
	// Dimensions of the ROIs.
	const int roiwd = 8;
	const int roiht = 8;
	const char** bitmaps = rois8x8;
	// Dimensions of the image.
	const int shrunk_wd = 5;
	const int shrunk_ht = 4;
	const int wd = blobSpacing * (shrunk_wd - 1) + 1;
	const int ht = blobSpacing * (shrunk_ht - 1) + 1;
	const int pixels = wd * ht;

	auto blob = makeBlob(1);
	
	Fista::Seudo seudo(wd, ht);
	seudo.blob_ = blob;
	seudo.blobSpacing_ = blobSpacing;

	for (int i = 0; i < nrois; i++) {
		auto sprite = std::make_shared<Fista::Sprite>(
			/*wd*/ roiwd, /*ht*/ roiht, 0, 0);
		seudo.rois_.emplace_back(sprite);
	}

	seudo.setWeights(0.);
	seudo.setLambda(1.);
	for (int i = 0; i < seudo.weights_.size(); i++) {
		seudo.weights_[i] = i;
	}

	Fista::Vector w;
	seudo.shrinkToBlobSpacing(seudo.weights_, w);
	if (w.size() != nrois + shrunk_wd * shrunk_ht) {
		printf("%s: wrong shrunk size %d, should be %d\n", tname, (int)w.size(), nrois + shrunk_wd * shrunk_ht);
		return 1;
	}
	for (int i = 0; i < nrois; i++) {
		if (w[i] != (double)i) {
			printf("%s: wrong shrunk rois weight at %d; %f != %f\n", tname, i, w[i], (double)i);
			return 1;
		}
	}
	for (int y = 0; y < shrunk_ht; y++) {
		for (int x = 0; x < shrunk_wd; x++) {
			if (w[nrois + y * shrunk_wd + x] != seudo.weights_[nrois + y * blobSpacing * wd + x * blobSpacing]) {
				printf("%s: wrong shrunk weight at (%d, %d); %f != %f\n", tname, x, y,
					w[nrois + y * shrunk_wd + x], seudo.weights_[nrois + y * blobSpacing * wd + x * blobSpacing]);
				return 1;
			}
		}
	}

	seudo.unshrinkToBlobSpacing(w, seudo.weights_);

	for (int i = 0; i < nrois; i++) {
		if (w[i] != (double)i) {
			printf("%s: wrong unshrunk rois weight at %d; %f != %f\n", tname, i, w[i], (double)i);
			return 1;
		}
	}

	for (int y = 0; y < shrunk_ht; y++) {
		for (int x = 0; x < shrunk_wd; x++) {
			if (w[nrois + y * shrunk_wd + x] != seudo.weights_[nrois + y * blobSpacing * wd + x * blobSpacing]) {
				printf("%s: wrong unshrunk weight at (%d, %d); %f != %f\n", tname, x * blobSpacing, y * blobSpacing,
					w[nrois + y * shrunk_wd + x], seudo.weights_[nrois + y * blobSpacing * wd + x * blobSpacing]);
				return 1;
			}
		}
	}
	for (int y = 0; y < ht; y++) {
		for (int x = 0; x < wd; x++) {
			if (y % blobSpacing == 0 && x % blobSpacing == 0)
				continue;
			if (seudo.weights_[nrois + y * wd + x] != 0.) {
				printf("%s: unshrunk weight at (%d, %d) not erased; %f\n", tname, x, y,
					seudo.weights_[nrois + y * wd + x]);
				return 1;
			}
		}
	}
}

// -------------------------------------------------------------------

int testNoisy(const char *tname)
{
	// Recognize an image with overlapping shapes and noise.

#if 1 // {
	// Larger sizes for performance measurements.
	// Although the higher value of blob spacing makes it fast too.

	// Choose the drawing implementation. Custom drawing inlines more
	// and is much faster.
	bool useCustomDrawing = true;
	// Choose the L computation implementation.
	bool useMultiGrad = true;
	// Fast brake mode in FISTA.
	bool useFastBrake = true;
	// Number of threads.
	int nthreads = 4;
	// Spacing on the blob grid.
	int blobSpacing = 8;
	// Lambda.
	double lambda = 0.01;
	// Precision
	double eps = 0.0001;
	// Blob radius
	int bradius = 10;
	// Margin around the target cell.
	int margin = 50;

#else // } {
	// Small sizes for sanity tests

	// Choose the drawing implementation. Custom drawing inlines more
	// and is much faster.
	bool useCustomDrawing = true;
	// Choose the L computation implementation. MultiGrad seems to be faster.
	bool useMultiGrad = true;
	// Fast brake mode in FISTA.
	bool useFastBrake = true;
	// Number of threads.
	int nthreads = 10;
	// Spacing on the blob grid.
	int blobSpacing = 1;
	// Lambda.
	double lambda = 0.01;
	// Precision
	double eps = 0.0001;
	// Blob radius
	int bradius = 1;
	// Margin around the target cell.
	int margin = 5;

#endif // }

	// Dimensions of the ROIs.
	const int roiwd = 8;
	const int roiht = 8;
	const char** bitmaps = rois8x8;
	// Dimensions of the image.
	const int wd = roiwd + margin * 2;
	const int ht = roiht + margin * 2;
	const int pixels = wd * ht;

	Fista::Vector dummyVector(1);
	dummyVector[0] = 1.;

	auto blob = makeBlob(bradius);
	
	if (0) {
		// Look at what the blob looks like
		Fista::DrawableImg img(wd, ht);
		blob->draw(dummyVector, 0, img, bradius + 1, bradius + 1);
		printImg(img, 999.);
		img.clear();
	}

	Fista::Seudo seudo(wd, ht);
	seudo.verbose_ = true;
	seudo.blob_ = blob;
	seudo.eps_ = eps;
	seudo.setNumThreads(nthreads);
	seudo.multiGrad_ = useMultiGrad;
	seudo.useCustomDrawing_ = useCustomDrawing;
	seudo.fastBrake_ = useFastBrake;
	seudo.blobSpacing_ = blobSpacing;

	for (const char **mapptr = bitmaps; *mapptr != NULL; mapptr++) {
		auto sprite = std::make_shared<Fista::Sprite>(
			/*wd*/ roiwd, /*ht*/ roiht, 0, 0);

		char text[2] = ".";
		text[0] = mapptr[0][0];
		drawTextSimple(roiwd, roiht, sprite->img_, /*charwd*/ roiwd, /*charht*/ roiht, mapptr, 
			/*color*/ 1., text, /*startx*/ 0, /*starty*/ 0);
			
		sprite->cropWhitespace();
		// verbose && printf("after crop wd=%d ht=%d origin_x=%d origin_y=%d pixels=%d\n",
			// sprite->wd_, sprite->ht_, sprite->origin_x_, sprite->origin_y_, (int)sprite->img_.size());
		// verbose && printImg(*sprite, 999.);

		seudo.rois_.emplace_back(sprite);
	}

	// These are set after ROIs count is known.
	seudo.setWeights(0.);
	seudo.setLambda(lambda);

	// Manually adjust the position of the ROIs.
	seudo.rois_[0]->origin_x_ -= margin;
	seudo.rois_[0]->origin_y_ -= margin;
	seudo.rois_[1]->origin_x_ -= margin + 4;
	seudo.rois_[1]->origin_y_ -= margin - 4;
	seudo.rois_[2]->origin_x_ -= margin + 4;
	seudo.rois_[2]->origin_y_ -= margin;

	// Draw the sample to recognize.
	seudo.rois_[0]->draw(dummyVector, 0, seudo.image_, 0, 0, /*k*/ 0.3);
	seudo.rois_[1]->draw(dummyVector, 0, seudo.image_, 0, 0, /*k*/ 0.2);
	seudo.rois_[2]->draw(dummyVector, 0, seudo.image_, 0, 0, /*k*/ 0.1);
	verbose && printf("Image to recognize before noise:\n");
	verbose && printImg(seudo.image_, 999.);

	// Add randomness to the image,
	for (int i = 0; i < pixels; i++) {
		seudo.image_.img_[i] += ((i / seudo.image_.wd_ + i % seudo.image_.wd_) % 3) * 0.01;
	}
	verbose && printf("Image to recognize:\n");
	verbose && printImg(seudo.image_, 999.);

	seudo.compute();
	if (!seudo.error_.empty()) {
		printf("%s: Seudo error: %s\n", tname, seudo.error_.c_str());
		return 1;
	}
	
	if (verbose) {
		Fista::DrawableImg img(wd, ht);
		printf("Stopped after %d steps\n", seudo.stepsTaken_);
		for (int i = 0; i < seudo.rois_.size(); i++) {
			printf("%d: %f  ", i, seudo.weights_[i]);
			seudo.rois_[i]->draw(seudo.weights_, i, img, 0, 0);
		}
		printf("\n");
		double dist = seudo.distance();
		printf("Distance: %f, avg %f\n", dist, dist / sqrt(wd*ht));

		printf("Image of recognized ROIs:\n");
		printImg(img, 999.);

		printf("Recognized blobs:\n");
		img.img_.assign(seudo.weights_.begin() + seudo.rois_.size(), seudo.weights_.end()),
		printImg(img, 999.);
	}

	if (verbose && !seudo.log_.empty()) {
		printf("------------- Seudo log: ----------------------\n%s", seudo.log_.c_str());
	}

	if (fabs(seudo.weights_[0] - 0.3) > 0.01
	|| fabs(seudo.weights_[1] - 0.2) > 0.01
	|| fabs(seudo.weights_[2] - 0.1) > 0.01) {
		printf("%s: Too far from the expected result!\n", tname);
		return 1;
	}
	return 0;
}

// -------------------------------------------------------------------

int testThreads(const char *tname)
{
	// Fundamentally the same as testNoisy() but does the same computation
	// with and without multithreading, and checks that the result
	// matches.

	// Choose the drawing implementation. Custom drawing inlines more
	// and is much faster.
	bool useCustomDrawing = true;
	// Blob radius
	int bradius = 1;
	// Dimensions of the ROIs.
	const int roiwd = 8;
	const int roiht = 8;
	const char** bitmaps = rois8x8;
	// Margin around the target cell.
	int margin = 5;
	// Dimensions of the image.
	const int wd = roiwd + margin * 2;
	const int ht = roiht + margin * 2;
	const int pixels = wd * ht;
	// Lambda.
	double lambda = 0.01;

	Fista::Vector dummyVector(1);
	dummyVector[0] = 1.;

	auto blob = makeBlob(bradius);
	
	if (0) {
		// Look at what the blob looks like
		Fista::DrawableImg img(wd, ht);
		blob->draw(dummyVector, 0, img, bradius + 1, bradius + 1);
		printImg(img, 999.);
		img.clear();
	}

	// --- Single-threaded
	Fista::Seudo seudo_s(wd, ht);
	seudo_s.verbose_ = true;
	seudo_s.blob_ = blob;

	for (const char **mapptr = bitmaps; *mapptr != NULL; mapptr++) {
		auto sprite = std::make_shared<Fista::Sprite>(
			/*wd*/ roiwd, /*ht*/ roiht, 0, 0);

		char text[2] = ".";
		text[0] = mapptr[0][0];
		drawTextSimple(roiwd, roiht, sprite->img_, /*charwd*/ roiwd, /*charht*/ roiht, mapptr, 
			/*color*/ 1., text, /*startx*/ 0, /*starty*/ 0);
			
		sprite->cropWhitespace();
		// verbose && printf("after crop wd=%d ht=%d origin_x=%d origin_y=%d pixels=%d\n",
			// sprite->wd_, sprite->ht_, sprite->origin_x_, sprite->origin_y_, (int)sprite->img_.size());
		// verbose && printImg(*sprite, 999.);

		seudo_s.rois_.emplace_back(sprite);
	}

	// seudo_s.eps_ = 0.00001;

	// Manually adjust the position of the ROIs.
	seudo_s.rois_[0]->origin_x_ -= margin;
	seudo_s.rois_[0]->origin_y_ -= margin;
	seudo_s.rois_[1]->origin_x_ -= margin + 4;
	seudo_s.rois_[1]->origin_y_ -= margin - 4;
	seudo_s.rois_[2]->origin_x_ -= margin + 4;
	seudo_s.rois_[2]->origin_y_ -= margin;

	// Draw the sample to recognize.
	seudo_s.rois_[0]->draw(dummyVector, 0, seudo_s.image_, 0, 0, /*k*/ 0.3);
	seudo_s.rois_[1]->draw(dummyVector, 0, seudo_s.image_, 0, 0, /*k*/ 0.2);
	seudo_s.rois_[2]->draw(dummyVector, 0, seudo_s.image_, 0, 0, /*k*/ 0.1);
	verbose && printf("Image to recognize before noise:\n");
	verbose && printImg(seudo_s.image_, 999.);

	// Add randomness to the image,
	for (int i = 0; i < pixels; i++) {
		seudo_s.image_.img_[i] += ((i / seudo_s.image_.wd_ + i % seudo_s.image_.wd_) % 3) * 0.01;
	}
	verbose && printf("Image to recognize:\n");
	verbose && printImg(seudo_s.image_, 999.);

	seudo_s.setWeights(0.);
	seudo_s.setLambda(lambda);
	seudo_s.setNumThreads(1);
	seudo_s.useCustomDrawing_ = useCustomDrawing;

	seudo_s.compute();
	if (!seudo_s.error_.empty()) {
		printf("%s: Single-threaded Seudo error: %s\n", tname, seudo_s.error_.c_str());
		return 1;
	}
	
	if (verbose) {
		Fista::DrawableImg img(wd, ht);
		printf("Single-threaded Stopped after %d steps\n", seudo_s.stepsTaken_);
		for (int i = 0; i < seudo_s.rois_.size(); i++) {
			printf("%d: %f  ", i, seudo_s.weights_[i]);
			seudo_s.rois_[i]->draw(seudo_s.weights_, i, img, 0, 0);
		}
		printf("\n");

		printf("Image of recognized ROIs:\n");
		printImg(img, 999.);

		printf("Recognized blobs:\n");
		img.img_.assign(seudo_s.weights_.begin() + seudo_s.rois_.size(), seudo_s.weights_.end()),
		printImg(img, 999.);
	}

	if (verbose && !seudo_s.log_.empty()) {
		printf("------------- Single-threaded Seudo log: ----------------------\n%s", seudo_s.log_.c_str());
	}

	// --- Multi-threaded
	Fista::Seudo seudo_m(wd, ht);
	seudo_m.verbose_ = true;
	seudo_m.blob_ = blob;
	seudo_m.rois_ = seudo_s.rois_;
	seudo_m.image_ = seudo_s.image_;

	seudo_m.setWeights(0.);
	seudo_m.setLambda(lambda);
	seudo_m.setNumThreads(10);
	seudo_m.useCustomDrawing_ = useCustomDrawing;

	seudo_m.compute();
	if (!seudo_m.error_.empty()) {
		printf("%s: Multi-threaded Seudo error: %s\n", tname, seudo_m.error_.c_str());
		return 1;
	}
	
	if (verbose) {
		Fista::DrawableImg img(wd, ht);
		printf("Multi-threaded Stopped after %d steps\n", seudo_m.stepsTaken_);
		for (int i = 0; i < seudo_m.rois_.size(); i++) {
			printf("%d: %f  ", i, seudo_m.weights_[i]);
			seudo_m.rois_[i]->draw(seudo_m.weights_, i, img, 0, 0);
		}
		printf("\n");

		printf("Image of recognized ROIs:\n");
		printImg(img, 999.);

		printf("Recognized blobs:\n");
		img.img_.assign(seudo_m.weights_.begin() + seudo_m.rois_.size(), seudo_m.weights_.end()),
		printImg(img, 999.);
	}

	if (verbose && !seudo_m.log_.empty()) {
		printf("------------- Multi-threaded Seudo log: ----------------------\n%s", seudo_m.log_.c_str());
	}

	for (int i = 0; i < seudo_s.weights_.size(); i++) {
		if (seudo_s.weights_[i] != seudo_m.weights_[i]) {
			printf("%s: Result mismatch at index %d: single=%f multi=%f\n",
				tname, i, seudo_s.weights_[i], seudo_m.weights_[i]);
			return 1;
		}
	}

	return 0;
}

// -------------------------------------------------------------------

int testBlobSpacing(const char *tname)
{
	// Same as testNoisy but always with blobSpacing > 1.

	// Small sizes for sanity tests

	// Choose the drawing implementation. Custom drawing inlines more
	// and is much faster.
	bool useCustomDrawing = true;
	// Choose the L computation implementation. MultiGrad seems to be faster.
	bool useMultiGrad = true;
	// Fast brake mode in FISTA.
	bool useFastBrake = true;
	// Number of threads.
	int nthreads = 10;
	// Spacing on the blob grid.
	int blobSpacing = 3;
	// Lambda.
	double lambda = 0.01;
	// Precision
	double eps = 0.0001;
	// Blob radius
	int bradius = 1;
	// Margin around the target cell.
	int margin = 5;

	// Dimensions of the ROIs.
	const int roiwd = 8;
	const int roiht = 8;
	const char** bitmaps = rois8x8;
	// Dimensions of the image.
	const int wd = roiwd + margin * 2;
	const int ht = roiht + margin * 2;
	const int pixels = wd * ht;

	Fista::Vector dummyVector(1);
	dummyVector[0] = 1.;

	auto blob = makeBlob(bradius);
	
	if (0) {
		// Look at what the blob looks like
		Fista::DrawableImg img(wd, ht);
		blob->draw(dummyVector, 0, img, bradius + 1, bradius + 1);
		printImg(img, 999.);
		img.clear();
	}

	Fista::Seudo seudo(wd, ht);
	seudo.verbose_ = true;
	seudo.blob_ = blob;
	seudo.eps_ = eps;
	seudo.setNumThreads(nthreads);
	seudo.multiGrad_ = useMultiGrad;
	seudo.useCustomDrawing_ = useCustomDrawing;
	seudo.fastBrake_ = useFastBrake;
	seudo.blobSpacing_ = blobSpacing;

	for (const char **mapptr = bitmaps; *mapptr != NULL; mapptr++) {
		auto sprite = std::make_shared<Fista::Sprite>(
			/*wd*/ roiwd, /*ht*/ roiht, 0, 0);

		char text[2] = ".";
		text[0] = mapptr[0][0];
		drawTextSimple(roiwd, roiht, sprite->img_, /*charwd*/ roiwd, /*charht*/ roiht, mapptr, 
			/*color*/ 1., text, /*startx*/ 0, /*starty*/ 0);
			
		sprite->cropWhitespace();
		// verbose && printf("after crop wd=%d ht=%d origin_x=%d origin_y=%d pixels=%d\n",
			// sprite->wd_, sprite->ht_, sprite->origin_x_, sprite->origin_y_, (int)sprite->img_.size());
		// verbose && printImg(*sprite, 999.);

		seudo.rois_.emplace_back(sprite);
	}

	// These are set after ROIs count is known.
	seudo.setWeights(0.);
	seudo.setLambda(lambda);

	// Manually adjust the position of the ROIs.
	seudo.rois_[0]->origin_x_ -= margin;
	seudo.rois_[0]->origin_y_ -= margin;
	seudo.rois_[1]->origin_x_ -= margin + 4;
	seudo.rois_[1]->origin_y_ -= margin - 4;
	seudo.rois_[2]->origin_x_ -= margin + 4;
	seudo.rois_[2]->origin_y_ -= margin;

	// Draw the sample to recognize.
	seudo.rois_[0]->draw(dummyVector, 0, seudo.image_, 0, 0, /*k*/ 0.3);
	seudo.rois_[1]->draw(dummyVector, 0, seudo.image_, 0, 0, /*k*/ 0.2);
	seudo.rois_[2]->draw(dummyVector, 0, seudo.image_, 0, 0, /*k*/ 0.1);
	verbose && printf("Image to recognize before noise:\n");
	verbose && printImg(seudo.image_, 999.);

	// Add randomness to the image,
	for (int i = 0; i < pixels; i++) {
		seudo.image_.img_[i] += ((i / seudo.image_.wd_ + i % seudo.image_.wd_) % 3) * 0.01;
	}
	verbose && printf("Image to recognize:\n");
	verbose && printImg(seudo.image_, 999.);

	seudo.compute();
	if (!seudo.error_.empty()) {
		printf("%s: Seudo error: %s\n", tname, seudo.error_.c_str());
		return 1;
	}
	
	if (verbose) {
		Fista::DrawableImg img(wd, ht);
		printf("Stopped after %d steps\n", seudo.stepsTaken_);
		for (int i = 0; i < seudo.rois_.size(); i++) {
			printf("%d: %f  ", i, seudo.weights_[i]);
			seudo.rois_[i]->draw(seudo.weights_, i, img, 0, 0);
		}
		printf("\n");
		double dist = seudo.distance();
		printf("Distance: %f, avg %f\n", dist, dist / sqrt(wd*ht));

		printf("Image of recognized ROIs:\n");
		printImg(img, 999.);

		printf("Recognized blobs:\n");
		img.img_.assign(seudo.weights_.begin() + seudo.rois_.size(), seudo.weights_.end()),
		printImg(img, 999.);
	}

	if (verbose && !seudo.log_.empty()) {
		printf("------------- Seudo log: ----------------------\n%s", seudo.log_.c_str());
	}

	if (fabs(seudo.weights_[0] - 0.3) > 0.01
	|| fabs(seudo.weights_[1] - 0.2) > 0.01
	|| fabs(seudo.weights_[2] - 0.1) > 0.01) {
		printf("%s: Too far from the expected result!\n", tname);
		return 1;
	}
	return 0;
}

// -------------------------------------------------------------------

int
main(int argc, char **argv)
{
	TestDriver td(argc, argv);
	int result = 0;

	RUN_TEST(result, td, testThreadInit);
	RUN_TEST(result, td, testNoisy);
	RUN_TEST(result, td, testThreads);
	RUN_TEST(result, td, testBlobSpacing);

	return result;
}
