// Test of FISTA code.

#include <stdio.h>
#include <math.h>
#include "fista_v1.hpp"
#include "fista_v1_minimizer.hpp"
#include "draw.hpp"
#include "test.hpp"

int
testDiagonal(const char *tname)
{
	// A basic example matching the one in the simpler tests (t_fista and t_vsq_fista_v1).
	Fista::Vector k(4);
	// row 0
	k[0] = 1.;
	k[1] = 0.;
	// row 1
	k[2] = 0.;
	k[3] = 1.;
	Fista::Vector b(2);
	b[0] = 10.;
	b[1] = 10.;
	Fista::Vector lambda(2);
	lambda[0] = 1.;
	lambda[1] = 1.;

	Fista::PosMatSquareMinimizer pl(k, b, lambda, 0.1);
	
	Fista::Vector x(2);
	x[0] = 2.;
	x[1] = 1.;
	int n = 20;

	// This is a copy of repeatedSteps() with printing.
	double t = 1.;
	bool stop;
	Fista::Vector x_next(x.size());
	stop = pl.compute(x, x_next, 0); // computes x_1 from x_0
	verbose && printf("step %2d: (%f, %f) stop=%d\n", 1, x_next[0], x_next[1], stop);

	Fista::Vector y;
	// the loop goes from x_2 and up
	for (int i = 1; i < n; i++) {
		stop = Fista::oneStep(pl, t, x, x_next, y, i, 0);
		verbose && printf("step %2d: (%f, %f) stop=%d\n", i + 1, x_next[0], x_next[1], stop);
	}

	verbose && printf("after %2d steps: (%f, %f) stop=%d\n", n, x_next[0], x_next[1], stop);

	if (fabs(x_next[0] - 9.5) > 0.1
	|| fabs(x_next[1] - 9.5) > 0.1) {
		printf("%s: Too far from the expected result!\n", tname);
		return 1;
	}
	if (!stop) {
		printf("%s: The stop condition not met!\n", tname);
		return 1;
	}
	return 0;
}

// Like Fista::PosMatSquareMinimizer but with 0 >= x <= 256.
class PixelMinimizer : public Fista::PosMatSquareMinimizer
{
public:
	// See the meaning of arguments in the comment to the class.
	// The sizes of all vectors must match as described.
	// @param k - a flattened matrix K*N, structured as N "rows"
	//		of K entries one after another:
	//            [ ... K ... ]  ...  [ ... K ... ] 
	// @param b - target values, of size N
	// @param lambda - LASSO coefficients, of size K, which matches
	// 		the size of vector X
	// @param eps - epsilon for detecting the stopping point, the
	//      stop will be triggered when norm2(grad_fg(x)) <= eps^2
	PixelMinimizer(
		const Fista::Vector &k,
		const Fista::Vector &b,
		const Fista::Vector &lambda,
		double eps)
		: Fista::PosMatSquareMinimizer(k, b, lambda, eps)
	{
		upperBounded_ = true;
		upperBound_ = 256.;
	}
};

int testDeblur(const char *tname)
{
	// Try to restore a blurred image.

	// Dimensions of the image. Small to run fast.
	const int wd = 5;
	const int ht = 5;
	const int pixels = wd * ht;
	const int bradius = 1; // blur radius
	Fista::Vector lambda(pixels, 0.); // 0 lambda works best
	const int n = 200; // number of steps
	const int restartEvery = 1000; // after how many steps restart the parameter t
	const int printEvery = 1; // after how many steps print an image

	Fista::Vector img(pixels);

	// Draw some lines.
	for (int i = 0; i < wd && i < ht; i++) {
		// A diagonal cross.
		img[i*wd + i] = 256.;
		img[i*wd + wd - 1 - i] = 256.;
	}

	verbose && printf("Original:\n");
	verbose && printImg(wd, ht, img);

	// Blurring filter matrix.
	// rows are destination pixels, columns source pixels
	Fista::Vector bfilter(pixels*pixels);
	for (int i = 0; i < wd; i++) {
		for (int j = 0; j < ht; j++) {
			int x0 = i - bradius;
			if (x0 < 0) // crop on edges
				x0 = 0;
			int x1 = i + bradius + 1;
			if (x1 > wd) // crop on edges
				x1 = wd;

			int y0 = j - bradius;
			if (y0 < 0) // crop on edges
				y0 = 0;
			int y1 = j + bradius + 1;
			if (y1 > ht) // crop on edges
				y1 = ht;

			double avg = 1. / ((x1 - x0) * (y1 - y0)); // averaging by num of pixels in a blurring region
			// double avg = 1. / ((bradius + 1) * (bradius + 1));
			
			for (int x = x0; x < x1; x++) {
				for (int y = y0; y < y1; y++) {
					bfilter[(j*wd + i)*pixels + (y*wd + x)] = avg;
				}
			}
		}
	}

	// Apply the blurring
	Fista::Vector bimg(pixels);
	for (int i = 0; i < wd; i++) {
		for (int j = 0; j < ht; j++) {
			for (int x = 0; x < wd; x++) {
				for (int y = 0; y < ht; y++) {
					bimg[j*wd + i] += bfilter[(j*wd + i)*pixels + (y*wd + x)] * img[y*wd + x];
				}
			}
		}
	}
	verbose && printf("Blurred:\n");
	verbose && printImg(wd, ht, bimg);
	
	// Do the deblurring.
	PixelMinimizer pl(/*k*/ bfilter, /*b*/ bimg, lambda, /*eps*/ 0.01);
	// pl.verbose_ = verbose;

	Fista::Vector deimg(pixels);

	// This is a copy of repeatedSteps() with printing.
	double t = 1.;
	bool stop;
	Fista::Vector deimg_next(pixels);
	stop = pl.compute(deimg, deimg_next, 0); // computes x_1 from x_0
	verbose && printf("step %2d: stop=%d\n", 1, stop);
	verbose && printImg(wd, ht, deimg_next);

	Fista::Vector y;
	for (int i = 1; i < n; i++) {
		// try to restart t every so often (100 seems much too often)
		if (i % restartEvery == 0) { t = 1.; }

		stop = Fista::oneStep(pl, t, deimg, deimg_next, y, i, /*nrestart*/ 0);
		if (i % printEvery == 0) {
			verbose && printf("step %2d: stop=%d\n", i + 1, stop);
			verbose && printImg(wd, ht, deimg_next);
		}
	}

	if (!stop) {
		printf("%s: The stop condition not met!\n", tname);
		return 1;
	}
	return 0;
}

int
main(int argc, char **argv)
{
	TestDriver td(argc, argv);
	int result = 0;

	RUN_TEST(result, td, testDiagonal);
	RUN_TEST(result, td, testDeblur);

	return result;
}
