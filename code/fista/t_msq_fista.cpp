// Test of FISTA code.

#include <stdio.h>
#include <math.h>
#include "fista.hpp"
#include "fista_gradient.hpp"
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

	Fista::Vector init_x(2);
	init_x[0] = 2.;
	init_x[1] = 1.;
	int n = 20;

	// This eps is stricter than in the tests of v1, because it includes algorithm's step too.
	Fista::Run run(std::make_shared<Fista::PosMatScaledGradient>(k, b, lambda),
		std::make_shared<Fista::PosLimiter>(), init_x, /*diffEps*/ 0.1);
	
	// This is a copy of repeatedSteps() with printing.
	bool stop;
	// the loop goes from x_2 and up
	for (int i = 0; i < n; i++) {
		stop = run.oneStep();
		verbose && printf("step %2d: (%f, %f) stop=%d\n", run.step_, run.x_[0], run.x_[1], stop);
	}

	if (fabs(run.x_[0] - 9.5) > 0.1
	|| fabs(run.x_[1] - 9.5) > 0.1) {
		printf("%s: Too far from the expected result!\n", tname);
		return 1;
	}
	if (!stop) {
		printf("%s: The stop condition not met!\n", tname);
		return 1;
	}
	return 0;
}

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
	// const int restartEvery = 1000; // after how many steps restart the parameter t
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
	// This eps is stricter than in the tests of v1, because it includes algorithm's step too.
	Fista::Run run(std::make_shared<Fista::PosMatScaledGradient>(/*k*/ bfilter, /*b*/ bimg, lambda),
		std::make_shared<Fista::RangeLimiter>(0., 256.), lambda.size(), /*diffEps*/ 0.01);
	
	// This is a copy of repeatedSteps() with printing.
	bool stop;
	// the loop goes from x_2 and up
	for (int i = 0; i < n; i++) {
		stop = run.oneStep();
		if (run.step_ % printEvery == 0) {
			verbose && printf("step %2d: stop=%d\n", run.step_, stop);
			verbose && printImg(wd, ht, run.x_);
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
