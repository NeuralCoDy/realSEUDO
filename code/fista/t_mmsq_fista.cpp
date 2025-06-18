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
	// The same test as in t_fista, t_vsq_fista and t_msq_fista, here the vector
	// gets represented as a diagonal of the matrix. But the function is slightly
	// different, with different k_00 and k_11, which makes the classic convergence
	// much slower. It also causes the much smaller steps that make the algorithm
	// think that it's time to stop while it's a rather long way away from the minimim.

	Fista::Vector k(4);
	// row 0
	k[0] = 1.;
	k[1] = 0.;
	// row 1
	k[2] = 0.;
	k[3] = 4.;
	Fista::Vector b(2);
	b[0] = 10.;
	b[1] = 10.;
	Fista::Vector lambda(2);
	lambda[0] = 1.;
	lambda[1] = 1.;

	Fista::Vector init_x(2);
	init_x[0] = 2.;
	init_x[1] = 1.;
	int n = 40;

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
	|| fabs(run.x_[1] - 2.46) > 0.1) {
		printf("%s: Too far from the expected result!\n", tname);
		return 1;
	}
	if (!stop) {
		printf("%s: The stop condition not met!\n", tname);
		return 1;
	}
	return 0;
}

int
testDiagonalMulti(const char *tname)
{
	// The same test as testDiagonal() but now using individual L
	// per dimension makes the convergence much faster, even in 1 step
	// for this kind of matrix.

	Fista::Vector k(4);
	// row 0
	k[0] = 1.;
	k[1] = 0.;
	// row 1
	k[2] = 0.;
	k[3] = 4.;
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
	Fista::Run run(std::make_shared<Fista::MultiPosMatScaledGradient>(k, b, lambda),
		std::make_shared<Fista::PosLimiter>(), init_x, /*diffEps*/ 0.1);
	
	// This is a copy of repeatedSteps() with printing.
	bool stop;
	// the loop goes from x_2 and up
	for (int i = 0; i < n; i++) {
		stop = run.oneStep();
		verbose && printf("step %2d: (%f, %f) stop=%d\n", run.step_, run.x_[0], run.x_[1], stop);
	}

	if (fabs(run.x_[0] - 9.5) > 0.1
	|| fabs(run.x_[1] - 2.46) > 0.1) {
		printf("%s: Too far from the expected result!\n", tname);
		return 1;
	}
	if (!stop) {
		printf("%s: The stop condition not met!\n", tname);
		return 1;
	}
	return 0;
}

int
testDiagonalMulti2(const char *tname)
{
	// The same test as testDiagonalMulti() but using the
	// _other_ diagonals of the matrix, symmetric. It also gets solved in 1 step.

	Fista::Vector k(4);
	// row 0
	k[0] = 0.;
	k[1] = 1.;
	// row 1
	k[2] = 4.;
	k[3] = 0.;
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
	Fista::Run run(std::make_shared<Fista::MultiPosMatScaledGradient>(k, b, lambda),
		std::make_shared<Fista::PosLimiter>(), init_x, /*diffEps*/ 0.1);
	
	// This is a copy of repeatedSteps() with printing.
	bool stop;
	// the loop goes from x_2 and up
	for (int i = 0; i < n; i++) {
		stop = run.oneStep();
		verbose && printf("step %2d: (%f, %f) stop=%d\n", run.step_, run.x_[0], run.x_[1], stop);
	}

	if (fabs(run.x_[0] - 2.46) > 0.1
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

int
testDiagonalMulti3(const char *tname)
{
	// The same test as testDiagonalMulti() but using the
	// _other_ diagonals of the matrix, asymmetric.

	Fista::Vector k(4);
	// row 0
	k[0] = 0.;
	k[1] = 1.;
	// row 1
	k[2] = 4.;
	k[3] = 0.;
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
	Fista::Run run(std::make_shared<Fista::MultiPosMatScaledGradient>(k, b, lambda),
		std::make_shared<Fista::PosLimiter>(), init_x, /*diffEps*/ 0.1);
	
	// This is a copy of repeatedSteps() with printing.
	bool stop;
	// the loop goes from x_2 and up
	for (int i = 0; i < n; i++) {
		stop = run.oneStep();
		verbose && printf("step %2d: (%f, %f) stop=%d\n", run.step_, run.x_[0], run.x_[1], stop);
	}

	if (fabs(run.x_[0] - 2.46) > 0.1
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

int
testAllMulti(const char *tname)
{
	// Fill all 4 elements with different values on different diagonals.

	Fista::Vector k(4);
	// row 0
	k[0] = 0.1;
	k[1] = 4.;
	// row 1
	k[2] = 2.;
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
	Fista::Run run(std::make_shared<Fista::MultiPosMatScaledGradient>(k, b, lambda),
		std::make_shared<Fista::PosLimiter>(), init_x, /*diffEps*/ 0.01);
	
	// This is a copy of repeatedSteps() with printing.
	bool stop;
	// the loop goes from x_2 and up
	for (int i = 0; i < n; i++) {
		stop = run.oneStep();
		verbose && printf("step %2d: (%f, %f) stop=%d\n", run.step_, run.x_[0], run.x_[1], stop);
	}

	if (fabs(run.x_[0] - 3.7) > 0.1
	|| fabs(run.x_[1] - 2.4) > 0.1) {
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
	// Try to restore a blurred image. With milti-L, it converges faster
	// than in t_msq_fista. The code is the same, differing only in the
	// subclass of ScaledGradient.

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
	
	int stopstep1 = -1;
	{
		// Do the deblurring.
		// This eps is stricter than in the tests of v1, because it includes algorithm's step too.
		Fista::Run run(std::make_shared<Fista::MultiPosMatScaledGradient>(/*k*/ bfilter, /*b*/ bimg, lambda),
			std::make_shared<Fista::RangeLimiter>(0., 256.), lambda.size(), /*diffEps*/ 0.01);
		run.fastBrake_ = false; // true erases the difference between stopping criteria
		
		// This is a copy of repeatedSteps() with printing.
		bool stop;
		// the loop goes from x_2 and up
		for (int i = 0; i < n; i++) {
			stop = run.oneStep();
			if (stop && stopstep1 < 0)
				stopstep1 = run.step_;
			if (run.step_ % printEvery == 0) {
				verbose && printf("step %2d: stop=%d\n", run.step_, stop);
				verbose && printImg(wd, ht, run.x_);
			}
		}

		if (!stop) {
			printf("%s: The stop condition not met!\n", tname);
			return 1;
		}
	}
	verbose && printf("stop step %d\n", stopstep1);

	int stopstep2;
	{
		// Do the same thing with a different stopping criteria.
		Fista::Run run(std::make_shared<Fista::MultiPosMatScaledGradient>(/*k*/ bfilter, /*b*/ bimg, lambda),
			std::make_shared<Fista::RangeLimiter>(0., 256.), lambda.size(), /*diffEps*/ 0.01);
		run.stopping_ = Fista::Run::StopEpsNorm2;
		run.fastBrake_ = false; // true erases the difference between stopping criteria
		stopstep2 = run.repeatedSteps(n);

		verbose && printf("StopEpsNorm2 stop step %d\n", stopstep2);

		if (stopstep2 > n) {
			printf("%s: The StopEpsNorm2 stop condition not met!\n", tname);
			return 1;
		}
		
		// This tests that the StopEpsNorm2 condition is more strict.
		if (stopstep1 >= stopstep2) {
			printf("%s: The StopEpsNorm2 stop condition is not stricter than StopEpsEveryDimension! %d vs %d\n",
				tname, stopstep2, stopstep1);
			return 1;
		}
	}

	return 0;
}

int
main(int argc, char **argv)
{
	TestDriver td(argc, argv);
	int result = 0;

	RUN_TEST(result, td, testDiagonal);
	RUN_TEST(result, td, testDiagonalMulti);
	RUN_TEST(result, td, testDiagonalMulti2);
	RUN_TEST(result, td, testDiagonalMulti3);
	RUN_TEST(result, td, testAllMulti);
	RUN_TEST(result, td, testDeblur);

	return result;
}
