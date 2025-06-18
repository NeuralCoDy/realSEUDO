// Test of FISTA code.

#include <stdio.h>
#include <math.h>
#include "fista.hpp"
#include "fista_gradient.hpp"
#include "draw.hpp"
#include "test.hpp"

// This code is a derivation of the one from t_msq_fista.cpp, a version optimized
// for large sparse matrices by allowing to "draw" the transformation instead of
// creating the matrix in memory. Other than that, all the same logic applies,
// and the code here builds on it.

int testDeblur(const char *tname)
{
	// Try to restore a blurred image. Same as testDeblur() in t_msq_fista but
	// instead of creating a transformation matrix, "draws" the pixels to
	// compute the gradient.

	// Dimensions of the image. Small to run fast.
	const int wd = 5;
	const int ht = 5;
	const int pixels = wd * ht;
	const int bradius = 1; // blur radius
	// 0 lambda provides the highest quality of deblurring, actually
	// reducing the spurious pixels with 1 or 2 in them (because lambda
	// pulls the nearby high pixels down)
	Fista::Vector lambda(pixels, 0.);
	const int n = 200; // number of steps
	// const int restartEvery = 1000; // after how many steps restart the parameter t
	const int printEvery = 1; // after how many steps print an image

	Fista::Vector img(pixels); // the original image
	Fista::DrawableImg bimg(wd, ht); // blurred image will go here

	// Draw some lines.
	for (int i = 0; i < wd && i < ht; i++) {
		// A diagonal cross.
		img[i*wd + i] = 256.;
		img[i*wd + wd - 1 - i] = 256.;
	}

	verbose && printf("Original:\n");
	verbose && printImg(wd, ht, img);

	// Apply the blurring
	std::shared_ptr<BlurDrawing> blurrer = std::make_shared<BlurDrawing>(wd, ht, bradius);

	blurrer->drawSimple(img, bimg);
	verbose && printf("Blurred:\n");
	verbose && printImg(wd, ht, bimg.img_);
	
	// Do the deblurring.
	// This eps is stricter than in the tests of v1, because it includes algorithm's step too.
	Fista::Run run(std::make_shared<Fista::PosDrawScaledGradient>(wd, ht, blurrer, /*b*/ bimg.img_, lambda),
		std::make_shared<Fista::RangeLimiter>(0., 256.), lambda.size(), /*diffEps*/ 0.01);

	// This is a copy of repeatedSteps() with printing.
	bool stop;
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

int testDeblurRecomputeL(const char *tname)
{
	// Try to restore a blurred image. Same as testDeblur() but recomputes
	// L on every step.

	// Dimensions of the image. Small to run fast.
	const int wd = 5;
	const int ht = 5;
	const int pixels = wd * ht;
	const int bradius = 1; // blur radius
	// 0 lambda provides the highest quality of deblurring, actually
	// reducing the spurious pixels with 1 or 2 in them (because lambda
	// pulls the nearby high pixels down)
	Fista::Vector lambda(pixels, 0.);
	const int n = 200; // number of steps
	// const int restartEvery = 1000; // after how many steps restart the parameter t
	const int printEvery = 1; // after how many steps print an image

	Fista::Vector img(pixels); // the original image
	Fista::DrawableImg bimg(wd, ht); // blurred image will go here

	// Draw some lines.
	for (int i = 0; i < wd && i < ht; i++) {
		// A diagonal cross.
		img[i*wd + i] = 256.;
		img[i*wd + wd - 1 - i] = 256.;
	}

	verbose && printf("Original:\n");
	verbose && printImg(wd, ht, img);

	// Apply the blurring
	std::shared_ptr<BlurDrawing> blurrer = std::make_shared<BlurDrawing>(wd, ht, bradius);

	blurrer->drawSimple(img, bimg);
	verbose && printf("Blurred:\n");
	verbose && printImg(wd, ht, bimg.img_);
	
	// Do the deblurring.
	// This eps is stricter than in the tests of v1, because it includes algorithm's step too.
	Fista::Run run(std::make_shared<Fista::PosDrawScaledGradient>(wd, ht, blurrer, /*b*/ bimg.img_, lambda, /*reinit_L_every*/ 1),
		std::make_shared<Fista::RangeLimiter>(0., 256.), lambda.size(), /*diffEps*/ 0.01);

	// This is a copy of repeatedSteps() with printing.
	bool stop;
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

int testDeblurMulti(const char *tname)
{
	// Try to restore a blurred image. Same as testDeblur() but uses the
	// multi-dimensional L.

	// Dimensions of the image. Small to run fast.
	const int wd = 5;
	const int ht = 5;
	const int pixels = wd * ht;
	const int bradius = 1; // blur radius
	// 0 lambda provides the highest quality of deblurring, actually
	// reducing the spurious pixels with 1 or 2 in them (because lambda
	// pulls the nearby high pixels down)
	Fista::Vector lambda(pixels, 0.);
	const int n = 200; // number of steps
	// const int restartEvery = 1000; // after how many steps restart the parameter t
	const int printEvery = 1; // after how many steps print an image

	Fista::Vector img(pixels); // the original image
	Fista::DrawableImg bimg(wd, ht); // blurred image will go here

	// Draw some lines.
	for (int i = 0; i < wd && i < ht; i++) {
		// A diagonal cross.
		img[i*wd + i] = 256.;
		img[i*wd + wd - 1 - i] = 256.;
	}

	verbose && printf("Original:\n");
	verbose && printImg(wd, ht, img);

	// Apply the blurring
	std::shared_ptr<BlurDrawing> blurrer = std::make_shared<BlurDrawing>(wd, ht, bradius);

	blurrer->drawSimple(img, bimg);
	verbose && printf("Blurred:\n");
	verbose && printImg(wd, ht, bimg.img_);
	
	// Do the deblurring.
	// This eps is stricter than in the tests of v1, because it includes algorithm's step too.
	Fista::Run run(std::make_shared<Fista::MultiPosDrawScaledGradient>(wd, ht, blurrer, /*b*/ bimg.img_, lambda),
		std::make_shared<Fista::RangeLimiter>(0., 256.), lambda.size(), /*diffEps*/ 0.01);

	// This is a copy of repeatedSteps() with printing.
	bool stop;
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

	RUN_TEST(result, td, testDeblur);
	RUN_TEST(result, td, testDeblurRecomputeL);
	RUN_TEST(result, td, testDeblurMulti);

	return result;
}
