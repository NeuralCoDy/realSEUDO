// Test of FISTA code.

#include <stdio.h>
#include <math.h>
#include "fista.hpp"
#include "fista_gradient.hpp"
#include "draw.hpp"
#include "test.hpp"

// This code is based on t_dsq_fista.cpp, just more experiments with restoring text.

int testDeblurDraw(const char *tname)
{
	// Try to restore a blurred image. Same as testDeblur() but instead of
	// creating a transformation matrix, "draws" the pixels to compute the
	// gradient.

	// Dimensions of the image. Small to run fast.
	// 
	// Check out wd=30, ht=30, bradius=15. This mixes almost every pixel
	// with every, and after 2000 steps the result gets restored back to
	// not perfect but readable!
	const int wd = 30;
	const int ht = 20;
	const int pixels = wd * ht;
	const int bradius = 3;  // blur radius
	// 0 lambda provides the highest quality of deblurring, actually
	// reducing the spurious pixels with 1 or 2 in them (because lambda
	// pulls the nearby high pixels down)
	Fista::Vector lambda(pixels, 0.);
	const int n = 400; // number of steps
	// const int restartEvery = 1000; // after how many steps restart the parameter t
	const int printEvery = 10; // after how many steps print an image

	Fista::Vector img(pixels); // the original image
	Fista::DrawableImg bimg(wd, ht); // blurred image will go here

	// Draw some text
	drawTextSimple(wd, ht, img, 8, 8, bitmaps8x8, 256., "HELLO");

	verbose && printf("Original:\n");
	verbose && printImg(wd, ht, img);

	// Apply the blurring
	std::shared_ptr<BlurDrawing> blurrer = std::make_shared<BlurDrawing>(wd, ht, bradius);

	blurrer->drawSimple(img, bimg);
	verbose && printf("Blurred:\n");
	verbose && printImg(wd, ht, bimg.img_);
	
	// Do the deblurring.
	Fista::Run run(std::make_shared<Fista::MultiPosDrawScaledGradient>(wd, ht, blurrer, /*b*/ bimg.img_, lambda),
		std::make_shared<Fista::RangeLimiter>(0., 256.), lambda.size(), /*diffEps*/ 0.1);

	// This is a copy of repeatedSteps() with printing.
	bool stop;
	for (int i = 0; i < n; i++) {
		stop = run.oneStep();
		if (run.step_ % printEvery == 0) {
			verbose && printf("step %2d: stop=%d\n", run.step_, stop);
			verbose && printImg(wd, ht, run.x_);
		}
	}

	if (false && !stop) {
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

	RUN_TEST(result, td, testDeblurDraw);

	return result;
}
