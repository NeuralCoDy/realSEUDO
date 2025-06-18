// Test of Sprite drawing.

#include <stdio.h>
#include "fista_types.hpp"
#include "draw.hpp"
#include "test.hpp"

int testDrawPositiveOrigin(const char *tname)
{
	// origin at center
	Fista::Sprite s(5, 3, 2, 1);
	s.img_.assign( {
		1., 2., 3., 4., 5.,
		10., 20., 30., 40., 50.,
		100., 200., 300., 400., 500.,
	} );

	Fista::DrawableImg canvas(20, 10);

	if (verbose) {
		printf("sprite:\n");
		printImg(s);
	}

	Fista::Vector argx({-1., 1., -2.});

	// too far to the left
	s.draw(argx, /*from_idx*/ 1, canvas, -3, 0);
	// too far to the right
	s.draw(argx, /*from_idx*/ 1, canvas, 22, 0);
	// too far up
	s.draw(argx, /*from_idx*/ 1, canvas, 0, -2);
	// too far down
	s.draw(argx, /*from_idx*/ 1, canvas, 0, 11);
	if (verbose) {
		printf("img after drawing too far:\n");
		printImg(canvas);
	}
	for (int i = 0; i < canvas.wd_*canvas.ht_; i++) {
		if (canvas.img_[i] != 0.) {
			printf("Unexpected value %f at position %d\n", canvas.img_[i], i);
			return 1;
		}
	}

	// Draw with origin in the upper left corner.
	s.draw(argx, /*from_idx*/ 1, canvas, 0, 0);
	// Draw with origin in the upper right corner.
	s.draw(argx, /*from_idx*/ 1, canvas, 19, 0);
	// Draw with origin in the lower left corner.
	s.draw(argx, /*from_idx*/ 1, canvas, 0, 9);
	// Draw with origin in the lower right corner.
	s.draw(argx, /*from_idx*/ 1, canvas, 19, 9);

	// Draw in (almost) middle
	s.draw(argx, /*from_idx*/ 1, canvas, 10, 5, 0.5);

	if (verbose) {
		printf("img after drawing:\n");
		printImg(canvas);
	}

	Fista::Vector expected({
    30.,  40.,  50.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  10.,  20.,  30.,
   300., 400., 500.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0., 100., 200., 300.,
     0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
     0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
     0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  0.5,   1.,  1.5,   2.,  2.5,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
     0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   5.,  10.,  15.,  20.,  25.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
     0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  50., 100., 150., 200., 250.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
     0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
     3.,   4.,   5.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   1.,   2.,   3.,
    30.,  40.,  50.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  10.,  20.,  30.,
	});
	for (int i = 0; i < canvas.wd_*canvas.ht_; i++) {
		if (canvas.img_[i] != expected[i]) {
			printf("Unexpected value %f at position %d\n", canvas.img_[i], i);
			return 1;
		}
	}

	return 0;
}

int testDrawNegativeOrigin(const char *tname)
{
	// origin 5 pixels to the left and 1 up
	Fista::Sprite s(5, 3, -5, -1);
	s.img_.assign( {
		1., 2., 3., 4., 5.,
		10., 20., 30., 40., 50.,
		100., 200., 300., 400., 500.,
	} );

	Fista::DrawableImg canvas(20, 10);

	if (verbose) {
		printf("sprite:\n");
		printImg(s);
	}

	Fista::Vector argx({-1., 1., -2.});

	// Draw with origin in the upper left corner.
	s.draw(argx, /*from_idx*/ 1, canvas, 0, 0);
	if (verbose) {
		printf("img after drawing:\n");
		printImg(canvas);
	}

	if (canvas.img_[25] != 1.) {
		printf("Unexpected value %f at position %d\n", canvas.img_[25], 25);
		return 1;
	}

	return 0;
}

int testDrawLimitY(const char *tname)
{
	// origin at center
	Fista::Sprite s(5, 3, 2, 1);
	s.img_.assign( {
		1., 2., 3., 4., 5.,
		10., 20., 30., 40., 50.,
		100., 200., 300., 400., 500.,
	} );

	Fista::DrawableImg canvas(8, 5);

	if (verbose) {
		printf("sprite:\n");
		printImg(s);
	}

	Fista::Vector argx({-1., 1., -2.});

	// Draw with no lines allowed
	s.draw(argx, /*from_idx*/ 1, canvas, 0, 0, 0, 0);

	if (verbose) {
		printf("img after drawing with no lines allowed:\n");
		printImg(canvas);
	}
	for (int i = 0; i < canvas.wd_*canvas.ht_; i++) {
		if (canvas.img_[i] != 0.) {
			printf("Unexpected value %f at position %d\n", canvas.img_[i], i);
			return 1;
		}
	}

	// Draw line 0
	s.draw(argx, /*from_idx*/ 1, canvas, 4, 1, 0, 1, 2.);
	// Draw line 1
	s.draw(argx, /*from_idx*/ 1, canvas, 4, 1, 1, 2, 1.);
	// Draw line 2
	s.draw(argx, /*from_idx*/ 1, canvas, 4, 1, 2, 3, 0.5);

	if (verbose) {
		printf("img after drawing:\n");
		printImg(canvas);
	}

	Fista::Vector expected({
		  0.,   0.,   2.,   4.,   6.,   8.,  10.,   0.,
		  0.,   0.,  10.,  20.,  30.,  40.,  50.,   0.,
		  0.,   0.,  50., 100., 150., 200., 250.,   0.,
		  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
		  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
	});
	for (int i = 0; i < canvas.wd_*canvas.ht_; i++) {
		if (canvas.img_[i] != expected[i]) {
			printf("Unexpected value %f at position %d\n", canvas.img_[i], i);
			return 1;
		}
	}

	return 0;
}

int testCropWhitespace(const char *tname)
{
	Fista::Sprite s(5, 4, 3, 2);
	s.img_.assign( {
		0., 0., 0., 0., 0.,
		0., 2., 0., 4., 0.,
		0., 20., 0., 40., 0.,
		0., 0., 0., 0., 0.,
	} );

	if (verbose) {
		printf("sprite:\n");
		printImg(s);
	}

	s.cropWhitespace();
	verbose && printf("after crop wd=%d ht=%d origin_x=%d origin_y=%d pixels=%d\n",
		s.wd_, s.ht_, s.origin_x_, s.origin_y_, (int)s.img_.size());

	if (s.wd_ != 3) {
		printf("Wrong wd=%d\n", s.wd_);
		return 1;
	}
	if (s.ht_ != 2) {
		printf("Wrong ht=%d\n", s.ht_);
		return 1;
	}
	if (s.origin_x_ != 2) {
		printf("Wrong origin_x=%d\n", s.origin_x_);
		return 1;
	}
	if (s.origin_y_ != 1) {
		printf("Wrong origin_y=%d\n", s.origin_y_);
		return 1;
	}
	if (s.img_.size() != 3*2) {
		printf("Wrong pixels=%d\n", (int)s.img_.size());
		return 1;
	}

	if (verbose) {
		printImg(s);
	}

	Fista::Vector expected({
		2., 0., 4.,
		20., 0., 40.,
	});

	for (int i = 0; i < expected.size(); i++) {
		if (s.img_[i] != expected[i]) {
			printf("Unexpected value %f at position %d\n", s.img_[i], i);
			return 1;
		}
	}

	// Try to crop again, should not change.
	s.cropWhitespace();
	verbose && printf("after 2nd crop wd=%d ht=%d origin_x=%d origin_y=%d pixels=%d\n",
		s.wd_, s.ht_, s.origin_x_, s.origin_y_, (int)s.img_.size());

	if (s.wd_ != 3) {
		printf("Wrong wd=%d\n", s.wd_);
		return 1;
	}
	if (s.ht_ != 2) {
		printf("Wrong ht=%d\n", s.ht_);
		return 1;
	}
	if (s.origin_x_ != 2) {
		printf("Wrong origin_x=%d\n", s.origin_x_);
		return 1;
	}
	if (s.origin_y_ != 1) {
		printf("Wrong origin_y=%d\n", s.origin_y_);
		return 1;
	}
	if (s.img_.size() != 3*2) {
		printf("Wrong pixels=%d\n", (int)s.img_.size());
		return 1;
	}

	if (verbose) {
		printImg(s);
	}

	for (int i = 0; i < expected.size(); i++) {
		if (s.img_[i] != expected[i]) {
			printf("Unexpected value %f at position %d\n", s.img_[i], i);
			return 1;
		}
	}

	return 0;
}

int testCropWhitespaceEmpty(const char *tname)
{
	Fista::Sprite s(5, 4, 3, 2);
	s.img_.assign( {
		0., 0., 0., 0., 0.,
		0., 0., 0., 0., 0.,
		0., 0., 0., 0., 0.,
		0., 0., 0., 0., 0.,
	} );

	if (verbose) {
		printf("sprite:\n");
		printImg(s);
	}

	s.cropWhitespace();
	verbose && printf("after crop wd=%d ht=%d origin_x=%d origin_y=%d pixels=%d\n",
		s.wd_, s.ht_, s.origin_x_, s.origin_y_, (int)s.img_.size());

	if (s.wd_ != 0) {
		printf("Wrong wd=%d\n", s.wd_);
		return 1;
	}
	if (s.ht_ != 0) {
		printf("Wrong ht=%d\n", s.ht_);
		return 1;
	}
	if (s.origin_x_ != 3) {
		printf("Wrong origin_x=%d\n", s.origin_x_);
		return 1;
	}
	if (s.origin_y_ != 2) {
		printf("Wrong origin_y=%d\n", s.origin_y_);
		return 1;
	}
	if (s.img_.size() != 0) {
		printf("Wrong pixels=%d\n", (int)s.img_.size());
		return 1;
	}

	if (verbose) {
		printImg(s);
	}

	// Make sure that drawing an empty image doesn't crash.
	Fista::DrawableImg canvas(20, 10);
	Fista::Vector argx({-1., 1., -2.});
	s.draw(argx, /*from_idx*/ 1, canvas, 0, 0);

	return 0;
}

// -------------------------------------------------------------------

int
main(int argc, char **argv)
{
	TestDriver td(argc, argv);
	int result = 0;

	RUN_TEST(result, td, testDrawPositiveOrigin);
	RUN_TEST(result, td, testDrawNegativeOrigin);
	RUN_TEST(result, td, testDrawLimitY);
	RUN_TEST(result, td, testCropWhitespace);
	RUN_TEST(result, td, testCropWhitespaceEmpty);

	return result;
}
