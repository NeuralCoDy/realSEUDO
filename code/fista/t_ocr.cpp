// Test of FISTA code.

#include <stdio.h>
#include <math.h>
#include "fista.hpp"
#include "fista_gradient.hpp"
#include "draw.hpp"
#include "test.hpp"

// This explores the code similar to SEUDO, except that it uses the
// characters instead of cell images, so it happens to implement a simple
// approach to Optical Charatter Recognition.

// Drawing of all possible characters of the font at a single
// position in the drawable. The weight of each character is controlled
// by a single value in the input vector.
class SingleCharDrawing : public Fista::Drawing
{
public:
	// @param wd - width of the image
	// @param ht - height of the image
	// @param charwd - width of each character in font bitmap
	// @param charht - height of each character in font bitmap
	// @param bitmaps - font bitmaps of characters; each entry in bitmap contains:
	// @param x - X of the upper-left corner of the character on canvas
	// @param y - Y of the upper-left corner of the character on canvas
	// @param from_idx - the starting position of from_idx for draw()
	SingleCharDrawing(int wd, int ht,
		int charwd, int charht, const char **bitmaps, int x, int y, int from_idx = 0)
		: wd_(wd), ht_(ht), charwd_(charwd), charht_(charht),
		bitmaps_(bitmaps), x_(x), y_(y), fontsz_(getFontSize(bitmaps)),
		from_idx_(from_idx)
	{ }

	// Get the precomputed size of the input. It happens to be equal to the
	// font size, with value controlling one bitmap entry.
	int getSourceSize() const
	{
		return fontsz_;
	}

	const char** getBitmaps() const
	{
		return bitmaps_;
	}

	// from Drawing
	virtual void draw(const Fista::Vector &input, Fista::Drawable &dest, Limits &limits);

protected:
	// Image width.
	int wd_;
	// Image height.
	int ht_;
	// character width in the font
	int charwd_;
	// character height in the font
	int charht_;
	// bitmaps of the font (determining the size of the font and the size of the input).
	const char **bitmaps_;
	// X position where the characters get drawn.
	int x_;
	// Y position where the characters get drawn.
	int y_;
	// Size of the font (i.e. the number of character bitmaps in it).
	int fontsz_;
	// the starting position of from_idx for draw()
	int from_idx_;
};

void SingleCharDrawing::draw(const Fista::Vector &input, Fista::Drawable &dest, Limits &limits)
{
	drawEveryChar(wd_, ht_, dest,
		charwd_, charht_, bitmaps_, x_, y_, /*n_chars*/ fontsz_,
		input, from_idx_);
}

int testSimpleOcr(const char *tname)
{
	// Try to draw a character and then recognize it.

	// Dimensions of the font and the font.
	const int charwd = 8;
	const int charht = 8;
	const char** bitmaps = bitmaps8x8;
	// Margin around the character.
	const int margin = 1;
	// Dimensions of the image.
	const int wd = charwd + margin * 2;
	const int ht = charht + margin * 2;
	const int pixels = wd * ht;
	const double lambda_val = 0.;

	const int n = 200; // number of steps
	const int restartEvery = 1000; // after how many steps restart the parameter t
	const int printEvery = 1; // after how many steps print an image

	// Draw the sample to recognize.
	Fista::DrawableImg img(wd, ht);
	drawTextSimple(wd, ht, img.img_, 8, 8, bitmaps8x8, 256., "o", /*x*/ margin, /*y*/ margin);

	verbose && printf("Image to recognize:\n");
	verbose && printImg(wd, ht, img.img_);

	// Define the drawing and the dimension of the source X.
	auto drawing = std::make_shared<SingleCharDrawing>(wd, ht,
		charwd, charht, bitmaps, /*x*/ margin, /*y*/ margin);
	int srcsz = drawing->getSourceSize();
	Fista::Vector lambda(srcsz, lambda_val);

	// Do the OCR.
	Fista::Run run(std::make_shared<Fista::MultiPosDrawScaledGradient>(wd, ht, drawing, /*b*/ img.img_, lambda),
		std::make_shared<Fista::RangeLimiter>(0., 256.), lambda.size(), /*diffEps*/ 0.01);

	// This is a copy of repeatedSteps() with printing.
	Fista::DrawableImg draft(wd, ht); // the current draft as guessed by OCR
	bool stop;
	for (int i = 0; i < n; i++) {
		stop = run.oneStep();
		if (verbose && run.step_ % printEvery == 0) {
			verbose && printf("step %2d: stop=%d\n", run.step_, stop);
			draft.clear();
			drawing->drawSimple(run.x_, draft);
			printImg(wd, ht, draft.img_);
			for (int i = 0; i < srcsz; i++) {
				printf("%c: %f  ", bitmaps[i][0], run.x_[i]);
			}
			printf("\n");
		}
	}

	if (!stop) {
		printf("%s: The stop condition not met!\n", tname);
		return 1;
	}
	return 0;
}

int testSimpleNoisyOcr(const char *tname)
{
	// Try to draw a character and then recognize it, with noise added.

	// Dimensions of the font and the font.
	const int charwd = 8;
	const int charht = 8;
	const char** bitmaps = bitmaps8x8;
	// Margin around the character.
	const int margin = 1;
	// Dimensions of the image.
	const int wd = charwd + margin * 2;
	const int ht = charht + margin * 2;
	const int pixels = wd * ht;
	const double lambda_val = 0.;

	const int n = 200; // number of steps
	const int restartEvery = 1000; // after how many steps restart the parameter t
	const int printEvery = 1; // after how many steps print an image

	// Draw the sample to recognize.
	Fista::DrawableImg img(wd, ht);
	drawTextSimple(wd, ht, img.img_, 8, 8, bitmaps8x8, 256., "o", /*x*/ margin, /*y*/ margin);
	// add noise
	for (int i = 0; i < pixels; i++) {
		img.img_[i] += i % 3;
	}

	verbose && printf("Image to recognize:\n");
	verbose && printImg(wd, ht, img.img_);

	// Define the drawing and the dimension of the source X.
	auto drawing = std::make_shared<SingleCharDrawing>(wd, ht,
		charwd, charht, bitmaps, /*x*/ margin, /*y*/ margin);
	int srcsz = drawing->getSourceSize();
	Fista::Vector lambda(srcsz, lambda_val);

	// Do the OCR.
	Fista::Run run(std::make_shared<Fista::MultiPosDrawScaledGradient>(wd, ht, drawing, /*b*/ img.img_, lambda),
		std::make_shared<Fista::RangeLimiter>(0., 256.), lambda.size(), /*diffEps*/ 0.01);

	// This is a copy of repeatedSteps() with printing.
	Fista::DrawableImg draft(wd, ht); // the current draft as guessed by OCR
	bool stop;
	for (int i = 0; i < n; i++) {
		stop = run.oneStep();
		if (verbose && run.step_ % printEvery == 0) {
			verbose && printf("step %2d: stop=%d\n", run.step_, stop);
			draft.clear();
			drawing->drawSimple(run.x_, draft);
			printImg(wd, ht, draft.img_);
			for (int i = 0; i < srcsz; i++) {
				printf("%c: %f  ", bitmaps[i][0], run.x_[i]);
			}
			printf("\n");
		}
	}

	if (!stop) {
		printf("%s: The stop condition not met!\n", tname);
		return 1;
	}
	return 0;
}

// -------------------------------------------------------------------

// A drawing that draws every possible single character and blurs it.
class SingleCharBlurred : public Fista::DrawPipeline
{
public:
	// @param wd - width of the image
	// @param ht - height of the image
	// @param bradius - blur radius (i.e. 0 will average each pixel with itself
	//      and cause no blur, 1 will include up to 9 pixels, going +-1 by width
	//      and heihgt from the target pixel, unless it's at the edge of the image,
	//      and so on)
	// @param charwd - width of each character in font bitmap
	// @param charht - height of each character in font bitmap
	// @param bitmaps - font bitmaps of characters; each entry in bitmap contains:
	// @param x - X of the upper-left corner of the character on canvas
	// @param y - Y of the upper-left corner of the character on canvas
	SingleCharBlurred(int wd, int ht, int bradius,
		int charwd, int charht, const char **bitmaps, int x, int y)
		: drawing_(std::make_shared<SingleCharDrawing>(
			wd, ht, charwd, charht, bitmaps, x, y))
	{ 
		setHead(drawing_);
		addFilter(std::make_shared<BlurFilter>(wd, ht, bradius));
	}

public:
	// Head of the pipeline that draws the characters.
	// Unlike head_, this preserves the subclass, and is public.
	const std::shared_ptr<SingleCharDrawing> drawing_;
};

int testBlurredOcr(const char *tname)
{
	// Try to draw a character with a blur and then recognize it.

	// Dimensions of the font and the font.
	const int charwd = 8;
	const int charht = 8;
	const char** bitmaps = bitmaps8x8;
	const int bradius = 5;  // blur radius
	// shift during recognition relative to drawing
	int xshift = 0;
	int yshift = 0;
	// Margin around the character.
	const int margin = bradius + 1 + std::max(std::abs(xshift), std::abs(yshift));
	// Dimensions of the image.
	const int wd = charwd + margin * 2;
	const int ht = charht + margin * 2;
	const int pixels = wd * ht;
	const double lambda_val = 0.;

	const int n = 200; // number of steps
	const int restartEvery = 1000; // after how many steps restart the parameter t
	const int printEvery = 1; // after how many steps print an image

	// Draw the sample to recognize.
	Fista::DrawableImg img(wd, ht);
	BlurFilter bfilter(wd, ht, bradius);
	bfilter.setDest(&img);
	Fista::Vector black(1, 256.);
	drawChar(wd, ht, bfilter, 8, 8, bitmaps8x8, /*x*/ margin, /*y*/ margin, 'o', black, 0);

	if (verbose) {
		printf("Image to recognize:\n");
		printImg(wd, ht, img.img_);
	}

	// Define the drawing and the dimension of the source X.
	auto drawing = std::make_shared<SingleCharBlurred>(wd, ht, bradius,
		charwd, charht, bitmaps, /*x*/ margin + xshift, /*y*/ margin + yshift);
	int srcsz = drawing->drawing_->getSourceSize();
	Fista::Vector lambda(srcsz, lambda_val);

	// Do the OCR.
	Fista::Run run(std::make_shared<Fista::MultiPosDrawScaledGradient>(wd, ht, drawing, /*b*/ img.img_, lambda),
		std::make_shared<Fista::RangeLimiter>(0., 256.), lambda.size(), /*diffEps*/ 0.01);

	// This is a copy of repeatedSteps() with printing.
	Fista::DrawableImg draft(wd, ht); // the current draft as guessed by OCR
	bool stop;
	for (int i = 0; i < n; i++) {
		stop = run.oneStep();
		if (verbose && run.step_ % printEvery == 0) {
			verbose && printf("step %2d: stop=%d\n", run.step_, stop);
			draft.clear();
			drawing->drawSimple(run.x_, draft);
			printImg(wd, ht, draft.img_);
			for (int i = 0; i < srcsz; i++) {
				printf("%c: %f  ", bitmaps[i][0], run.x_[i]);
			}
			printf("\n");
		}
	}

	if (!stop) {
		printf("%s: The stop condition not met!\n", tname);
		return 1;
	}
	return 0;
}

// -------------------------------------------------------------------

// The input X array consists of (wd*ht) values representing extra pixels that
// just get directly mapped, followed by the values representing the character
// images. This is similar to what SEUDO does, only without blurring of the
// pixels.
class SingleCharBlurredExtra : public Fista::Drawing
{
public:
	// @param wd - width of the image
	// @param ht - height of the image
	// @param bradius - blur radius (i.e. 0 will average each pixel with itself
	//      and cause no blur, 1 will include up to 9 pixels, going +-1 by width
	//      and heihgt from the target pixel, unless it's at the edge of the image,
	//      and so on)
	// @param charwd - width of each character in font bitmap
	// @param charht - height of each character in font bitmap
	// @param bitmaps - font bitmaps of characters; each entry in bitmap contains:
	// @param x - X of the upper-left corner of the character on canvas
	// @param y - Y of the upper-left corner of the character on canvas
	SingleCharBlurredExtra(int wd, int ht, int bradius,
		int charwd, int charht, const char **bitmaps, int x, int y)
		: wd_(wd), ht_(ht),
		chars_(std::make_shared<SingleCharDrawing>(
			wd, ht, charwd, charht, bitmaps, x, y, wd*ht)),
		bfilter_(std::make_shared<BlurFilter>(wd, ht, bradius))
	{ }

	int getSourceSize() const
	{
		return wd_*ht_ + chars_->getSourceSize();
	}

	int getFontOffset() const
	{
		return wd_*ht_;
	}

	// from Drawing
	virtual void draw(const Fista::Vector &input, Fista::Drawable &dest, Limits &limits)
	{
		// draw the direct pixels
		int i = 0;
		for (int y = 0; y < ht_; y++) {
			for (int x = 0; x < wd_; x++, i++) {
				dest.drawPixel(input, /*from_idx*/ i, x, y, 1.);
			}
		}

		// draw the blurred character
		bfilter_->setDest(&dest);
		chars_->draw(input, *bfilter_, limits);
	}

public:
	int wd_;
	int ht_;
	// Part that draws the characters.
	const std::shared_ptr<SingleCharDrawing> chars_;
	const std::shared_ptr<BlurFilter> bfilter_;
};

int testBlurredExtraOcr(const char *tname)
{
	// Try to draw a character with a blur and then recognize it.
	// The recognition is done with a set of extra pixels that absorb
	// anything that is not used, and get discarded.

	// Dimensions of the font and the font.
	const int charwd = 8;
	const int charht = 8;
	const char** bitmaps = bitmaps8x8;
	const int bradius = 0;  // blur radius
	// shift during recognition relative to drawing
	int xshift = 0;
	int yshift = 0;
	// Margin around the character.
	const int margin = bradius + 1 + std::max(std::abs(xshift), std::abs(yshift));
	// Dimensions of the image.
	const int wd = charwd + margin * 2;
	const int ht = charht + margin * 2;
	const int pixels = wd * ht;
	// Lambda for the individual pixels.
	// 1 drives down the individual pixels slightly better than 0, and 10 even better,
	// when bradius=0. But with a higher blur radius, say 2, using 0 in all lambda works better.
	const double lambda_pixels = 10.;
	// Lambda for the recognized characters.
	const double lambda_ocr = 0.;

	const int n = 400; // number of steps
	// const int restartEvery = 1000; // after how many steps restart the parameter t
	const int printEvery = 10; // after how many steps print an image

	// Draw the sample to recognize.
	Fista::DrawableImg img(wd, ht);
	BlurFilter bfilter(wd, ht, bradius);
	bfilter.setDest(&img);
	Fista::Vector black(1, 256.);
	Fista::Vector gray(1, 100.);
	drawChar(wd, ht, bfilter, 8, 8, bitmaps8x8, /*x*/ margin, /*y*/ margin, 'o', black, 0);

	if (verbose) {
		printf("Image to recognize:\n");
		printImg(wd, ht, img.img_);
	}

	// Define the drawing and the dimension of the source X.
	auto drawing = std::make_shared<SingleCharBlurredExtra>(wd, ht, bradius,
		charwd, charht, bitmaps, /*x*/ margin + xshift, /*y*/ margin + yshift);
	int srcsz = drawing->getSourceSize();
	Fista::Vector lambda(srcsz, lambda_pixels);
	for (int i = pixels; i < srcsz; i++) {
		lambda[i] = lambda_ocr;
	}

	// Do the OCR.
	Fista::Run run(std::make_shared<Fista::MultiPosDrawScaledGradient>(wd, ht, drawing, /*b*/ img.img_, lambda),
		std::make_shared<Fista::RangeLimiter>(0., 256.), lambda.size(), /*diffEps*/ 0.01);

	// This is a copy of repeatedSteps() with printing.
	Fista::DrawableImg draft(wd, ht); // the current draft as guessed by OCR
	bool stop;
	for (int i = 0; i < n; i++) {
		stop = run.oneStep();
		if (verbose && run.step_ % printEvery == 0) {
			verbose && printf("step %2d: stop=%d\n", run.step_, stop);
			verbose && printImg(wd, ht, run.x_);
			draft.clear();
			drawing->drawSimple(run.x_, draft);
			printImg(wd, ht, draft.img_);
			int offset = drawing->getFontOffset();
			for (int i = offset; i < srcsz; i++) {
				printf("%c: %f  ", bitmaps[i - offset][0], run.x_[i]);
			}
			printf("\n");
		}
	}

	if (!stop) {
		printf("%s: The stop condition not met!\n", tname);
		return 1;
	}
	return 0;
}

int testExtraOverlapOcr(const char *tname)
{
	// Try to draw two characters on top of each other and recognize
	// them. The various drawing variants can be uncommented to see the
	// effects.
	// This test is based on testBlurredExtraOcr, so it still contains
	// the blurring option but it's not important.

	// Dimensions of the font and the font.
	const int charwd = 8;
	const int charht = 8;
	const char** bitmaps = bitmaps8x8;
	const int bradius = 0;  // blur radius
	// shift during recognition relative to drawing
	int xshift = 0;
	int yshift = 0;
	// Margin around the character.
	const int margin = bradius + 1 + std::max(std::abs(xshift), std::abs(yshift));
	// Dimensions of the image.
	const int wd = charwd + margin * 2;
	const int ht = charht + margin * 2;
	const int pixels = wd * ht;
	// Lambda for the individual pixels.
	// 1 drives down the individual pixels slightly better than 0, and 10 even better,
	// when bradius=0. But with a higher blur radius, say 2, using 0 in all lambda works better.
	const double lambda_pixels = 10.;
	// Lambda for the recognized characters.
	const double lambda_ocr = 0.;

	const int n = 400; // number of steps
	// const int restartEvery = 1000; // after how many steps restart the parameter t
	const int printEvery = 10; // after how many steps print an image

	// Draw the sample to recognize.
	Fista::DrawableImg img(wd, ht);
	BlurFilter bfilter(wd, ht, bradius);
	bfilter.setDest(&img);
	Fista::Vector black(1, 256.);
	Fista::Vector gray(1, 100.);

	// Drawing additively tends to be recognized fine, even if the characters are as 
	// similar as "e" and "o".
	// Drawing with saturation (using drawTextSimple) has the brighter character win,
	// and the weak character to be seen as a set of points.
	// Saturation can probably be compensated for by defining for the saturated
	// pixels an additive value in X with a negative range and a negative lambda.

	// drawChar(wd, ht, bfilter, 8, 8, bitmaps8x8, /*x*/ margin, /*y*/ margin, 'o', black, 0);
	drawChar(wd, ht, bfilter, 8, 8, bitmaps8x8, /*x*/ margin, /*y*/ margin, 'o', gray, 0);
	drawChar(wd, ht, bfilter, 8, 8, bitmaps8x8, /*x*/ margin, /*y*/ margin, 'e', gray, 0);
	// drawTextSimple(wd, ht, img.img_, 8, 8, bitmaps8x8, /*color*/ black[0], "H", /*startx*/ margin, /*starty*/ margin);

	if (verbose) {
		printf("Image to recognize:\n");
		printImg(wd, ht, img.img_);
	}

	// Define the drawing and the dimension of the source X.
	auto drawing = std::make_shared<SingleCharBlurredExtra>(wd, ht, bradius,
		charwd, charht, bitmaps, /*x*/ margin + xshift, /*y*/ margin + yshift);
	int srcsz = drawing->getSourceSize();
	Fista::Vector lambda(srcsz, lambda_pixels);
	for (int i = pixels; i < srcsz; i++) {
		lambda[i] = lambda_ocr;
	}

	// Do the OCR.
	Fista::Run run(std::make_shared<Fista::MultiPosDrawScaledGradient>(wd, ht, drawing, /*b*/ img.img_, lambda),
		std::make_shared<Fista::RangeLimiter>(0., 256.), lambda.size(), /*diffEps*/ 0.01);

	// This is a copy of repeatedSteps() with printing.
	Fista::DrawableImg draft(wd, ht); // the current draft as guessed by OCR
	bool stop;
	for (int i = 0; i < n; i++) {
		stop = run.oneStep();
		if (verbose && run.step_ % printEvery == 0) {
			verbose && printf("step %2d: stop=%d\n", run.step_, stop);
			verbose && printImg(wd, ht, run.x_);
			draft.clear();
			drawing->drawSimple(run.x_, draft);
			printImg(wd, ht, draft.img_);
			int offset = drawing->getFontOffset();
			for (int i = offset; i < srcsz; i++) {
				printf("%c: %f  ", bitmaps[i - offset][0], run.x_[i]);
			}
			printf("\n");
		}
	}

	if (!stop) {
		printf("%s: The stop condition not met!\n", tname);
		return 1;
	}
	return 0;
}

int testExtraNoisyOcr(const char *tname)
{
	// Add noise to the image (for repeatability, not random but regular,
	// but it doesnt matter). Otherwise based on testBlurredExtraOcr.

	// Dimensions of the font and the font.
	const int charwd = 8;
	const int charht = 8;
	const char** bitmaps = bitmaps8x8;
	const int bradius = 0;  // blur radius
	// shift during recognition relative to drawing
	int xshift = 0;
	int yshift = 0;
	// Margin around the character.
	const int margin = bradius + 1 + std::max(std::abs(xshift), std::abs(yshift));
	// Dimensions of the image.
	const int wd = charwd + margin * 2;
	const int ht = charht + margin * 2;
	const int pixels = wd * ht;

	// Using a small positive lambda for both pixels and characters seems to work best.
	// Lambda for the individual pixels.
	const double lambda_pixels = 1.;
	// Lambda for the recognized characters.
	const double lambda_ocr = 1.;

	const int n = 400; // number of steps
	// const int restartEvery = 1000; // after how many steps restart the parameter t
	const int printEvery = 10; // after how many steps print an image

	// Draw the sample to recognize.
	Fista::DrawableImg img(wd, ht);
	BlurFilter bfilter(wd, ht, bradius);
	bfilter.setDest(&img);
	Fista::Vector black(1, 256.);
	Fista::Vector gray(1, 100.);
	drawChar(wd, ht, bfilter, 8, 8, bitmaps8x8, /*x*/ margin, /*y*/ margin, 'o', gray, 0);
	// add noise
	for (int i = 0; i < pixels; i++) {
		img.img_[i] += i % 3;
	}

	if (verbose) {
		printf("Image to recognize:\n");
		printImg(wd, ht, img.img_);
	}

	// Define the drawing and the dimension of the source X.
	auto drawing = std::make_shared<SingleCharBlurredExtra>(wd, ht, bradius,
		charwd, charht, bitmaps, /*x*/ margin + xshift, /*y*/ margin + yshift);
	int srcsz = drawing->getSourceSize();
	Fista::Vector lambda(srcsz, lambda_pixels);
	for (int i = pixels; i < srcsz; i++) {
		lambda[i] = lambda_ocr;
	}

	// Setting the threads has no effect on a non-parallel drawing.
	auto gradient =
		std::make_shared<Fista::MultiPosDrawScaledGradient>(wd, ht, drawing, /*b*/ img.img_, lambda, /*nthreads*/ 10);

	// Test that the thread partitioning got initialized sensibly.
	auto limits1 = gradient->limitsPass1();
	if (limits1.size() != 1) {
		printf("%s: Limits1 size is wrong: size=%d, should be 1\n",
			tname, (int)limits1.size());
		return 1;
	}
	if (limits1.back().end_dest_y_ != ht) {
		printf("%s: Limits1 ends wrong: end=%d, ht=%d\n",
			tname, limits1.back().end_dest_y_, ht);
		return 1;
	}

	auto limits2 = gradient->limitsPass2();
	if (limits2.size() != 1) {
		printf("%s: Limits2 size is wrong: size=%d, should be 1\n",
			tname, (int)limits2.size());
		return 1;
	}
	if (limits2.back().end_input_ != srcsz) {
		printf("%s: Limits2 ends wrong: end=%d, ninputs=%d\n",
			tname, limits2.back().end_input_, srcsz);
		return 1;
	}

	// Do the OCR.
	Fista::Run run(gradient, 
		std::make_shared<Fista::RangeLimiter>(0., 256.), lambda.size(), /*diffEps*/ 0.01);

	// This is a copy of repeatedSteps() with printing.
	Fista::DrawableImg draft(wd, ht); // the current draft as guessed by OCR
	bool stop;
	for (int i = 0; i < n; i++) {
		stop = run.oneStep();
		if (verbose && run.step_ % printEvery == 0) {
			verbose && printf("step %2d: stop=%d\n", run.step_, stop);
			verbose && printImg(wd, ht, run.x_);
			draft.clear();
			drawing->drawSimple(run.x_, draft);
			printImg(wd, ht, draft.img_);
			int offset = drawing->getFontOffset();
			for (int i = offset; i < srcsz; i++) {
				printf("%c: %f  ", bitmaps[i - offset][0], run.x_[i]);
			}
			printf("\n");
		}
	}

	if (!stop) {
		printf("%s: The stop condition not met!\n", tname);
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

	RUN_TEST(result, td, testSimpleOcr);
	RUN_TEST(result, td, testSimpleNoisyOcr);
	RUN_TEST(result, td, testBlurredOcr);
	RUN_TEST(result, td, testBlurredExtraOcr);
	RUN_TEST(result, td, testExtraOverlapOcr);
	RUN_TEST(result, td, testExtraNoisyOcr);

	return result;
}
