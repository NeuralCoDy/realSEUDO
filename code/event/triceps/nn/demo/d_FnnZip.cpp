//
// (C) Copyright 2022 Sergey A. Babkin.
// This file is a part of Triceps.
// See the file COPYRIGHT for the copyright notice and license information
//
//
// Demo of the recongnition of handwritten digits from the MNIST database of ZIP codes.

#include <utest/Utest.h>
#include <nn/test/TestFloatNn.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdio.h>
#include <string>
#include <memory>
#include <map>

using namespace std;

static long random_seed = time(NULL);

// Each element of training data contains:
//   input: 16 x 16 pixels = 256 elements, ranging from -1 to 1
//   ouput: 10 elements, corresponding to '0' through '9', contains 1 for
//     the match, -1 for the rest
typedef TestFloatNn::Training Training;
typedef TestFloatNn::TrainingVector TrainingVector;

// Tries to locate the file zip.train.gz by looking for subdirectory
// "zipnn" in the current directory, then sequentially in its parents
// up to 10 levels up.
// Returns the directory path including "zipnn", or an empty string
// if not found.
static string
findZipFiles()
{
	string path = "zipnn";

	for (int i = 0; i <= 10; i++) {
		string f = path + "/zip.train.gz";
		struct stat st;
		if (stat(f.c_str(), &st) == 0) {
			return path;
		}
		path = "../" + path;
	}
	return ""; // not found;
}

// To make processing faster, shrink the original 16x16 image to 8x8
// by avegaring each group of 4 pixels.
void shrinkImage(Training &data)
{
	constexpr int rate = 2; // shrinkage rate
	constexpr int dim = 16 / rate; // new dimensions
	
	// replace the pixels in place
	for (int r = 0; r < dim; r++) {
		for (int c = 0; c < dim; c++) {
			double v = 0.;
			int old_y = r * rate;
			int old_x = c * rate;
			int old_offset = old_y * 16 + old_x;
			for (int i = 0; i < rate; i++) {
				for (int j = 0; j < rate; j++) {
					v += data.input[old_offset + i * 16 + j];
				}
			}
			data.input[r * dim + c] = v / (rate * rate);
		}
	}
	data.input.resize(dim * dim);
}

// Change the image to strict black & white
void bwImage(Training &data)
{
	for (size_t i = 0; i < data.input.size(); i++) {
		if (data.input[i] > -0.1) {
			data.input[i] = 1.;
		} else {
			data.input[i] = -1.;
		}
	}
}

// Encode the B&W image as run length by rows
void runLenImage(Training &data)
{
	int dim = (int)sqrt((double)data.input.size());
	
	// replace the pixels in place
	for (int r = 0; r < dim; r++) {
		int tc = 0;
		int count = 0;
		bool space = true; // space vs drawing, a row always starts with space
		for (int c = 0; c < dim; c++) {
			double v = data.input[r * dim + c];
			if ((v < 0) ^ space) {
				// end of run
				data.input[r * dim + tc] = count;
				++tc;
				space = !space;
				count = 1; // the current pixel
			} else {
				++count; // continue the current run length
			}
		}

		// The last run
		data.input[r * dim + tc] = count;
		// All the other runs in this row are unused
		for (++tc; tc < dim; ++tc)
			data.input[r * dim + tc] = -1.; // 0. or -1. ?
	}
}

struct Trapeze {
	Trapeze() = default;
	Trapeze(const Trapeze&) = default;

	Trapeze(FloatNeuralNet::Value top, FloatNeuralNet::Value bottom, FloatNeuralNet::Value slope)
		: topWd_(top), bottomWd_(bottom), leftSlope_(slope)
	{ }

	// Width of trapeze at the top (in pixels or some relative units).
	FloatNeuralNet::Value topWd_ = -1.;
	// Width of trapeze at the bottom (in pixels or some relative units).
	FloatNeuralNet::Value bottomWd_ = -1.;
	// Average slope of the left edge of the trapeze.
	// Since the vertical line is possible here and a horizontal line is
	// impossible, we measure the slope "sideways", dx/dy, where dy is always
	// positive, and a vertical line has a slope of 0.
	FloatNeuralNet::Value leftSlope_ = 0.;

	// The part below is used in conversion to a reduced bitmap

	FloatNeuralNet::Value color_ = 0.; // -1 for whitespace, 1 for drawing
};

// One row of trapezes, representing a horizontal slice of a B&W image.
struct TrapezeRow {
	TrapezeRow() {}

	// A convenience constructor for an empty row.
	// @param height - height of the trapeze in image units
	// @param twidth - width in trapezes, size for this many trapezes
	//    will be reserved in the trapeze vector and set to widths of -1.
	TrapezeRow(FloatNeuralNet::Value height, size_t twidth)
		: height_(height), widths_(twidth)
	{ }

	// Constructor for a filled row from pixels.
	// @param height - height of the trapeze in pixels, must not be 0
	// @param topStartPos - position of starting pixels for trapezes at the top of row, must
	//   contain at leats 1 element
	// @param bottomStartPos - position of starting pixels for trapezes at the bottom of row,
	//   must have the same size as topStartPos
	// @param bbox_right - position of the rightmost pixel in the bounding box, that
	//   provides the end of the last trapeze (both top and bottom); in case if the rightmost
	//   trapeze has 0 width, its starting position will be (bbox_right+1)
	TrapezeRow(int height, const vector<int> &topStartPos, const vector<int> &bottomStartPos, int bbox_right)
		: height_(height), widths_(topStartPos.size())
	{
		for (size_t i = 0; i < topStartPos.size() - 1; i++) {
			Trapeze &t = widths_[i];
			t.topWd_ = topStartPos[i + 1] - topStartPos[i];
			t.bottomWd_ = bottomStartPos[i + 1] - bottomStartPos[i];
			t.leftSlope_ = height_ <= 1? 0. : (bottomStartPos[i] - topStartPos[i]) / (height_ - 1.);
		}
		{
			size_t i = topStartPos.size() - 1;
			Trapeze &t = widths_[i];
			t.topWd_ = bbox_right + 1 - topStartPos[i];
			t.bottomWd_ = bbox_right + 1 - bottomStartPos[i];
			t.leftSlope_ = height_ <= 1? 0. : (bottomStartPos[i] - topStartPos[i]) / (height_ - 1.);
		}
	}

	// Reset to empty
	void clear()
	{
		height_ = -1.;
		widths_.clear();
	}

	// Height of the row
	FloatNeuralNet::Value height_ = -1.;
	// Widths of trapezes in the row, the values go in pairs
	// (top width, bottom width), always starting and ending on a space
	// trapeze (both of which may be of width 0 on top, bottom, or both).
	vector<Trapeze> widths_;
};

// A B&W image represented as a stack of rows of trapezes.
struct TrapezeGlyph {
	void computeMaxWidth() {
		size_t mw = 0;
		for (auto it = rows_.begin(); it != rows_.end(); ++it) {
			size_t wd = it->widths_.size();
			if (wd > mw)
				mw = wd;
		}
		maxWidth_ = mw;
	}

	size_t maxWidth_ = 0.;  // max width (i.e. vector size) of all the rows
	vector<TrapezeRow> rows_;
	int label_ = -1; // label for training
};

// Encode the B&W image as rows of trapezes, returning the new glyph.
shared_ptr<TrapezeGlyph>
trapezeGlyphImage(Training &data)
{
	int dim = (int)sqrt((double)data.input.size());

	// Start by finding the bounding box.
	int bb_left = dim - 1;
	int bb_right = 0;
	int bb_up = dim - 1;
	int bb_down = 0;

	for (int r = 0; r < dim; r++) {
		for (int c = 0; c < dim; c++) {
			if (data.input[r * dim + c] > 0) {
				if (c < bb_left)
					bb_left = c;
				if (c > bb_right)
					bb_right = c;
				if (r < bb_up)
					bb_up = r;
				if (r > bb_down)
					bb_down = r;
			}
		}
	}

	auto glyph = make_shared<TrapezeGlyph>();

	// For last first and (tentative) last pixel row, the start points of every run-length
	// segment in it.
	vector<int> fpxStart, lpxStart;
	// Direction of the start side of each trapeze (which is also the end side of the
	// next trapeze). We don't care about the end side of the last trapeze because it's
	// always vertical. The start of the first trapeze is also always vertical but
	// it's included for convenience of indexing.
	// -1 means that it's expanding to the left, 0 vertical, 1 expanding to the right.
	vector<int> dir;
	// Top pixel row of the current trapeze.
	int top = -1;

	if (bb_left > bb_right) {
		// it's empty, make a single space trapeze
		fpxStart.push_back(0);
		glyph->rows_.emplace_back(TrapezeRow(dim, fpxStart, fpxStart, dim - 1));
		glyph->computeMaxWidth();
		return glyph;
	}

	for (int r = bb_up; r <= bb_down; r++) {
		// For the current pixel row, start and end points of the run-length segments,
		// similar to lpxStart.
		vector<int> cpxStart;
		bool space = true; // space vs drawing, a row always starts with space
		// Build the run-length segment for the current pixel row.
		cpxStart.push_back(bb_left);
		for (int c = bb_left; c <= bb_right; c++) {
			double v = data.input[r * dim + c];
			if ((v < 0) ^ space) {
				// end of run
				cpxStart.push_back(c);
				space = !space;
			}
		}
		// make sure that the last segment is a space, even if an empty one
		if (!space) {
			cpxStart.push_back(bb_right + 1);
		}

		// Now see if the current trapeze can be extended.
		bool extend = true;
		if (cpxStart.size() == lpxStart.size()) {
			for (size_t seg = 1; seg < lpxStart.size(); seg++) {
				// the boundary must not overlap with the last line's previous segment
				// of the same color (since the colors alternate, that would be the end
				// of segment at (seg-2) which is also the start of (seg-1)).
				if (cpxStart[seg] < lpxStart[seg - 1]) {
					extend = false;
					break;
				}

				// the trapeze must continue shifting the start in the same
				// direction as before (or be vertical).
				int sdir = cpxStart[seg] - lpxStart[seg];
				if (sdir * dir[seg] < 0) {
					extend = false;
					break;
				}
				// If shifting from vertical, record the new definite direction.
				if (dir[seg] == 0 && sdir != 0) {
					if (sdir > 0)
						dir[seg] = 1;
					else
						dir[seg] = -1;
				}
			}
		} else {
			extend = false;
		}

		if (!extend) {
			// The previous trapeze has ended
			if (top >= bb_up) {
				// there really was a previous trapeze
				glyph->rows_.emplace_back(TrapezeRow(r - top, fpxStart, lpxStart, bb_right));
				top = r;
			} else {
				top = bb_up;
			}
			fpxStart = cpxStart;
			dir.assign(cpxStart.size(), 0.);
		}
		// The current row becomes the new last
		lpxStart = cpxStart;
	}

	// Complete the last trapeze row.
	if (top < bb_up)
		top = bb_up;
	glyph->rows_.emplace_back(TrapezeRow(bb_down + 1 - top, fpxStart, lpxStart, bb_right));

	glyph->computeMaxWidth();
	glyph->label_ = data.label;
	return glyph;
}

// Convert a trapeze glyph to a training format.
// @param absCoord - flag: instead of widths, store the absolute X coordinate of the left side
// @param glyph - the glyph to convert, it will also be resized up to the width and height
// @param wd - fixed width that is max for all glyphs
// @param ht - fixed height that is max for all glyphs
// @param prototype - if not NULL, source to copy the expected output and other values from,
//   otherwise they get generated from the glyph
shared_ptr<Training>
trapezeGlyphToTraining(bool absCoord, TrapezeGlyph &glyph, size_t wd, size_t ht, const Training *prototype = nullptr)
{
	// Fill the empty columns on the right
	for (auto grit = glyph.rows_.begin(); grit !=  glyph.rows_.end(); ++grit) {
		grit->widths_.resize(wd);
	}
	// Fill the empty rows on the bottom
	for (size_t i = glyph.rows_.size(); i < ht; i++) {
		glyph.rows_.emplace_back(TrapezeRow(-1., wd));
	}

	constexpr size_t VALS_PER_TRAPEZE = 3;
	shared_ptr<Training> t = make_shared<Training>(wd * ht * VALS_PER_TRAPEZE, 10);
	
	// Do the conversion for the inputs
	size_t r = 0;
	for (auto grit = glyph.rows_.begin(); grit != glyph.rows_.end() && r < ht; ++grit, ++r) {
		const TrapezeRow &row = *grit;
		size_t c = 0;
		FloatNeuralNet::Value topX = 0., bottomX = 0.;
		for (auto it = row.widths_.begin(); it != row.widths_.end() && c < wd; ++it, ++c) {
			size_t pos = (r * wd + c) * VALS_PER_TRAPEZE;
			t->input[pos++] = it->leftSlope_;
			if (absCoord) {
				if (it->topWd_ < 0.) {
					t->input[pos++] = -1.;
					t->input[pos++] = -1.;
				} else {
					t->input[pos++] = topX;
					t->input[pos++] = bottomX;
				}
			} else {
				t->input[pos++] = it->topWd_;
				t->input[pos++] = it->bottomWd_;
			}

			topX += it->topWd_;
			bottomX += it->bottomWd_;
		}
		// Since the very first trapeze always has the left slope of 0,
		// its location is reused for the row height
		t->input[r * wd * VALS_PER_TRAPEZE] = row.height_;
	}

	if (prototype != nullptr) {
		// Copy the rest from prototype
		t->output = prototype->output;
		t->label = prototype->label;
		t->count = prototype->count;
		t->effectiveCount = prototype->effectiveCount;
	} else {
		t->output[glyph.label_] = 1.;
		t->label = glyph.label_;
		t->count = 1;
		t->effectiveCount = 1;
	}

	return t;
}

// Convert a trapeze glyph in the training format to the high-level form.
// @param absCoord - flag: instead of widths, the training format has the absolute X coordinate of the left side
// @param t - the glyph to convert
// @param wd - fixed width (in trapezes) in the training format
// @param ht - fixed height (in trapezes) in the training format
shared_ptr<TrapezeGlyph>
trainingToTrapezeGlyph(bool absCoord, const Training &t, size_t wd, size_t ht)
{
	constexpr size_t VALS_PER_TRAPEZE = 3;

	auto glyph = make_shared<TrapezeGlyph>();
	glyph->maxWidth_ = wd;
	glyph->label_ = t.label;

	for (size_t r = 0; r < ht; r++) {
		glyph->rows_.emplace_back(TrapezeRow(-1., wd));
		TrapezeRow &row = glyph->rows_[r];

		for (size_t c = 0; c < wd; c++) {
			size_t pos = (r * wd + c) * VALS_PER_TRAPEZE;
			row.widths_[c].leftSlope_ = t.input[pos++];

			row.widths_[c].topWd_ = t.input[pos++];
			row.widths_[c].bottomWd_ = t.input[pos++];

			if (absCoord && row.widths_[c].topWd_  >= 0.) {
				if (c + 1 >= wd || t.input[pos + 1] < 0.) {
					// No info about the width of the last column
					row.widths_[c].topWd_ = 0.;
					row.widths_[c].bottomWd_ = 0.;
				} else {
					// Compute the width by subtracting from the next values
					row.widths_[c].topWd_ = t.input[pos + 1] - row.widths_[c].topWd_;
					row.widths_[c].bottomWd_ = t.input[pos + 2] - row.widths_[c].bottomWd_;
				}
			}
		}

		// Extract the row height from the first slope.
		row.height_ = row.widths_[0].leftSlope_;
		row.widths_[0].leftSlope_ = 0.;
	}

	return glyph;
}

// Convert a trapeze glyph to a training data in bitmap format, shrinking it if necessary.
// @param glyph - the glyph to convert
// @param wd - fixed width of bitmap
// @param ht - fixed height of bitmap
// @param prototype - if not NULL, source to copy the expected output and other values from,
//   otherwise they get generated from the glyph
shared_ptr<Training>
trapezeGlyphToBitmap(const TrapezeGlyph &glyph, size_t wd, size_t ht, const Training *prototype = nullptr)
{
	// First convert to the bitmap as-is, to best fit the trapezes.

	// This glyph will contain the absolute X coordinate values of the left side of trapezes.
	TrapezeGlyph glabs = glyph;

	// Collect all the unique X coordinates here
	map<FloatNeuralNet::Value, size_t> xmap;

	// height of the first bitmap
	size_t ht1 = 0;

	// Convert to absolute values and collect the unique X coordinates
	for (auto grit = glabs.rows_.begin(); grit != glabs.rows_.end(); ++grit) {
		TrapezeRow &row = *grit;
		if (row.height_ < 0)
			break; // it's a filler

		FloatNeuralNet::Value topX = 0., bottomX = 0.;
		bool vertical = true;
		for (auto it = row.widths_.begin(); it != row.widths_.end(); ++it) {
			if (it->bottomWd_ < 0)
				break; // it's a filler

			FloatNeuralNet::Value top = topX, bottom = bottomX;

			topX += it->topWd_;
			bottomX += it->bottomWd_;

			it->topWd_ = top;
			it->bottomWd_ = bottom;

			xmap[top] = 0;
			xmap[bottom] = 0;

			if (top != bottom)
				vertical = false;
		}

		// If all slopes are vertical, we can represent them with 1 pixel high,
		// otherwise we need 2 pixels.
		if (vertical) {
			row.height_ = 1.;
			ht1 += 1;
		} else {
			row.height_ = 2.;
			ht1 += 2;
		}
	}

	// leave 1 pixel between each distinct X coordinate.
	size_t x = 0;
	for (auto it = xmap.begin(); it != xmap.end(); ++it, ++x) {
		it->second = x;
	}

	// Build the first cut at the bitmap.
	size_t wd1 = xmap.size() - 1; // number of pixels is 1 less than number of boundaries
	if (wd1 == 0)
		wd1 = 1;
	if (ht1 == 0)
		ht1 = 1;

	FloatNeuralNet::ValueVector vec1(wd1 * ht1, -1.);
	FloatNeuralNet::ValueSubMatrix bitmap(vec1, 0, ht1, wd1);

	size_t r = 0;
	for (auto grit = glabs.rows_.begin(); grit != glabs.rows_.end(); ++grit) {
		TrapezeRow &row = *grit;
		if (row.height_ < 0)
			break; // it's a filler

		size_t end = row.widths_.size();
		// The even trapezes contain space which is already filled, the
		// odd trapezes contain drawings, and the valid values are guaranteed
		// to start and end with a space.
		for (size_t c = 1; c < end; c += 2) {
			auto &left = row.widths_[c];
			auto &right = row.widths_[c + 1];

			if (left.bottomWd_ < 0)
				break; // hit a filler

			if (row.height_ < 2.) {
				// a vertical gets represented with one bitmap row
				size_t xl = xmap[left.topWd_];
				size_t xr = xmap[right.topWd_];
				for (size_t i = xl; i < xr; i++)
					bitmap.at(r, i) = 1.;
			} else {
				// a sloped drawing gets represented by 2 bitmap rows

				// The top part as it starts
				size_t txl = xmap[left.topWd_];
				size_t txr = xmap[right.topWd_];
				for (size_t i = txl; i < txr; i++)
					bitmap.at(r, i) = 1.;

				// The bottom part as it ends
				size_t bxl = xmap[left.bottomWd_];
				size_t bxr = xmap[right.bottomWd_];
				for (size_t i = bxl; i < bxr; i++)
					bitmap.at(r + 1, i) = 1.;

				// Connect any missing pixels in the middle
				if (txr < bxl) {
					// sloping to the right
					while (txr < bxl) {
						if (txr + 1 == bxl) {
							// Make the overlap at half- brightness
							bitmap.at(r, txr++) = 0.;
							bitmap.at(r + 1, --bxl) = 0.;
						} else {
							bitmap.at(r, txr++) = 1.;
							bitmap.at(r + 1, --bxl) = 1.;
						}
					}
				} else {
					// sloping to the left
					while (bxr < txl) {
						if (bxr + 1 == txl) {
							// Make the overlap at half- brightness
							bitmap.at(r, --txl) = 0.;
							bitmap.at(r + 1, bxr++) = 0.;
						} else {
							bitmap.at(r, --txl) = 1.;
							bitmap.at(r + 1, bxr++) = 1.;
						}
					}
				}
			}
		}

		r += (size_t) row.height_;
	}

#if 0
	// Print the intermediate result of conversion
	extern void printTrapezeGlyphDigit(const TrapezeGlyph &glyph);
	printf("=======\n");
	printf("%d\n", glyph.label_);
	printf("   --- %zd x %zd\n", glyph.maxWidth_, glyph.rows_.size());
	printTrapezeGlyphDigit(glyph);
	printf("   --- %zd x %zd\n", wd1, ht1);
	for (int row = 0; row < ht1; row++) {
		printf ("  ");
		for (int col = 0; col < wd1; col++) {
			// rescale from [-1, 1] to [0, 999]
			int v = (int)((bitmap.at(row, col) + 1.) * (999./2.));
			if (v == 0) {
				printf("  . ");
			} else {
				printf("%3d ", v);
			}
		}
		printf("\n");
	}
#endif

	// Shrink horizontally if needed by merging the most similar columns
	size_t wds = wd1;
	if (wds > wd) {
		while (wds > wd) {
			double diff = 1e10; // difference between the best column pair
			double draw = 0.; // drawn pixels in the best column pair
			size_t rmc = 0; // column to merge with the next one
			for (size_t c = 0; c < wds - 1; c++) {
				double xdiff = 0.;
				double xdraw = 0.;
				for (size_t r = 0; r < ht1; r++) {
					xdiff += abs(bitmap.at(r, c) - bitmap.at(r, c+1));
					xdraw += bitmap.at(r, c) + bitmap.at(r, c+1) + 2.;
				}
				if (xdiff > 0) {
					xdiff /= xdraw;
				}
				if (xdiff < diff
				|| (xdiff == diff && xdraw > draw)) {
					diff = xdiff;
					draw = xdraw;
					rmc = c;
				}
			}

			// Remove the column
			for (size_t r = 0; r < ht1; r++) {
				bitmap.at(r, rmc) = (bitmap.at(r, rmc) + bitmap.at(r, rmc + 1)) / 2.;
				for (size_t c = rmc + 2; c < wds; c++) {
					bitmap.at(r, c - 1) = bitmap.at(r, c);
				}
			}
			wds--;
		}
#if 0
		// Print the intermediate result of column removal
		printf("   --- %zd x %zd\n", wds, ht1);
		for (int row = 0; row < ht1; row++) {
			printf ("  ");
			for (int col = 0; col < wds; col++) {
				// rescale from [-1, 1] to [0, 999]
				int v = (int)((bitmap.at(row, col) + 1.) * (999./2.));
				if (v == 0) {
					printf("  . ");
				} else {
					printf("%3d ", v);
				}
			}
			printf("\n");
		}
#endif
	}

	// Shrink vertically if needed by merging the most similar rows
	size_t hts = ht1;
	if (hts > ht) {
		while (hts > ht) {
			double diff = 1e10; // difference between the best row pair
			double draw = 0.; // drawn pixels in the best row pair
			size_t rmr = 0; // row to merge with the next one
			for (size_t r = 0; r < hts - 1; r++) {
				double xdiff = 0.;
				double xdraw = 0.;
				for (size_t c = 0; c < wds; c++) {
					xdiff += abs(bitmap.at(r, c) - bitmap.at(r+1, c));
					xdraw += bitmap.at(r, c) + bitmap.at(r+1, c) + 2.;
				}
				if (xdiff > 0) {
					xdiff /= xdraw;
				}
				if (xdiff < diff
				|| (xdiff == diff && xdraw > draw)) {
					diff = xdiff;
					draw = xdraw;
					rmr = r;
				}
			}

			// Remove the row
			for (size_t c = 0; c < wds; c++) {
				bitmap.at(rmr, c) = (bitmap.at(rmr, c) + bitmap.at(rmr + 1, c)) / 2.;
				for (size_t r = rmr + 2; r < hts; r++) {
					bitmap.at(r - 1, c) = bitmap.at(r, c);
				}
			}
			hts--;
		}
#if 0
		// Print the intermediate result of row removal
		printf("   --- %zd x %zd\n", wds, hts);
		for (int row = 0; row < hts; row++) {
			printf ("  ");
			for (int col = 0; col < wds; col++) {
				// rescale from [-1, 1] to [0, 999]
				int v = (int)((bitmap.at(row, col) + 1.) * (999./2.));
				if (v == 0) {
					printf("  . ");
				} else {
					printf("%3d ", v);
				}
			}
			printf("\n");
		}
#endif
	}

	// Finally, return the training bitmap
	shared_ptr<Training> t = make_shared<Training>(wd * ht, 10);
	FloatNeuralNet::ValueSubMatrix input(t->input, 0, ht, wd);

	for (size_t r = 0; r < ht; r++) {
		for (size_t c = 0; c < wd; c++) {
			if (r >= hts || c >= wds)
				input.at(r, c) = -1.;
			else
				input.at(r, c) = bitmap.at(r, c);
		}
	}

	if (prototype != nullptr) {
		// Copy the rest from prototype
		t->output = prototype->output;
		t->label = prototype->label;
		t->count = prototype->count;
		t->effectiveCount = prototype->effectiveCount;
	} else {
		t->output[glyph.label_] = 1.;
		t->label = glyph.label_;
		t->count = 1;
		t->effectiveCount = 1;
	}

	return t;
}


// Print one digit from its point weights.
void printDigit(const Training &tr)
{
	int limit = 8;
	if (tr.input.size() == 64) {
		// nothing to do
	} else if (tr.input.size() == 256) {
		limit = 16;
	} else {
		printf("  Unsupported input size %zd\n", tr.input.size());
		return;
	}

	for (int row = 0; row < limit; row++) {
		printf ("  ");
		for (int col = 0; col < limit; col++) {
			// rescale from [-1, 1] to [0, 999]
			int v = (int)((tr.input[row * limit + col] + 1.) * (999./2.));
			if (v == 0) {
				printf("  . ");
			} else {
				printf("%3d ", v);
			}
		}
		printf("\n");
	}
}

// Print one digit from its run-length encoding.
void printRunLenDigit(const Training &tr)
{
	int limit = 8;
	if (tr.input.size() == 64) {
		// nothing to do
	} else if (tr.input.size() == 256) {
		limit = 16;
	} else {
		printf("  Unsupported input size %zd\n", tr.input.size());
		return;
	}

	for (int row = 0; row < limit; row++) {
		printf ("  ");
		bool space = true;
		for (int col = 0; col < limit; col++) {
			int v = (int)tr.input[row * limit + col];
			for (int i = 0; i < v; i++) {
				printf(space ? "  . " : "999 ");
			}
			space = !space;
		}
		printf("\n");
	}
}

// Print one digit from its trapeze encoding, assuming that it's in whole pixels
void printTrapezeGlyphDigit(const TrapezeGlyph &glyph)
{
	for (size_t gr = 0; gr < glyph.rows_.size(); gr++) {
		const TrapezeRow &row = glyph.rows_[gr];

		if (row.height_ < 0.) // skip the filler 
			break;

		// printf("+ %f %zd\n", row.height_, row.widths_.size());
		for (size_t r = 0; r < row.height_; r++) {
			// Proportions of bottom and top width for this pixel row.
			double bscale = row.height_ <= 1. ? 1. : (r / (row.height_ - 1.));
			double tscale = 1. - bscale;

			int px = 0;
			double end = 0.;

			bool space = true;

			if (r == 0)
				printf (" ^");
			else
				printf ("  ");

			for (auto it = row.widths_.begin(); it != row.widths_.end(); ++it, space = !space) {
				if (it->topWd_ < 0.) // skip the filler 
					break;

				end += it->topWd_ * tscale + it->bottomWd_ * bscale;
				for (; px < end - 0.5; px++) {
					printf(space ? "  . " : "999 ");
				}
			}

			if (r == 0) {
				printf("  ");
				// print the slopes
				for (auto it = row.widths_.begin(); it != row.widths_.end(); ++it, space = !space) {
					printf(" %5.2f", it->leftSlope_);
				}
			}

			printf("\n");
		}
	}
}

enum TrainingFormat {
	TF_16X16, // as-loaded 16x16
	TF_8X8, // shrunk down to 8x8
	TF_16X16_BW, // 16x16 converted to black & white
	TF_8X8_BW, // shrunk down to 8x8 converted to black & white
	TF_16X16_RL, // 16x16 converted to black & white run-length
	TF_16X16_TRAPEZE, // 16x16 converted to black & white trapezes
	TF_16X16_TRAPEZE_ABS, // 16x16 converted to black & white trapezes, with absolute X coordinates
	TF_8X8_TRAPEZE_BITMAP, // 16x16 converted to black & white trapezes and then to a 8x8 bitmap
};

// Loads a gzipped training file.
static Erref 
loadZipFile(const string &gzname, TrainingVector &result, TrainingFormat format)
{
	// Since the inputs are < 1, producing 1 by their multiplication would be very difficult.
	// So can set a more limited target range that can be produced.
	// Since this would reduce gradients, increase the training rate by the same factor.
	constexpr double target = 1.; 

	Erref err;

	FILE *infile = popen(("gzip -d <" + gzname).c_str(), "r");
	if (infile == NULL) {
		err.f("Failed to run 'gzip -d <%s': %s.", gzname.c_str(), strerror(errno));
		return err;
	}

	vector<shared_ptr<TrapezeGlyph>> glyphs;

	int entry = 1;
	int scanres;
	int xout;
	double pixel, fout;
	for (; (scanres = fscanf(infile, "%lf", &fout)) > 0; ++entry) {
		xout = (int)fout;
		if (xout < 0 || xout > 9 || xout != fout) {
			err.f("Failed to read from ungzipped %s: entry %d contains output %f, should be 0..9.", gzname.c_str(), entry, fout); 
			break; // to pclose()
		}

		shared_ptr<Training> t = make_shared<Training>(256, 10);
		t->label = xout;
		t->output.assign(10,  -target);
		t->output[xout] = target;

		for (int i = 0; i < 256; i++) {
			scanres = fscanf(infile, "%lf", &pixel);
			if (scanres < 0) {
				err.f("Failed to read from ungzipped %s at entry %d input %d: %s.", gzname.c_str(), entry, i+1, strerror(errno));
				pclose(infile);
				return err;
			}
			if (pixel < -1. || pixel > 1.) {
				err.f("Failed to read from ungzipped %s: entry %d input %d contains %f, should be -1..1.",
					gzname.c_str(), entry, i+1, pixel); 
				pclose(infile);
				return err;
			}
			t->input[i] = pixel;
		}

		switch (format) {
		case TF_16X16:
			result.push_back(t);
			break;
		case TF_8X8:
			shrinkImage(*t);
			result.push_back(t);
			break;
		case TF_16X16_BW:
			bwImage(*t);
			result.push_back(t);
			break;
		case TF_8X8_BW:
			shrinkImage(*t);
			bwImage(*t);
			result.push_back(t);
			break;
		case TF_16X16_RL:
			bwImage(*t);
			runLenImage(*t);
			result.push_back(t);
			break;
		case TF_16X16_TRAPEZE:
		case TF_16X16_TRAPEZE_ABS:
			{
				auto gl = trapezeGlyphImage(*t);
#if 0
				// Print the glyphs as they are generated.
				printf("=======\n");
				printf("%d\n", gl->label_);
				printDigit(*t);
				printf("   --- %zd x %zd\n", gl->maxWidth_, gl->rows_.size());
				printTrapezeGlyphDigit(*gl);
#endif
				glyphs.emplace_back(std::move(gl));
			}
			break;
		case TF_8X8_TRAPEZE_BITMAP:
			{
				auto gl = trapezeGlyphImage(*t);
				t = trapezeGlyphToBitmap(*gl, 8, 8);
				result.push_back(t);
			}
			break;
		}
	}

	if (format == TF_16X16_TRAPEZE) {
		size_t maxht = 0;
		size_t maxwd = 0;
		for (auto it = glyphs.begin(); it != glyphs.end(); ++it) {
			if ((*it)->rows_.size() > maxht)
				maxht = (*it)->rows_.size();
			if ((*it)->maxWidth_ > maxwd)
				maxwd = (*it)->maxWidth_;
		}
		printf("Trapeze format dimensions %zd x %zd\n", maxwd, maxht);

		for (auto it = glyphs.begin(); it != glyphs.end(); ++it) {
			auto t = trapezeGlyphToTraining(false, **it, maxwd, maxht);
			t->output.assign(10,  -target);
			t->output[t->label] = target;
			result.push_back(t);

#if 0
			// Print the glyphs by restoring them back from training data.
			printf("=======\n");
			shared_ptr<TrapezeGlyph> gl = trainingToTrapezeGlyph(false, *t, maxwd, maxht);
			printf("%d\n", gl->label_);
			printf("   --- %zd x %zd\n", gl->maxWidth_, gl->rows_.size());
			printTrapezeGlyphDigit(*gl);
#endif
		}
	} else if (format == TF_16X16_TRAPEZE_ABS) {
		size_t maxht = 0;
		size_t maxwd = 0;
		for (auto it = glyphs.begin(); it != glyphs.end(); ++it) {
			if ((*it)->rows_.size() > maxht)
				maxht = (*it)->rows_.size();
			if ((*it)->maxWidth_ > maxwd)
				maxwd = (*it)->maxWidth_;
		}
		printf("Trapeze format dimensions %zd x %zd\n", maxwd, maxht);

		for (auto it = glyphs.begin(); it != glyphs.end(); ++it) {
			auto t = trapezeGlyphToTraining(true, **it, maxwd, maxht);
			t->output.assign(10,  -target);
			t->output[t->label] = target;
			result.push_back(t);

#if 0
			// Print the glyphs by restoring them back from training data.
			printf("=======\n");
			shared_ptr<TrapezeGlyph> gl = trainingToTrapezeGlyph(true, *t, maxwd, maxht);
			printf("%d\n", gl->label_);
			printf("   --- %zd x %zd\n", gl->maxWidth_, gl->rows_.size());
			printTrapezeGlyphDigit(*gl);
#endif
		}
	}

	if (scanres < 0 && !feof(infile)) {
		err.f("Failed to read from ungzipped %s at entry %d: %s.", gzname.c_str(), entry, strerror(errno));
		// Fall through to pclose().
	}
	pclose(infile);
	return err;
}

// Find the cases that have failed and print the images from them.
void
printFailures(Utest *utest, FloatNeuralNet &nn, const TrainingVector &vec)
{
	FloatNeuralNet::ValueVector outs;
	for (size_t elem = 0; elem < vec.size(); ++elem) {
		size_t highest;
		if (UT_NOERROR(nn.compute(vec[elem]->input, outs, highest)))
			return;
		if (vec[elem]->output[highest] > 0.) {
			continue;
		}
		size_t correct = -1;
		for (size_t i = 0; i < outs.size(); i++) {
			if (vec[elem]->output[i] > 0) {
				correct = i;
				break;
			}
		}

		printf("%zd mistaken as %zd at index %zd\n", correct, highest, elem);

		// print the computed weights
		printf("  -> ");
		for (size_t i = 0; i < outs.size(); i++) {
			printf("%zd:%4.2f ", i, outs[i]);
		}
		printf("\n");
		printDigit(*vec[elem]);
		printf("\n");
	}
}

// Print all the cases
void
printCases(Utest *utest, const TrainingVector &vec)
{
	// Group by the label.
	for (int label = 0; label < 10; label++) {
		for (auto it = vec.begin(); it != vec.end(); ++it) {
			if ((*it)->label != label)
				continue;
			printf("%d\n", label);
			printDigit(**it);
		}
		printf("---\n");
	}
}

// Print all the cases from the run length encoding
void
printRunLenCases(Utest *utest, const TrainingVector &vec)
{
	// Group by the label.
	for (int label = 0; label < 10; label++) {
		for (auto it = vec.begin(); it != vec.end(); ++it) {
			if ((*it)->label != label)
				continue;
			printf("%d\n", label);
			printRunLenDigit(**it);
		}
		printf("---\n");
	}
}

UTESTCASE zipcodes(Utest *utest)
{
	Erref err;
	
	string fpath = findZipFiles();
	if (fpath.empty()) {
		fprintf(stderr, "Could not find the file zipnn/zip.train.gz in current directory or its ancestors.\n");
		fprintf(stderr, "Dowload the files zip.train.gz and zip.test.gz from\n");
		fprintf(stderr, "https://hastie.su.domains/StatLearnSparsity_files/DATA/zipcode.html\n");
		fprintf(stderr, "and put them into the directory 'zipnn' next to Triceps.\n");
		UT_ASSERT(false);
		return;
	}

	// bool debug = true;  // enable extra printouts
	long seed = random_seed;

	FloatNeuralNet::Options options;

	// --- settings {

	options.isClassifier_ = true;
	options.maxMultiplier_ = 100;
	options.weightSaturation_ = 1.;
	options.momentum_ = true;
	options.autoRate2_ = true;
	// options.enableWeightFloor_ = true;
	options.tweakRate_ = 0.3;
#if 0 
	options.autoRate_ = true;
	// initial value
	options.trainingRate_ = 1e-6;
	// options.scaleRatePerLayer_ = true;
	// Also see the adjustment in trainSquareFunction()
	options.trainingRateScale_ = 0.01; // 5e-4;
#endif

	TrainingFormat format = TF_8X8_TRAPEZE_BITMAP;
	FloatNeuralNet::ActivationFunction activation = FloatNeuralNet::CORNER;

	// From the internets, 3 levels should be enough for RELU, and 2 layers
	// seem to work OK for CORNER.
	// The first element gets auto-filled.
	FloatNeuralNet::LevelSizeVector levels = {0, 256, 64, 10};

	string checkpoint = fpath + "/corner_0_256_64_10_nofl_trb.ckp";
	// batch > 1 doesn't mix with options.isClassifier_
	int batchSize = 1;
	seed = 1667553859;

	double rate = 0.0001;
	double trainRate = 0.;
	double applyRate = rate;

	if (0) { 
		trainRate = rate;
		applyRate = 0.;
	}

	// --- } settings

	TrainingVector testVec;
	testVec.reserve(2007);
	if (UT_NOERROR(loadZipFile(fpath + "/zip.test.gz", testVec, format))) {
		return;
	}

	TrainingVector trainVec;
	trainVec.reserve(7291);
	if (UT_NOERROR(loadZipFile(fpath + "/zip.train.gz", trainVec, format))) {
		return;
	}

#if 0
	// Include the test vector into train vector, to test the theory that
	// the test contains shapes that aren't represented in the training vector.
	trainVec.reserve(trainVec.size() + testVec.size());
	for (auto it = testVec.begin(); it != testVec.end(); it++) {
		trainVec.push_back(*it);
	}
#endif

#if 0
	// print the training vector
	switch (format) {
	case TF_16X16:
	case TF_8X8:
	case TF_16X16_BW:
	case TF_8X8_BW:
	case TF_8X8_TRAPEZE_BITMAP:
		printCases(utest, trainVec);
		printCases(utest, testVec);
		break;
	case TF_16X16_RL:
		printRunLenCases(utest, trainVec);
		printRunLenCases(utest, testVec);
		break;
	case TF_16X16_TRAPEZE:
	case TF_16X16_TRAPEZE_ABS:
		printf("The trapeze vector should be printed from loadZipFile()\n");
		break;
	}
	return;
#endif

	// Auto-fill the input size
	levels[0] = trainVec[0]->input.size();

	TestFloatNn nn(levels, activation, &options);
	UT_NOERROR(nn.getErrors());

	// Do the training.

	printf("Seed: %ld\n", seed);
	srand48(seed);
	nn.randomize();

	printf("Checkpointing to %s\n", checkpoint.c_str());
	{
		struct stat st;
		if (stat(checkpoint.c_str(), &st) == 0) {
			printf("Restoring checkpoint from %s\n", checkpoint.c_str());
			if (UT_NOERROR(nn.uncheckpoint(checkpoint))) {
				return;
			}
			printf("Checkpoint restored after pass %zd\n", (size_t)nn.getTrainingPass());
			// nn.printSimpleDump(utest);

#if 0 // {
			switch (format) {
			case TF_16X16:
			case TF_8X8:
			case TF_16X16_BW:
			case TF_8X8_BW:
			case TF_8X8_TRAPEZE_BITMAP:
				printFailures(utest, nn, testVec);
				break;
			case TF_16X16_RL:
			case TF_16X16_TRAPEZE:
			case TF_16X16_TRAPEZE_ABS:
				printf("Failure printing in this format is not supported yet\n");
				break;
			}
			return;
#endif // }
		}
	}

	int nPasses = 20000;
	int printEvery = 1; // nPasses / 10;
	int testEvery = printEvery;

	nn.trainClassifier(utest, checkpoint, trainVec, testVec, nPasses, printEvery, testEvery,
		trainRate, applyRate, /*reclaim*/ true, batchSize);

#if 0
	trainRate *= 1.;
	applyRate *= 1.;
	nn.trainClassifier(utest, trainVec, testVec, nPasses, printEvery, testEvery, trainRate, applyRate, /*reclaim*/ true);
#endif
}


/*

Some examples

Full size 16x16:

{
	FloatNeuralNet::LevelSizeVector levels = {256, 256, 256, 10};
	options.weightSaturation_ = 1.;
	TestFloatNn nn(levels, FloatNeuralNet::LEAKY_RELU, &options);
	seed = 1667553859;
	double rate = 0.01;
	double trainRate = 0.;
	double applyRate = rate;

      pass    0, meansq err 1.011812, gradient last 89706.368773 all 117633.340051
         p    0, test: meansq err 0.891919, error rate 0.885899
      pass    1, meansq err 0.890724, gradient last 73222.826513 all 98125.573742
         p    1, test: meansq err 0.788324, error rate 0.865471
      pass    2, meansq err 0.785993, gradient last 64408.244236 all 92235.019823
         p    2, test: meansq err 0.689150, error rate 0.841555
      pass    3, meansq err 0.685812, gradient last 49866.620848 all 74826.295823
         p    3, test: meansq err 0.623963, error rate 0.712506
      pass    4, meansq err 0.620786, gradient last 28143.998873 all 43472.936329
         p    4, test: meansq err 0.601881, error rate 0.705531
      pass    5, meansq err 0.599590, gradient last 12898.380646 all 22542.537048
         p    5, test: meansq err 0.594446, error rate 0.687095
      pass    6, meansq err 0.592461, gradient last 9417.459012 all 19025.985065
         p    6, test: meansq err 0.588803, error rate 0.673642
      pass    7, meansq err 0.586825, gradient last 8802.896698 all 18245.258820
         p    7, test: meansq err 0.583552, error rate 0.655207
      pass    8, meansq err 0.581543, gradient last 8574.750912 all 17751.384177
         p    8, test: meansq err 0.578539, error rate 0.638266
      pass    9, meansq err 0.576487, gradient last 8423.882130 all 17326.196376
         p    9, test: meansq err 0.573730, error rate 0.622820
      pass   10, meansq err 0.571624, gradient last 8297.175794 all 16932.518737
         p   10, test: meansq err 0.569110, error rate 0.603886
      pass   11, meansq err 0.566941, gradient last 8182.224023 all 16554.648772
         p   11, test: meansq err 0.564671, error rate 0.586946
      pass   12, meansq err 0.562426, gradient last 8074.826443 all 16197.809981
         p   12, test: meansq err 0.560400, error rate 0.573493
      pass   13, meansq err 0.558070, gradient last 7973.952357 all 15850.134649
         p   13, test: meansq err 0.556289, error rate 0.558047
      pass   14, meansq err 0.553865, gradient last 7877.319113 all 15520.083833
         p   14, test: meansq err 0.552330, error rate 0.542601
      pass   15, meansq err 0.549800, gradient last 7783.464443 all 15220.753705
         p   15, test: meansq err 0.548508, error rate 0.524664
      pass   16, meansq err 0.545860, gradient last 7691.557969 all 14939.177355
         p   16, test: meansq err 0.544814, error rate 0.509716
      pass   17, meansq err 0.542037, gradient last 7600.422950 all 14666.843576
         p   17, test: meansq err 0.541241, error rate 0.497260
      pass   18, meansq err 0.538325, gradient last 7509.101724 all 14405.734751
         p   18, test: meansq err 0.537782, error rate 0.486298
      pass   19, meansq err 0.534719, gradient last 7417.646455 all 14154.897708
         p   19, test: meansq err 0.534431, error rate 0.471849
      pass   20, meansq err 0.531214, gradient last 7325.366357 all 13914.937720
         p   20, test: meansq err 0.531186, error rate 0.462382
      pass   21, meansq err 0.527804, gradient last 7232.286745 all 13679.078937
         p   21, test: meansq err 0.528041, error rate 0.450922
      pass   22, meansq err 0.524487, gradient last 7138.782495 all 13451.496850
         p   22, test: meansq err 0.524992, error rate 0.439960
      pass   23, meansq err 0.521259, gradient last 7045.136798 all 13231.855216
         p   23, test: meansq err 0.522033, error rate 0.429497
      pass   24, meansq err 0.518115, gradient last 6951.543626 all 13021.858059
         p   24, test: meansq err 0.519161, error rate 0.419033
      pass   25, meansq err 0.515051, gradient last 6858.605941 all 12820.812916
         p   25, test: meansq err 0.516369, error rate 0.412058
      pass   26, meansq err 0.512064, gradient last 6766.806267 all 12627.244774
         p   26, test: meansq err 0.513654, error rate 0.406577
      pass   27, meansq err 0.509149, gradient last 6676.440083 all 12440.982021
         p   27, test: meansq err 0.511014, error rate 0.398107
      pass   28, meansq err 0.506302, gradient last 6588.022054 all 12259.838772
         p   28, test: meansq err 0.508443, error rate 0.392626
      pass   29, meansq err 0.503522, gradient last 6501.684924 all 12089.870291
         p   29, test: meansq err 0.505939, error rate 0.386647
      pass   30, meansq err 0.500804, gradient last 6417.728213 all 11920.413303
         p   30, test: meansq err 0.503500, error rate 0.380668
      pass   31, meansq err 0.498146, gradient last 6336.527593 all 11762.183233
         p   31, test: meansq err 0.501121, error rate 0.375685
      pass   32, meansq err 0.495544, gradient last 6258.271554 all 11612.667785
         p   32, test: meansq err 0.498796, error rate 0.370204
      pass   33, meansq err 0.492994, gradient last 6182.641707 all 11464.579255
         p   33, test: meansq err 0.496523, error rate 0.364225
      pass   34, meansq err 0.490495, gradient last 6109.761101 all 11327.085149
         p   34, test: meansq err 0.494298, error rate 0.360737
      pass   35, meansq err 0.488043, gradient last 6039.451280 all 11197.321676
         p   35, test: meansq err 0.492119, error rate 0.353762
      pass   36, meansq err 0.485635, gradient last 5971.909856 all 11071.045727
         p   36, test: meansq err 0.489984, error rate 0.348281
      pass   37, meansq err 0.483268, gradient last 5906.685529 all 10949.507665
         p   37, test: meansq err 0.487890, error rate 0.343797
      pass   38, meansq err 0.480942, gradient last 5843.721807 all 10830.791430
         p   38, test: meansq err 0.485835, error rate 0.338316
      pass   39, meansq err 0.478655, gradient last 5782.785294 all 10717.445826
         p   39, test: meansq err 0.483816, error rate 0.334330
      pass   40, meansq err 0.476404, gradient last 5723.898179 all 10605.506187
         p   40, test: meansq err 0.481836, error rate 0.332337
      pass   41, meansq err 0.474190, gradient last 5666.671497 all 10499.831199
         p   41, test: meansq err 0.479890, error rate 0.326856
      pass   42, meansq err 0.472009, gradient last 5611.077175 all 10394.198798
         p   42, test: meansq err 0.477977, error rate 0.319880
      pass   43, meansq err 0.469862, gradient last 5556.894404 all 10291.650156
         p   43, test: meansq err 0.476097, error rate 0.315396
      pass   44, meansq err 0.467747, gradient last 5504.311259 all 10193.020931
         p   44, test: meansq err 0.474247, error rate 0.309417
      pass   45, meansq err 0.465664, gradient last 5452.981054 all 10093.335765
         p   45, test: meansq err 0.472427, error rate 0.305929
      pass   46, meansq err 0.463612, gradient last 5402.979475 all 9994.818138
         p   46, test: meansq err 0.470637, error rate 0.302441
      pass   47, meansq err 0.461590, gradient last 5353.851561 all 9902.267863
         p   47, test: meansq err 0.468873, error rate 0.295964
      pass   48, meansq err 0.459597, gradient last 5306.063071 all 9812.103084
         p   48, test: meansq err 0.467136, error rate 0.291978
      pass   49, meansq err 0.457631, gradient last 5259.204875 all 9723.437584
         p   49, test: meansq err 0.465425, error rate 0.288989
      pass   50, meansq err 0.455692, gradient last 5213.292724 all 9636.916355
         p   50, test: meansq err 0.463739, error rate 0.287992
      pass   51, meansq err 0.453780, gradient last 5168.115657 all 9552.545009
         p   51, test: meansq err 0.462077, error rate 0.282511
      pass   52, meansq err 0.451893, gradient last 5123.884511 all 9468.272452
         p   52, test: meansq err 0.460440, error rate 0.279023
      pass   53, meansq err 0.450031, gradient last 5080.418685 all 9385.451146
         p   53, test: meansq err 0.458827, error rate 0.276532
      pass   54, meansq err 0.448195, gradient last 5037.607597 all 9303.874992
         p   54, test: meansq err 0.457237, error rate 0.272546
      pass   55, meansq err 0.446382, gradient last 4995.545545 all 9227.182782
         p   55, test: meansq err 0.455670, error rate 0.268560
      pass   56, meansq err 0.444592, gradient last 4954.049817 all 9152.254384
         p   56, test: meansq err 0.454124, error rate 0.264076
      pass   57, meansq err 0.442824, gradient last 4913.291549 all 9076.053721
         p   57, test: meansq err 0.452599, error rate 0.259591
      pass   58, meansq err 0.441078, gradient last 4872.984897 all 9001.582341
         p   58, test: meansq err 0.451095, error rate 0.257100
      pass   59, meansq err 0.439353, gradient last 4833.229312 all 8930.373640
         p   59, test: meansq err 0.449610, error rate 0.253612
      pass   60, meansq err 0.437650, gradient last 4794.054729 all 8859.083043
         p   60, test: meansq err 0.448145, error rate 0.252118
      pass   61, meansq err 0.435967, gradient last 4755.219334 all 8787.735020
         p   61, test: meansq err 0.446700, error rate 0.248132
      pass   62, meansq err 0.434304, gradient last 4716.690684 all 8720.156004
         p   62, test: meansq err 0.445273, error rate 0.243647
      pass   63, meansq err 0.432660, gradient last 4678.695022 all 8650.422662
         p   63, test: meansq err 0.443865, error rate 0.240658
      pass   64, meansq err 0.431037, gradient last 4640.813018 all 8582.560344
         p   64, test: meansq err 0.442476, error rate 0.238166
      pass   65, meansq err 0.429433, gradient last 4603.192941 all 8516.430072
         p   65, test: meansq err 0.441104, error rate 0.235177
      pass   66, meansq err 0.427848, gradient last 4565.723360 all 8449.538111
         p   66, test: meansq err 0.439752, error rate 0.232686
      pass   67, meansq err 0.426282, gradient last 4528.594520 all 8382.439810
         p   67, test: meansq err 0.438417, error rate 0.231191
      pass   68, meansq err 0.424735, gradient last 4491.595776 all 8315.656291
         p   68, test: meansq err 0.437100, error rate 0.227703
      pass   69, meansq err 0.423207, gradient last 4454.566456 all 8249.543649
         p   69, test: meansq err 0.435799, error rate 0.226208
      pass   70, meansq err 0.421697, gradient last 4417.511232 all 8184.443056
         p   70, test: meansq err 0.434515, error rate 0.223717
      pass   71, meansq err 0.420206, gradient last 4380.652764 all 8119.491504
         p   71, test: meansq err 0.433248, error rate 0.221724
      pass   72, meansq err 0.418734, gradient last 4343.962475 all 8053.511187
         p   72, test: meansq err 0.431997, error rate 0.220727
      pass   73, meansq err 0.417281, gradient last 4307.155583 all 7990.184221
         p   73, test: meansq err 0.430762, error rate 0.219731
      pass   74, meansq err 0.415845, gradient last 4270.533980 all 7925.715437
         p   74, test: meansq err 0.429545, error rate 0.219233
      pass   75, meansq err 0.414427, gradient last 4233.841047 all 7861.940624
         p   75, test: meansq err 0.428345, error rate 0.216741
      pass   76, meansq err 0.413028, gradient last 4197.211315 all 7794.900451
         p   76, test: meansq err 0.427161, error rate 0.215247
      pass   77, meansq err 0.411648, gradient last 4160.359254 all 7731.797909
         p   77, test: meansq err 0.425995, error rate 0.214250
      pass   78, meansq err 0.410285, gradient last 4123.861516 all 7669.548553
         p   78, test: meansq err 0.424846, error rate 0.210762
      pass   79, meansq err 0.408940, gradient last 4087.279304 all 7604.668951
         p   79, test: meansq err 0.423712, error rate 0.209766
      pass   80, meansq err 0.407614, gradient last 4050.808482 all 7539.497985
         p   80, test: meansq err 0.422596, error rate 0.205780
      pass   81, meansq err 0.406306, gradient last 4014.558399 all 7476.018446
         p   81, test: meansq err 0.421495, error rate 0.204285
      pass   82, meansq err 0.405015, gradient last 3978.222734 all 7413.144661
         p   82, test: meansq err 0.420411, error rate 0.203787
      pass   83, meansq err 0.403742, gradient last 3942.055911 all 7349.699633
         p   83, test: meansq err 0.419342, error rate 0.203288
      pass   84, meansq err 0.402487, gradient last 3906.028162 all 7286.006178
         p   84, test: meansq err 0.418289, error rate 0.202292
      pass   85, meansq err 0.401250, gradient last 3870.060455 all 7224.060314
         p   85, test: meansq err 0.417252, error rate 0.200299
      pass   86, meansq err 0.400030, gradient last 3834.088850 all 7162.507001
         p   86, test: meansq err 0.416230, error rate 0.199302
      pass   87, meansq err 0.398827, gradient last 3798.420527 all 7100.821548
         p   87, test: meansq err 0.415224, error rate 0.197808
      pass   88, meansq err 0.397642, gradient last 3763.004452 all 7039.099385
         p   88, test: meansq err 0.414233, error rate 0.196811
      pass   89, meansq err 0.396473, gradient last 3727.568208 all 6976.578141
         p   89, test: meansq err 0.413257, error rate 0.194320
      pass   90, meansq err 0.395322, gradient last 3692.185111 all 6916.073543
         p   90, test: meansq err 0.412295, error rate 0.192327
      pass   91, meansq err 0.394187, gradient last 3657.089671 all 6858.223904
         p   91, test: meansq err 0.411348, error rate 0.192327
      pass   92, meansq err 0.393068, gradient last 3622.221440 all 6797.013842
         p   92, test: meansq err 0.410416, error rate 0.192327
      pass   93, meansq err 0.391966, gradient last 3587.646709 all 6735.976651
         p   93, test: meansq err 0.409498, error rate 0.191330
      pass   94, meansq err 0.390881, gradient last 3553.205348 all 6676.016994
         p   94, test: meansq err 0.408594, error rate 0.189836
      pass   95, meansq err 0.389811, gradient last 3518.947313 all 6617.231301
         p   95, test: meansq err 0.407704, error rate 0.187344
      pass   96, meansq err 0.388758, gradient last 3485.160523 all 6560.689394
         p   96, test: meansq err 0.406827, error rate 0.186348
      pass   97, meansq err 0.387720, gradient last 3451.740394 all 6502.823037
         p   97, test: meansq err 0.405964, error rate 0.184853
      pass   98, meansq err 0.386697, gradient last 3418.428673 all 6448.324867
         p   98, test: meansq err 0.405114, error rate 0.184355
      pass   99, meansq err 0.385689, gradient last 3385.487654 all 6391.001335
         p   99, test: meansq err 0.404277, error rate 0.183857
}

{
	FloatNeuralNet::LevelSizeVector levels = {256, 256, 256, 10};
	options.weightSaturation_ = 1.;
	TestFloatNn nn(levels, FloatNeuralNet::LEAKY_RELU, &options);
	seed = 1667553859;
	double rate = 0.01;
	double trainRate = rate;
	double applyRate = 0.;

      pass    0, meansq err 0.952484, gradient last 82440.501565 all 108538.328909
         p    0, test: meansq err 0.895539, error rate 0.884903
      pass    1, meansq err 0.837920, gradient last 72435.463584 all 100398.285352
         p    1, test: meansq err 0.783233, error rate 0.864474
      pass    2, meansq err 0.729083, gradient last 59774.722350 all 87372.207564
         p    2, test: meansq err 0.685501, error rate 0.831091
      pass    3, meansq err 0.649728, gradient last 41287.965603 all 62350.325459
         p    3, test: meansq err 0.628401, error rate 0.704534
      pass    4, meansq err 0.611755, gradient last 24152.568881 all 37744.888587
         p    4, test: meansq err 0.605261, error rate 0.694569
      pass    5, meansq err 0.597088, gradient last 14244.746738 all 24315.370791
         p    5, test: meansq err 0.595298, error rate 0.672646
      pass    6, meansq err 0.589575, gradient last 10358.246097 all 19782.634519
         p    6, test: meansq err 0.588809, error rate 0.660688
      pass    7, meansq err 0.583769, gradient last 9127.134260 all 18419.309225
         p    7, test: meansq err 0.583272, error rate 0.642750
      pass    8, meansq err 0.578465, gradient last 8695.972360 all 17804.231190
         p    8, test: meansq err 0.578120, error rate 0.626308
      pass    9, meansq err 0.573420, gradient last 8485.446619 all 17347.358205
         p    9, test: meansq err 0.573222, error rate 0.611858
      pass   10, meansq err 0.568580, gradient last 8342.406283 all 16935.398253
         p   10, test: meansq err 0.568535, error rate 0.594918
      pass   11, meansq err 0.563925, gradient last 8223.518789 all 16548.709986
         p   11, test: meansq err 0.564040, error rate 0.578974
      pass   12, meansq err 0.559438, gradient last 8115.178899 all 16193.321162
         p   12, test: meansq err 0.559716, error rate 0.563528
      pass   13, meansq err 0.555102, gradient last 8012.436225 all 15862.714936
         p   13, test: meansq err 0.555550, error rate 0.546089
      pass   14, meansq err 0.550907, gradient last 7912.890701 all 15549.790807
         p   14, test: meansq err 0.551531, error rate 0.531639
      pass   15, meansq err 0.546844, gradient last 7814.698082 all 15246.916610
         p   15, test: meansq err 0.547653, error rate 0.515695
      pass   16, meansq err 0.542906, gradient last 7717.391645 all 14963.269640
         p   16, test: meansq err 0.543906, error rate 0.500249
      pass   17, meansq err 0.539086, gradient last 7620.470260 all 14682.227151
         p   17, test: meansq err 0.540287, error rate 0.483807
      pass   18, meansq err 0.535380, gradient last 7522.950973 all 14414.973795
         p   18, test: meansq err 0.536788, error rate 0.471849
      pass   19, meansq err 0.531782, gradient last 7424.952601 all 14158.293706
         p   19, test: meansq err 0.533404, error rate 0.460389
      pass   20, meansq err 0.528288, gradient last 7326.655903 all 13907.060854
         p   20, test: meansq err 0.530129, error rate 0.449925
      pass   21, meansq err 0.524890, gradient last 7228.327760 all 13673.127405
         p   21, test: meansq err 0.526956, error rate 0.437469
      pass   22, meansq err 0.521587, gradient last 7130.193688 all 13439.519823
         p   22, test: meansq err 0.523883, error rate 0.429497
      pass   23, meansq err 0.518373, gradient last 7032.686788 all 13218.973905
         p   23, test: meansq err 0.520902, error rate 0.419532
      pass   24, meansq err 0.515244, gradient last 6936.287608 all 13003.772308
         p   24, test: meansq err 0.518009, error rate 0.409068
      pass   25, meansq err 0.512195, gradient last 6841.161277 all 12803.582836
         p   25, test: meansq err 0.515198, error rate 0.401594
      pass   26, meansq err 0.509221, gradient last 6747.811889 all 12608.779365
         p   26, test: meansq err 0.512464, error rate 0.395615
      pass   27, meansq err 0.506320, gradient last 6656.654939 all 12419.269498
         p   27, test: meansq err 0.509807, error rate 0.391629
      pass   28, meansq err 0.503489, gradient last 6567.816207 all 12239.045116
         p   28, test: meansq err 0.507222, error rate 0.383657
      pass   29, meansq err 0.500721, gradient last 6481.603990 all 12067.435542
         p   29, test: meansq err 0.504704, error rate 0.379671
      pass   30, meansq err 0.498016, gradient last 6398.111032 all 11901.840095
         p   30, test: meansq err 0.502248, error rate 0.371699
      pass   31, meansq err 0.495368, gradient last 6317.367349 all 11746.538970
         p   31, test: meansq err 0.499850, error rate 0.367713
      pass   32, meansq err 0.492774, gradient last 6239.455139 all 11596.111616
         p   32, test: meansq err 0.497509, error rate 0.361734
      pass   33, meansq err 0.490232, gradient last 6164.402373 all 11454.235534
         p   33, test: meansq err 0.495220, error rate 0.355257
      pass   34, meansq err 0.487738, gradient last 6092.128221 all 11314.133910
         p   34, test: meansq err 0.492980, error rate 0.352765
      pass   35, meansq err 0.485292, gradient last 6022.474719 all 11180.164572
         p   35, test: meansq err 0.490788, error rate 0.345291
      pass   36, meansq err 0.482890, gradient last 5955.243517 all 11056.525083
         p   36, test: meansq err 0.488640, error rate 0.341804
      pass   37, meansq err 0.480530, gradient last 5890.349689 all 10929.802855
         p   37, test: meansq err 0.486535, error rate 0.334828
      pass   38, meansq err 0.478211, gradient last 5827.622594 all 10807.742511
         p   38, test: meansq err 0.484470, error rate 0.331340
      pass   39, meansq err 0.475933, gradient last 5766.880531 all 10690.944939
         p   39, test: meansq err 0.482444, error rate 0.326856
      pass   40, meansq err 0.473692, gradient last 5707.792224 all 10577.019852
         p   40, test: meansq err 0.480456, error rate 0.324365
      pass   41, meansq err 0.471489, gradient last 5650.359601 all 10465.501193
         p   41, test: meansq err 0.478502, error rate 0.320379
      pass   42, meansq err 0.469322, gradient last 5594.474451 all 10352.785949
         p   42, test: meansq err 0.476583, error rate 0.312905
      pass   43, meansq err 0.467190, gradient last 5540.149754 all 10248.920443
         p   43, test: meansq err 0.474698, error rate 0.308919
      pass   44, meansq err 0.465092, gradient last 5487.225636 all 10146.786752
         p   44, test: meansq err 0.472846, error rate 0.304933
      pass   45, meansq err 0.463024, gradient last 5435.548742 all 10050.954965
         p   45, test: meansq err 0.471023, error rate 0.298455
      pass   46, meansq err 0.460987, gradient last 5384.959868 all 9956.639952
         p   46, test: meansq err 0.469230, error rate 0.292476
      pass   47, meansq err 0.458979, gradient last 5335.554962 all 9860.987050
         p   47, test: meansq err 0.467465, error rate 0.288490
      pass   48, meansq err 0.457000, gradient last 5287.203083 all 9770.826916
         p   48, test: meansq err 0.465728, error rate 0.284006
      pass   49, meansq err 0.455048, gradient last 5239.789614 all 9681.506573
         p   49, test: meansq err 0.464017, error rate 0.282013
      pass   50, meansq err 0.453124, gradient last 5193.281499 all 9594.301822
         p   50, test: meansq err 0.462333, error rate 0.276532
      pass   51, meansq err 0.451226, gradient last 5147.666135 all 9509.009418
         p   51, test: meansq err 0.460674, error rate 0.275536
      pass   52, meansq err 0.449354, gradient last 5102.773271 all 9425.395603
         p   52, test: meansq err 0.459039, error rate 0.269058
      pass   53, meansq err 0.447506, gradient last 5058.640589 all 9342.038321
         p   53, test: meansq err 0.457430, error rate 0.266069
      pass   54, meansq err 0.445683, gradient last 5015.273175 all 9262.934538
         p   54, test: meansq err 0.455843, error rate 0.265072
      pass   55, meansq err 0.443884, gradient last 4972.595608 all 9183.131338
         p   55, test: meansq err 0.454279, error rate 0.262083
      pass   56, meansq err 0.442109, gradient last 4930.547279 all 9105.950181
         p   56, test: meansq err 0.452736, error rate 0.259591
      pass   57, meansq err 0.440355, gradient last 4889.011899 all 9031.040888
         p   57, test: meansq err 0.451214, error rate 0.254111
      pass   58, meansq err 0.438623, gradient last 4847.991272 all 8956.889839
         p   58, test: meansq err 0.449714, error rate 0.251619
      pass   59, meansq err 0.436913, gradient last 4807.479988 all 8883.186563
         p   59, test: meansq err 0.448234, error rate 0.249626
      pass   60, meansq err 0.435225, gradient last 4767.421621 all 8811.254346
         p   60, test: meansq err 0.446774, error rate 0.244145
      pass   61, meansq err 0.433557, gradient last 4727.714581 all 8739.393606
         p   61, test: meansq err 0.445333, error rate 0.241156
      pass   62, meansq err 0.431910, gradient last 4688.308269 all 8671.707918
         p   62, test: meansq err 0.443912, error rate 0.238166
      pass   63, meansq err 0.430282, gradient last 4649.208194 all 8600.128168
         p   63, test: meansq err 0.442510, error rate 0.236672
      pass   64, meansq err 0.428675, gradient last 4610.363351 all 8531.972444
         p   64, test: meansq err 0.441126, error rate 0.234180
      pass   65, meansq err 0.427087, gradient last 4571.670673 all 8463.693125
         p   65, test: meansq err 0.439761, error rate 0.233184
      pass   66, meansq err 0.425519, gradient last 4533.130683 all 8395.817248
         p   66, test: meansq err 0.438415, error rate 0.229198
      pass   67, meansq err 0.423971, gradient last 4494.731823 all 8328.942243
         p   67, test: meansq err 0.437085, error rate 0.227205
      pass   68, meansq err 0.422441, gradient last 4456.438053 all 8260.484000
         p   68, test: meansq err 0.435775, error rate 0.225212
      pass   69, meansq err 0.420931, gradient last 4418.234950 all 8193.179114
         p   69, test: meansq err 0.434482, error rate 0.222222
      pass   70, meansq err 0.419441, gradient last 4380.170286 all 8125.809148
         p   70, test: meansq err 0.433208, error rate 0.222720
      pass   71, meansq err 0.417970, gradient last 4342.175030 all 8058.981272
         p   71, test: meansq err 0.431951, error rate 0.222222
      pass   72, meansq err 0.416517, gradient last 4304.282587 all 7991.989021
         p   72, test: meansq err 0.430712, error rate 0.221226
      pass   73, meansq err 0.415085, gradient last 4266.479644 all 7922.411850
         p   73, test: meansq err 0.429491, error rate 0.219233
      pass   74, meansq err 0.413672, gradient last 4228.673394 all 7853.811260
         p   74, test: meansq err 0.428287, error rate 0.217738
      pass   75, meansq err 0.412279, gradient last 4190.951375 all 7787.922609
         p   75, test: meansq err 0.427100, error rate 0.215745
      pass   76, meansq err 0.410905, gradient last 4153.279932 all 7720.485072
         p   76, test: meansq err 0.425931, error rate 0.212755
      pass   77, meansq err 0.409549, gradient last 4115.646977 all 7654.571570
         p   77, test: meansq err 0.424779, error rate 0.209766
      pass   78, meansq err 0.408213, gradient last 4078.147362 all 7588.605637
         p   78, test: meansq err 0.423644, error rate 0.207275
      pass   79, meansq err 0.406895, gradient last 4040.711312 all 7524.142502
         p   79, test: meansq err 0.422526, error rate 0.204783
      pass   80, meansq err 0.405595, gradient last 4003.373019 all 7459.677715
         p   80, test: meansq err 0.421423, error rate 0.203288
      pass   81, meansq err 0.404314, gradient last 3966.190143 all 7394.425051
         p   81, test: meansq err 0.420338, error rate 0.202790
      pass   82, meansq err 0.403051, gradient last 3929.156759 all 7328.151574
         p   82, test: meansq err 0.419268, error rate 0.201295
      pass   83, meansq err 0.401806, gradient last 3892.231037 all 7264.649182
         p   83, test: meansq err 0.418216, error rate 0.199801
      pass   84, meansq err 0.400580, gradient last 3855.453666 all 7198.721780
         p   84, test: meansq err 0.417179, error rate 0.198804
      pass   85, meansq err 0.399372, gradient last 3818.861758 all 7135.169860
         p   85, test: meansq err 0.416159, error rate 0.197309
      pass   86, meansq err 0.398182, gradient last 3782.422698 all 7072.071045
         p   86, test: meansq err 0.415154, error rate 0.196313
      pass   87, meansq err 0.397009, gradient last 3746.167837 all 7008.098784
         p   87, test: meansq err 0.414165, error rate 0.194320
      pass   88, meansq err 0.395854, gradient last 3710.105060 all 6945.533416
         p   88, test: meansq err 0.413191, error rate 0.193323
      pass   89, meansq err 0.394716, gradient last 3674.223522 all 6883.001415
         p   89, test: meansq err 0.412232, error rate 0.192327
      pass   90, meansq err 0.393595, gradient last 3638.588970 all 6822.550237
         p   90, test: meansq err 0.411289, error rate 0.190832
      pass   91, meansq err 0.392491, gradient last 3603.201379 all 6763.363261
         p   91, test: meansq err 0.410360, error rate 0.188839
      pass   92, meansq err 0.391402, gradient last 3568.084801 all 6705.294200
         p   92, test: meansq err 0.409446, error rate 0.187843
      pass   93, meansq err 0.390329, gradient last 3533.237854 all 6647.390120
         p   93, test: meansq err 0.408545, error rate 0.186846
      pass   94, meansq err 0.389272, gradient last 3498.685758 all 6587.956241
         p   94, test: meansq err 0.407659, error rate 0.186846
      pass   95, meansq err 0.388232, gradient last 3464.394650 all 6528.706327
         p   95, test: meansq err 0.406786, error rate 0.184853
      pass   96, meansq err 0.387206, gradient last 3430.422526 all 6471.563566
         p   96, test: meansq err 0.405928, error rate 0.182860
      pass   97, meansq err 0.386196, gradient last 3396.762813 all 6414.609623
         p   97, test: meansq err 0.405082, error rate 0.182860
      pass   98, meansq err 0.385201, gradient last 3363.464135 all 6358.799962
         p   98, test: meansq err 0.404250, error rate 0.181863
      pass   99, meansq err 0.384221, gradient last 3330.493054 all 6303.017164
         p   99, test: meansq err 0.403430, error rate 0.179870
}

{
	FloatNeuralNet::LevelSizeVector levels = {256, 256, 256, 10};
	options.weightSaturation_ = 1.;
	TestFloatNn nn(levels, FloatNeuralNet::CORNER, &options);
	seed = 1667553859;
	double rate = 0.001;
	double trainRate = 0.;
	double applyRate = rate;

      pass    0, meansq err 1.037811, gradient last 71161.181078 all 95103.327095
         p    0, test: meansq err 1.030078, error rate 0.900847
      pass    1, meansq err 1.029797, gradient last 67480.040897 all 89221.119827
         p    1, test: meansq err 1.023048, error rate 0.900349
      pass    2, meansq err 1.022708, gradient last 63986.448189 all 83644.892474
         p    2, test: meansq err 1.016862, error rate 0.895366
      pass    3, meansq err 1.016470, gradient last 60347.114330 all 77747.348018
         p    3, test: meansq err 1.011524, error rate 0.893373
      pass    4, meansq err 1.011094, gradient last 56421.807161 all 71287.585322
         p    4, test: meansq err 1.007009, error rate 0.893373
      pass    5, meansq err 1.006564, gradient last 52939.682669 all 65439.645342
         p    5, test: meansq err 1.003180, error rate 0.892875
      pass    6, meansq err 1.002729, gradient last 49816.842316 all 60144.175832
         p    6, test: meansq err 0.999931, error rate 0.895366
      pass    7, meansq err 0.999493, gradient last 46964.884139 all 55232.828878
         p    7, test: meansq err 0.997164, error rate 0.895366
      pass    8, meansq err 0.996745, gradient last 44712.474346 all 51315.604973
         p    8, test: meansq err 0.994753, error rate 0.896861
      pass    9, meansq err 0.994352, gradient last 43066.992407 all 48428.456765
         p    9, test: meansq err 0.992590, error rate 0.899352
      pass   10, meansq err 0.992208, gradient last 41813.403671 all 46230.262830
         p   10, test: meansq err 0.990610, error rate 0.897857
      pass   11, meansq err 0.990246, gradient last 40789.738069 all 44416.112156
         p   11, test: meansq err 0.988771, error rate 0.898356
      pass   12, meansq err 0.988427, gradient last 39975.998068 all 42969.561248
         p   12, test: meansq err 0.987044, error rate 0.899851
      pass   13, meansq err 0.986717, gradient last 39316.137744 all 41815.658659
         p   13, test: meansq err 0.985401, error rate 0.900349
      pass   14, meansq err 0.985092, gradient last 38787.092087 all 40904.186862
         p   14, test: meansq err 0.983822, error rate 0.900847
      pass   15, meansq err 0.983531, gradient last 38377.524803 all 40220.846997
         p   15, test: meansq err 0.982294, error rate 0.898356
      pass   16, meansq err 0.982019, gradient last 38027.092495 all 39640.662205
         p   16, test: meansq err 0.980808, error rate 0.899352
      pass   17, meansq err 0.980546, gradient last 37735.735217 all 39170.047970
         p   17, test: meansq err 0.979355, error rate 0.900847
      pass   18, meansq err 0.979105, gradient last 37483.705402 all 38771.729860
         p   18, test: meansq err 0.977929, error rate 0.900847
      pass   19, meansq err 0.977691, gradient last 37258.226789 all 38417.854559
         p   19, test: meansq err 0.976527, error rate 0.898854
      pass   20, meansq err 0.976300, gradient last 37062.165468 all 38116.886780
         p   20, test: meansq err 0.975146, error rate 0.897359
      pass   21, meansq err 0.974928, gradient last 36886.243371 all 37852.467375
         p   21, test: meansq err 0.973784, error rate 0.895864
      pass   22, meansq err 0.973574, gradient last 36717.080514 all 37599.645715
         p   22, test: meansq err 0.972438, error rate 0.897359
      pass   23, meansq err 0.972235, gradient last 36567.587756 all 37382.227605
         p   23, test: meansq err 0.971107, error rate 0.895366
      pass   24, meansq err 0.970910, gradient last 36431.654305 all 37193.036369
         p   24, test: meansq err 0.969788, error rate 0.892875
      pass   25, meansq err 0.969596, gradient last 36295.267503 all 37000.213132
         p   25, test: meansq err 0.968480, error rate 0.889885
      pass   26, meansq err 0.968294, gradient last 36167.602574 all 36822.835262
         p   26, test: meansq err 0.967182, error rate 0.889387
      pass   27, meansq err 0.967003, gradient last 36047.784065 all 36660.617512
         p   27, test: meansq err 0.965894, error rate 0.888391
      pass   28, meansq err 0.965722, gradient last 35932.444519 all 36505.396847
         p   28, test: meansq err 0.964617, error rate 0.885899
      pass   29, meansq err 0.964450, gradient last 35822.002849 all 36358.669648
         p   29, test: meansq err 0.963348, error rate 0.885899
      pass   30, meansq err 0.963187, gradient last 35712.437972 all 36214.572400
         p   30, test: meansq err 0.962087, error rate 0.883906
      pass   31, meansq err 0.961931, gradient last 35612.638212 all 36087.174963
         p   31, test: meansq err 0.960833, error rate 0.879920
      pass   32, meansq err 0.960683, gradient last 35521.151420 all 35975.107287
         p   32, test: meansq err 0.959587, error rate 0.878426
      pass   33, meansq err 0.959442, gradient last 35431.832614 all 35866.082770
         p   33, test: meansq err 0.958347, error rate 0.876931
      pass   34, meansq err 0.958206, gradient last 35341.015787 all 35754.378093
         p   34, test: meansq err 0.957114, error rate 0.874439
      pass   35, meansq err 0.956977, gradient last 35250.101095 all 35642.539274
         p   35, test: meansq err 0.955887, error rate 0.873941
      pass   36, meansq err 0.955754, gradient last 35163.105908 all 35538.636113
         p   36, test: meansq err 0.954667, error rate 0.872446
      pass   37, meansq err 0.954536, gradient last 35077.152273 all 35436.039743
         p   37, test: meansq err 0.953452, error rate 0.870453
      pass   38, meansq err 0.953324, gradient last 34994.379703 all 35339.107098
         p   38, test: meansq err 0.952242, error rate 0.868460
      pass   39, meansq err 0.952118, gradient last 34912.084725 all 35243.230161
         p   39, test: meansq err 0.951038, error rate 0.866966
      pass   40, meansq err 0.950916, gradient last 34834.079045 all 35155.513591
         p   40, test: meansq err 0.949839, error rate 0.866966
      pass   41, meansq err 0.949720, gradient last 34752.809931 all 35062.152693
         p   41, test: meansq err 0.948645, error rate 0.865471
      pass   42, meansq err 0.948528, gradient last 34674.019724 all 34972.202662
         p   42, test: meansq err 0.947456, error rate 0.864474
      pass   43, meansq err 0.947341, gradient last 34596.112719 all 34884.296836
         p   43, test: meansq err 0.946272, error rate 0.862980
      pass   44, meansq err 0.946158, gradient last 34516.874711 all 34793.676179
         p   44, test: meansq err 0.945092, error rate 0.862481
      pass   45, meansq err 0.944981, gradient last 34441.459421 all 34710.117519
         p   45, test: meansq err 0.943917, error rate 0.861485
      pass   46, meansq err 0.943807, gradient last 34369.358691 all 34632.837355
         p   46, test: meansq err 0.942746, error rate 0.858994
      pass   47, meansq err 0.942638, gradient last 34295.764580 all 34552.508494
         p   47, test: meansq err 0.941579, error rate 0.857499
      pass   48, meansq err 0.941473, gradient last 34222.353765 all 34471.887870
         p   48, test: meansq err 0.940416, error rate 0.855506
      pass   49, meansq err 0.940312, gradient last 34147.886930 all 34389.752159
         p   49, test: meansq err 0.939258, error rate 0.855007
      pass   50, meansq err 0.939155, gradient last 34076.013916 all 34312.100431
         p   50, test: meansq err 0.938103, error rate 0.853513
      pass   51, meansq err 0.938002, gradient last 34004.590240 all 34234.955576
         p   51, test: meansq err 0.936952, error rate 0.853513
      pass   52, meansq err 0.936853, gradient last 33932.466847 all 34156.081360
         p   52, test: meansq err 0.935806, error rate 0.851520
      pass   53, meansq err 0.935708, gradient last 33862.916883 all 34082.436001
         p   53, test: meansq err 0.934663, error rate 0.851021
      pass   54, meansq err 0.934567, gradient last 33791.392592 all 34005.127961
         p   54, test: meansq err 0.933524, error rate 0.850523
      pass   55, meansq err 0.933430, gradient last 33722.950035 all 33933.270184
         p   55, test: meansq err 0.932389, error rate 0.849527
      pass   56, meansq err 0.932296, gradient last 33652.945425 all 33858.510117
         p   56, test: meansq err 0.931258, error rate 0.849028
      pass   57, meansq err 0.931166, gradient last 33582.266817 all 33782.074985
         p   57, test: meansq err 0.930130, error rate 0.849028
      pass   58, meansq err 0.930040, gradient last 33515.403047 all 33712.723857
         p   58, test: meansq err 0.929006, error rate 0.848032
      pass   59, meansq err 0.928918, gradient last 33446.574662 all 33639.986125
         p   59, test: meansq err 0.927886, error rate 0.848032
      pass   60, meansq err 0.927799, gradient last 33377.727125 all 33566.693353
         p   60, test: meansq err 0.926769, error rate 0.847534
      pass   61, meansq err 0.926684, gradient last 33310.537194 all 33496.799549
         p   61, test: meansq err 0.925656, error rate 0.844046
      pass   62, meansq err 0.925572, gradient last 33243.454803 all 33426.713165
         p   62, test: meansq err 0.924547, error rate 0.844046
      pass   63, meansq err 0.924464, gradient last 33176.689054 all 33357.514670
         p   63, test: meansq err 0.923440, error rate 0.844046
      pass   64, meansq err 0.923359, gradient last 33109.991643 all 33287.958821
         p   64, test: meansq err 0.922338, error rate 0.843049
      pass   65, meansq err 0.922257, gradient last 33043.301286 all 33218.635593
         p   65, test: meansq err 0.921239, error rate 0.843049
      pass   66, meansq err 0.921160, gradient last 32976.476299 all 33148.793051
         p   66, test: meansq err 0.920143, error rate 0.842053
      pass   67, meansq err 0.920065, gradient last 32910.078351 all 33079.453858
         p   67, test: meansq err 0.919051, error rate 0.840558
      pass   68, meansq err 0.918974, gradient last 32842.710885 all 33008.403144
         p   68, test: meansq err 0.917962, error rate 0.839562
      pass   69, meansq err 0.917887, gradient last 32775.987035 all 32938.144094
         p   69, test: meansq err 0.916877, error rate 0.839063
      pass   70, meansq err 0.916803, gradient last 32709.200764 all 32867.892740
         p   70, test: meansq err 0.915796, error rate 0.838067
      pass   71, meansq err 0.915722, gradient last 32644.582496 all 32801.627009
         p   71, test: meansq err 0.914717, error rate 0.836572
      pass   72, meansq err 0.914645, gradient last 32579.708992 all 32734.661153
         p   72, test: meansq err 0.913643, error rate 0.834579
      pass   73, meansq err 0.913571, gradient last 32514.773108 all 32667.626558
         p   73, test: meansq err 0.912571, error rate 0.835077
      pass   74, meansq err 0.912500, gradient last 32450.252711 all 32601.404404
         p   74, test: meansq err 0.911503, error rate 0.835575
      pass   75, meansq err 0.911433, gradient last 32385.727427 all 32534.784468
         p   75, test: meansq err 0.910438, error rate 0.834579
      pass   76, meansq err 0.910368, gradient last 32322.153442 all 32469.820795
         p   76, test: meansq err 0.909376, error rate 0.833582
      pass   77, meansq err 0.909307, gradient last 32258.506011 all 32404.758983
         p   77, test: meansq err 0.908317, error rate 0.833582
      pass   78, meansq err 0.908250, gradient last 32193.988523 all 32338.126696
         p   78, test: meansq err 0.907262, error rate 0.832586
      pass   79, meansq err 0.907195, gradient last 32129.000198 all 32270.477253
         p   79, test: meansq err 0.906210, error rate 0.831589
      pass   80, meansq err 0.906144, gradient last 32065.967297 all 32206.365308
         p   80, test: meansq err 0.905160, error rate 0.830593
      pass   81, meansq err 0.905096, gradient last 32002.761541 all 32141.749028
         p   81, test: meansq err 0.904115, error rate 0.830593
      pass   82, meansq err 0.904051, gradient last 31940.676253 all 32079.292275
         p   82, test: meansq err 0.903072, error rate 0.830095
      pass   83, meansq err 0.903009, gradient last 31878.049296 all 32015.665341
         p   83, test: meansq err 0.902032, error rate 0.830095
      pass   84, meansq err 0.901970, gradient last 31814.546125 all 31950.230230
         p   84, test: meansq err 0.900996, error rate 0.829596
      pass   85, meansq err 0.900934, gradient last 31752.462465 all 31887.588613
         p   85, test: meansq err 0.899962, error rate 0.829098
      pass   86, meansq err 0.899902, gradient last 31690.470189 all 31824.912066
         p   86, test: meansq err 0.898932, error rate 0.829098
      pass   87, meansq err 0.898872, gradient last 31627.971807 all 31761.053358
         p   87, test: meansq err 0.897905, error rate 0.829098
      pass   88, meansq err 0.897846, gradient last 31565.833535 all 31697.846056
         p   88, test: meansq err 0.896881, error rate 0.828600
      pass   89, meansq err 0.896822, gradient last 31504.382066 all 31635.716542
         p   89, test: meansq err 0.895859, error rate 0.828600
      pass   90, meansq err 0.895802, gradient last 31442.766032 all 31573.222237
         p   90, test: meansq err 0.894841, error rate 0.828102
      pass   91, meansq err 0.894785, gradient last 31380.588645 all 31509.687769
         p   91, test: meansq err 0.893826, error rate 0.827105
      pass   92, meansq err 0.893771, gradient last 31318.705363 all 31446.370422
         p   92, test: meansq err 0.892814, error rate 0.827105
      pass   93, meansq err 0.892759, gradient last 31256.614592 all 31382.832054
         p   93, test: meansq err 0.891805, error rate 0.827105
      pass   94, meansq err 0.891751, gradient last 31195.928612 all 31321.549022
         p   94, test: meansq err 0.890799, error rate 0.826607
      pass   95, meansq err 0.890746, gradient last 31136.029534 all 31261.736888
         p   95, test: meansq err 0.889796, error rate 0.826607
      pass   96, meansq err 0.889744, gradient last 31075.690882 all 31201.062523
         p   96, test: meansq err 0.888796, error rate 0.826607
      pass   97, meansq err 0.888745, gradient last 31015.326301 all 31140.338140
         p   97, test: meansq err 0.887799, error rate 0.826109
      pass   98, meansq err 0.887748, gradient last 30955.017653 all 31079.459440
         p   98, test: meansq err 0.886805, error rate 0.826607
      pass   99, meansq err 0.886755, gradient last 30895.298528 all 31019.748294
         p   99, test: meansq err 0.885813, error rate 0.826109
}

{
	FloatNeuralNet::LevelSizeVector levels = {256, 256*4, 10};
	options.weightSaturation_ = 1.;
	TestFloatNn nn(levels, FloatNeuralNet::CORNER, &options);
	seed = 1667553859;
	double rate = 0.001;
	double trainRate = 0.;
	double applyRate = rate;

      pass    0, meansq err 1.164340, gradient last 369773.484985 all 393296.150146
         p    0, test: meansq err 1.058735, error rate 0.918784
      pass    1, meansq err 1.058379, gradient last 224754.638923 all 238183.716439
         p    1, test: meansq err 1.020566, error rate 0.906328
      pass    2, meansq err 1.019735, gradient last 119300.190070 all 125665.514514
         p    2, test: meansq err 1.008428, error rate 0.889387
      pass    3, meansq err 1.007741, gradient last 81275.929810 all 85038.216326
         p    3, test: meansq err 1.002324, error rate 0.876432
      pass    4, meansq err 1.001814, gradient last 65514.239478 all 68089.112695
         p    4, test: meansq err 0.998301, error rate 0.867962
      pass    5, meansq err 0.997874, gradient last 57223.870990 all 59102.520508
         p    5, test: meansq err 0.995226, error rate 0.860987
      pass    6, meansq err 0.994873, gradient last 52142.432420 all 53562.738114
         p    6, test: meansq err 0.992697, error rate 0.852516
      pass    7, meansq err 0.992389, gradient last 48961.351361 all 50070.715436
         p    7, test: meansq err 0.990489, error rate 0.848032
      pass    8, meansq err 0.990218, gradient last 46864.020295 all 47753.102985
         p    8, test: meansq err 0.988487, error rate 0.845541
      pass    9, meansq err 0.988245, gradient last 45379.592038 all 46103.108940
         p    9, test: meansq err 0.986630, error rate 0.843049
      pass   10, meansq err 0.986409, gradient last 44345.025334 all 44949.922880
         p   10, test: meansq err 0.984876, error rate 0.835575
      pass   11, meansq err 0.984668, gradient last 43649.529902 all 44169.455250
         p   11, test: meansq err 0.983186, error rate 0.833582
      pass   12, meansq err 0.982990, gradient last 43177.982320 all 43637.382112
         p   12, test: meansq err 0.981547, error rate 0.827603
      pass   13, meansq err 0.981357, gradient last 42812.279159 all 43223.683871
         p   13, test: meansq err 0.979940, error rate 0.824614
      pass   14, meansq err 0.979759, gradient last 42509.282018 all 42880.745004
         p   14, test: meansq err 0.978364, error rate 0.822123
      pass   15, meansq err 0.978192, gradient last 42263.405647 all 42600.800817
         p   15, test: meansq err 0.976818, error rate 0.822123
      pass   16, meansq err 0.976648, gradient last 42094.098022 all 42408.032465
         p   16, test: meansq err 0.975290, error rate 0.822621
      pass   17, meansq err 0.975121, gradient last 41984.195962 all 42281.542624
         p   17, test: meansq err 0.973775, error rate 0.822621
      pass   18, meansq err 0.973607, gradient last 41873.956464 all 42155.655475
         p   18, test: meansq err 0.972271, error rate 0.820130
      pass   19, meansq err 0.972105, gradient last 41784.276969 all 42053.955137
         p   19, test: meansq err 0.970780, error rate 0.818137
      pass   20, meansq err 0.970614, gradient last 41694.895759 all 41952.131362
         p   20, test: meansq err 0.969298, error rate 0.818137
      pass   21, meansq err 0.969132, gradient last 41656.677135 all 41907.314816
         p   21, test: meansq err 0.967824, error rate 0.816143
      pass   22, meansq err 0.967655, gradient last 41625.612853 all 41872.738870
         p   22, test: meansq err 0.966353, error rate 0.813154
      pass   23, meansq err 0.966184, gradient last 41591.851513 all 41834.985256
         p   23, test: meansq err 0.964888, error rate 0.810663
      pass   24, meansq err 0.964717, gradient last 41571.128725 all 41812.863866
         p   24, test: meansq err 0.963425, error rate 0.810164
      pass   25, meansq err 0.963254, gradient last 41565.066043 all 41805.233819
         p   25, test: meansq err 0.961965, error rate 0.809168
      pass   26, meansq err 0.961794, gradient last 41556.271333 all 41794.864511
         p   26, test: meansq err 0.960507, error rate 0.809168
      pass   27, meansq err 0.960336, gradient last 41567.629371 all 41808.195458
         p   27, test: meansq err 0.959051, error rate 0.808670
      pass   28, meansq err 0.958879, gradient last 41576.305255 all 41820.199951
         p   28, test: meansq err 0.957595, error rate 0.807175
      pass   29, meansq err 0.957423, gradient last 41604.904120 all 41854.110536
         p   29, test: meansq err 0.956139, error rate 0.806677
      pass   30, meansq err 0.955966, gradient last 41634.760647 all 41889.990311
         p   30, test: meansq err 0.954681, error rate 0.807673
      pass   31, meansq err 0.954507, gradient last 41680.743429 all 41944.917286
         p   31, test: meansq err 0.953219, error rate 0.805182
}

{
	FloatNeuralNet::LevelSizeVector levels = {256, 256*4, 10};
	options.weightSaturation_ = 1.;
	options.autoRate_ = true;
	options.trainingRate_ = 1e-6;
	options.enableWeightFloor_ = true;
	options.trainingRateScale_ = 1e-4;
	TestFloatNn nn(levels, FloatNeuralNet::CORNER, &options);
	seed = 1667553859;

      pass    0, meansq err 1.164450, gradient last 369911.051889 all 393431.691825
         p    0, test: meansq err 0.921875, error rate 0.899352
DEBUG 1/L = 1.01324e-05, old rate = 1e-05, dx = 59982.8, dgrad = 591991
DEBUG bump the rate down to 0.000007
DEBUG restored weights for redo
DEBUG this is a redo of the last pass
      pass    0, meansq err 1.164450, gradient last 369911.069706 all 393431.709158
         p    0, test: meansq err 0.949880, error rate 0.914300
      pass    1, meansq err 0.949148, gradient last 322492.890404 all 322540.332689
         p    1, test: meansq err 2.478110, error rate 0.920279
DEBUG 1/L = 1.59644e-07, old rate = 6.75493e-06, dx = 2300.6, dgrad = 1.44108e+06
DEBUG bump the rate down to 0.000000
DEBUG restored weights for redo
DEBUG this is a redo of the last pass
      pass    1, meansq err 0.949148, gradient last 322492.890404 all 322540.332689
         p    1, test: meansq err 0.919607, error rate 0.920279
      pass    2, meansq err 0.918341, gradient last 302169.714410 all 302280.936286
         p    2, test: meansq err 0.893225, error rate 0.920279
DEBUG 1/L = 2.58491e-07, old rate = 1.06429e-07, dx = 54.1112, dgrad = 20933.5
DEBUG bump the rate up to 0.000000
      pass    3, meansq err 0.891508, gradient last 283580.029355 all 283763.397905
         p    3, test: meansq err 0.870098, error rate 0.920279
DEBUG 1/L = 3.59621e-07, old rate = 1.72327e-07, dx = 67.2548, dgrad = 18701.6
DEBUG bump the rate up to 0.000000
      pass    4, meansq err 0.867999, gradient last 267097.681490 all 267352.332938
         p    4, test: meansq err 0.837638, error rate 0.920279
DEBUG 1/L = 4.58101e-07, old rate = 2.39747e-07, dx = 124.397, dgrad = 27154.9
DEBUG bump the rate up to 0.000000
      pass    5, meansq err 0.835038, gradient last 243496.444631 all 243854.341004
         p    5, test: meansq err 0.801441, error rate 0.920279
DEBUG 1/L = 6.15981e-07, old rate = 3.05401e-07, dx = 195.612, dgrad = 31756.2
DEBUG bump the rate up to 0.000000
      pass    6, meansq err 0.798351, gradient last 216345.306817 all 216804.855719
         p    6, test: meansq err 0.767120, error rate 0.920279
DEBUG 1/L = 8.31568e-07, old rate = 4.10654e-07, dx = 265.179, dgrad = 31889
DEBUG bump the rate up to 0.000001
      pass    7, meansq err 0.763661, gradient last 189525.835693 all 190044.717034
         p    7, test: meansq err 0.734649, error rate 0.920279
DEBUG 1/L = 1.10705e-06, old rate = 5.54378e-07, dx = 352.337, dgrad = 31826.6
DEBUG bump the rate up to 0.000001
      pass    8, meansq err 0.730985, gradient last 163317.533744 all 163830.912194
         p    8, test: meansq err 0.705897, error rate 0.920279
DEBUG 1/L = 1.48673e-06, old rate = 7.38033e-07, dx = 433.112, dgrad = 29131.8
DEBUG bump the rate up to 0.000001
      pass    9, meansq err 0.702231, gradient last 139848.562881 all 140289.924621
         p    9, test: meansq err 0.682074, error rate 0.920279
DEBUG 1/L = 2.01204e-06, old rate = 9.91156e-07, dx = 480.884, dgrad = 23900.3
DEBUG bump the rate up to 0.000001
      pass   10, meansq err 0.678591, gradient last 121553.794641 all 121894.553029
         p   10, test: meansq err 0.662618, error rate 0.920279
DEBUG 1/L = 2.61546e-06, old rate = 1.34136e-06, dx = 500.74, dgrad = 19145.4
DEBUG bump the rate up to 0.000002
      pass   11, meansq err 0.659459, gradient last 108628.701341 all 108885.241843
         p   11, test: meansq err 0.646483, error rate 0.920279
DEBUG 1/L = 3.34051e-06, old rate = 1.74364e-06, dx = 501.374, dgrad = 15008.9
DEBUG bump the rate up to 0.000002
      pass   12, meansq err 0.643736, gradient last 100504.878313 all 100717.374931
         p   12, test: meansq err 0.632521, error rate 0.920279
DEBUG 1/L = 4.6356e-06, old rate = 2.22701e-06, dx = 489.077, dgrad = 10550.5
DEBUG bump the rate up to 0.000003
      pass   13, meansq err 0.630147, gradient last 96112.227464 all 96321.922236
         p   13, test: meansq err 0.618894, error rate 0.911809
DEBUG 1/L = 7.86946e-06, old rate = 3.0904e-06, dx = 534.338, dgrad = 6790.01
DEBUG bump the rate up to 0.000005
      pass   14, meansq err 0.616715, gradient last 93306.573520 all 93530.006869
         p   14, test: meansq err 0.604074, error rate 0.848530
DEBUG 1/L = 1.1135e-05, old rate = 5.24631e-06, dx = 707.071, dgrad = 6349.97
DEBUG bump the rate up to 0.000007
      pass   15, meansq err 0.601881, gradient last 89806.239904 all 90028.713387
         p   15, test: meansq err 0.590204, error rate 0.672646
DEBUG 1/L = 7.31041e-06, old rate = 7.42334e-06, dx = 1033.54, dgrad = 14137.9
DEBUG bump the rate down to 0.000005
DEBUG restored weights for redo
DEBUG this is a redo of the last pass
      pass   15, meansq err 0.601881, gradient last 89806.239904 all 90028.713387
         p   15, test: meansq err 0.590801, error rate 0.676134
      pass   16, meansq err 0.588490, gradient last 85136.963624 all 85336.993443
         p   16, test: meansq err 0.585418, error rate 0.665172
DEBUG 1/L = 6.25581e-06, old rate = 4.87361e-06, dx = 603.485, dgrad = 9646.78
      pass   17, meansq err 0.583180, gradient last 79808.469624 all 80044.210096
         p   17, test: meansq err 0.582769, error rate 0.609367
DEBUG 1/L = 2.91771e-06, old rate = 4.87361e-06, dx = 826.44, dgrad = 28325
DEBUG bump the rate down to 0.000002
DEBUG restored weights for redo
DEBUG this is a redo of the last pass
      pass   17, meansq err 0.583180, gradient last 79808.469624 all 80044.210096
         p   17, test: meansq err 0.583629, error rate 0.638764
      pass   18, meansq err 0.581135, gradient last 81763.033440 all 81970.581513
         p   18, test: meansq err 0.581190, error rate 0.644743
DEBUG 1/L = 2.01773e-06, old rate = 1.94514e-06, dx = 234.348, dgrad = 11614.4
DEBUG bump the rate down to 0.000001
DEBUG restored weights for redo
DEBUG this is a redo of the last pass
      pass   18, meansq err 0.581135, gradient last 81763.033440 all 81970.581513
         p   18, test: meansq err 0.581827, error rate 0.644743
      pass   19, meansq err 0.579455, gradient last 78014.721038 all 78248.462284
         p   19, test: meansq err 0.580350, error rate 0.628301
DEBUG 1/L = 8.15841e-06, old rate = 1.34515e-06, dx = 208.405, dgrad = 2554.48
DEBUG bump the rate up to 0.000003
      pass   20, meansq err 0.577897, gradient last 77855.256953 all 78083.421326
         p   20, test: meansq err 0.578748, error rate 0.622820
DEBUG 1/L = 1.10498e-05, old rate = 2.69031e-06, dx = 192.442, dgrad = 1741.59
DEBUG bump the rate up to 0.000005
      pass   21, meansq err 0.576306, gradient last 76713.734637 all 76948.324430
         p   21, test: meansq err 0.575546, error rate 0.604883
DEBUG 1/L = 1.78135e-05, old rate = 5.38061e-06, dx = 404.069, dgrad = 2268.32
DEBUG bump the rate up to 0.000011
      pass   22, meansq err 0.573051, gradient last 75297.864121 all 75535.906109
         p   22, test: meansq err 0.568970, error rate 0.576981
DEBUG 1/L = 1.56712e-05, old rate = 1.07612e-05, dx = 812.651, dgrad = 5185.64
      pass   23, meansq err 0.566447, gradient last 71672.583393 all 71933.072002
         p   23, test: meansq err 0.558542, error rate 0.558047
DEBUG 1/L = 7.68341e-06, old rate = 1.07612e-05, dx = 1932.22, dgrad = 25147.9
DEBUG bump the rate down to 0.000005
DEBUG restored weights for redo
DEBUG this is a redo of the last pass
      pass   23, meansq err 0.566447, gradient last 71672.583393 all 71933.072002
         p   23, test: meansq err 0.563365, error rate 0.534629
      pass   24, meansq err 0.560377, gradient last 74293.260310 all 74511.465647
         p   24, test: meansq err 0.561010, error rate 0.563029
DEBUG 1/L = 1.72361e-06, old rate = 5.12227e-06, dx = 654.949, dgrad = 37998.6
DEBUG bump the rate down to 0.000001
DEBUG restored weights for redo
DEBUG this is a redo of the last pass
      pass   24, meansq err 0.560377, gradient last 74293.260310 all 74511.465647
         p   24, test: meansq err 0.561284, error rate 0.547583
      pass   25, meansq err 0.558503, gradient last 69656.952889 all 69907.226436
         p   25, test: meansq err 0.559852, error rate 0.541106
DEBUG 1/L = 1.84569e-05, old rate = 1.14908e-06, dx = 192.323, dgrad = 1042.01
DEBUG bump the rate up to 0.000002
      pass   26, meansq err 0.557014, gradient last 69322.174841 all 69571.353568
         p   26, test: meansq err 0.558403, error rate 0.537120
DEBUG 1/L = 2.67717e-05, old rate = 2.29815e-06, dx = 191.977, dgrad = 717.089
DEBUG bump the rate up to 0.000005
      pass   27, meansq err 0.555525, gradient last 68808.581030 all 69058.867492
         p   27, test: meansq err 0.555523, error rate 0.534131
DEBUG 1/L = 2.74884e-05, old rate = 4.59631e-06, dx = 390.154, dgrad = 1419.34
DEBUG bump the rate up to 0.000009
      pass   28, meansq err 0.552548, gradient last 67843.250012 all 68094.931688
         p   28, test: meansq err 0.549841, error rate 0.527653
DEBUG 1/L = 3.58856e-05, old rate = 9.19261e-06, dx = 803.869, dgrad = 2240.09
DEBUG bump the rate up to 0.000018
      pass   29, meansq err 0.546677, gradient last 66190.154660 all 66435.624739
         p   29, test: meansq err 0.535812, error rate 0.495267
DEBUG 1/L = 1.25214e-05, old rate = 1.83852e-05, dx = 1586.98, dgrad = 12674.2
DEBUG bump the rate down to 0.000008
DEBUG restored weights for redo
DEBUG this is a redo of the last pass
      pass   29, meansq err 0.546677, gradient last 66190.154660 all 66435.624739
         p   29, test: meansq err 0.537221, error rate 0.498754
      pass   30, meansq err 0.533697, gradient last 64854.098939 all 65579.135522
         p   30, test: meansq err 0.556908, error rate 0.646238
DEBUG 1/L = 4.60024e-06, old rate = 8.34758e-06, dx = 3483.38, dgrad = 75721.6
DEBUG bump the rate down to 0.000003
DEBUG restored weights for redo
DEBUG this is a redo of the last pass
      pass   30, meansq err 0.533697, gradient last 64854.098939 all 65579.135522
         p   30, test: meansq err 0.530886, error rate 0.561535
      pass   31, meansq err 0.526463, gradient last 63593.024285 all 63876.579496
         p   31, test: meansq err 0.526946, error rate 0.494768
DEBUG 1/L = 4.05531e-06, old rate = 3.06683e-06, dx = 695.559, dgrad = 17151.8
      pass   32, meansq err 0.522563, gradient last 60286.402193 all 60721.600208
         p   32, test: meansq err 0.523850, error rate 0.506726
DEBUG 1/L = 5.72382e-06, old rate = 3.06683e-06, dx = 952.581, dgrad = 16642.4
DEBUG bump the rate up to 0.000004
      pass   33, meansq err 0.519079, gradient last 63192.880913 all 63450.870558
         p   33, test: meansq err 0.520523, error rate 0.489287
DEBUG 1/L = 3.20251e-06, old rate = 3.81588e-06, dx = 607.614, dgrad = 18973
DEBUG bump the rate down to 0.000002
DEBUG restored weights for redo
DEBUG this is a redo of the last pass
      pass   33, meansq err 0.519079, gradient last 63192.880913 all 63450.870558
         p   33, test: meansq err 0.520871, error rate 0.493772
      pass   34, meansq err 0.516173, gradient last 57979.186906 all 58275.557415
         p   34, test: meansq err 0.518656, error rate 0.476333
DEBUG 1/L = 6.1116e-06, old rate = 2.13501e-06, dx = 483.675, dgrad = 7914.04
DEBUG bump the rate up to 0.000004
      pass   35, meansq err 0.513771, gradient last 60424.586267 all 60632.577493
         p   35, test: meansq err 0.516338, error rate 0.481315
DEBUG 1/L = 4.68952e-06, old rate = 4.0744e-06, dx = 288.546, dgrad = 6153
DEBUG bump the rate down to 0.000003
      pass   36, meansq err 0.511484, gradient last 57206.237882 all 57453.508272
         p   36, test: meansq err 0.512649, error rate 0.457399
DEBUG 1/L = 8.99862e-06, old rate = 3.12635e-06, dx = 750.813, dgrad = 8343.64
DEBUG bump the rate up to 0.000006
      pass   37, meansq err 0.507528, gradient last 59867.711709 all 60054.960976
         p   37, test: meansq err 0.509899, error rate 0.484803
DEBUG 1/L = 2.54961e-06, old rate = 5.99908e-06, dx = 376.206, dgrad = 14755.4
DEBUG bump the rate down to 0.000002
DEBUG restored weights for redo
DEBUG this is a redo of the last pass
      pass   37, meansq err 0.507528, gradient last 59867.711709 all 60054.960976
         p   37, test: meansq err 0.510831, error rate 0.475336
      pass   38, meansq err 0.505769, gradient last 55541.784861 all 55772.460849
         p   38, test: meansq err 0.509412, error rate 0.465869
DEBUG 1/L = 1.02576e-05, old rate = 1.69974e-06, dx = 302.415, dgrad = 2948.2
DEBUG bump the rate up to 0.000003
      pass   39, meansq err 0.504235, gradient last 56232.658544 all 56438.956396
         p   39, test: meansq err 0.507994, error rate 0.468361
DEBUG 1/L = 1.803e-05, old rate = 3.39949e-06, dx = 259.281, dgrad = 1438.05
DEBUG bump the rate up to 0.000007
      pass   40, meansq err 0.502774, gradient last 55244.459231 all 55456.965160
         p   40, test: meansq err 0.505323, error rate 0.464375
DEBUG 1/L = 3.85228e-05, old rate = 6.79897e-06, dx = 549.493, dgrad = 1426.41
DEBUG bump the rate up to 0.000014
      pass   41, meansq err 0.499992, gradient last 54586.490716 all 54787.143475
         p   41, test: meansq err 0.500360, error rate 0.456403
DEBUG 1/L = 2.53341e-05, old rate = 1.35979e-05, dx = 1032.86, dgrad = 4076.94
DEBUG bump the rate up to 0.000017
      pass   42, meansq err 0.494866, gradient last 51561.371444 all 51792.046403
         p   42, test: meansq err 0.494915, error rate 0.409567
DEBUG 1/L = 1.07987e-05, old rate = 1.68894e-05, dx = 2481.6, dgrad = 22980.5
DEBUG bump the rate down to 0.000007
DEBUG restored weights for redo
DEBUG this is a redo of the last pass
      pass   42, meansq err 0.494866, gradient last 51561.371444 all 51792.046403
         p   42, test: meansq err 0.496498, error rate 0.423019
      pass   43, meansq err 0.490343, gradient last 56074.554410 all 56238.325866
         p   43, test: meansq err 0.502827, error rate 0.456403
DEBUG 1/L = 1.52025e-06, old rate = 7.19917e-06, dx = 709.029, dgrad = 46639
DEBUG bump the rate down to 0.000001
DEBUG restored weights for redo
DEBUG this is a redo of the last pass
      pass   43, meansq err 0.490343, gradient last 56074.554410 all 56238.325866
         p   43, test: meansq err 0.495048, error rate 0.432985
      pass   44, meansq err 0.488958, gradient last 51424.179959 all 51618.991199
         p   44, test: meansq err 0.494310, error rate 0.432985
DEBUG 1/L = 6.61317e-06, old rate = 1.0135e-06, dx = 138.055, dgrad = 2087.58
DEBUG bump the rate up to 0.000002
      pass   45, meansq err 0.488188, gradient last 50249.352819 all 50458.281002
         p   45, test: meansq err 0.493640, error rate 0.430992
DEBUG 1/L = 2.44925e-05, old rate = 2.027e-06, dx = 154.44, dgrad = 630.56
DEBUG bump the rate up to 0.000004
      pass   46, meansq err 0.487458, gradient last 49778.724097 all 49994.373258
         p   46, test: meansq err 0.492314, error rate 0.426507
DEBUG 1/L = 4.76891e-05, old rate = 4.054e-06, dx = 320.248, dgrad = 671.532
DEBUG bump the rate up to 0.000008
      pass   47, meansq err 0.485989, gradient last 49151.114650 all 49370.681604
         p   47, test: meansq err 0.489692, error rate 0.412556
DEBUG 1/L = 6.22763e-05, old rate = 8.10799e-06, dx = 646.842, dgrad = 1038.67
DEBUG bump the rate up to 0.000016
      pass   48, meansq err 0.483041, gradient last 48195.241065 all 48414.170519
         p   48, test: meansq err 0.484409, error rate 0.398107
DEBUG 1/L = 4.98529e-05, old rate = 1.6216e-05, dx = 1256.64, dgrad = 2520.69
DEBUG bump the rate up to 0.000032
      pass   49, meansq err 0.477164, gradient last 45842.172155 all 46074.480497
         p   49, test: meansq err 0.474310, error rate 0.357250
DEBUG 1/L = 3.51778e-05, old rate = 3.2432e-05, dx = 2590.84, dgrad = 7364.98
DEBUG bump the rate down to 0.000023
      pass   50, meansq err 0.465862, gradient last 45764.059549 all 45949.788305
         p   50, test: meansq err 0.597242, error rate 0.353264
DEBUG 1/L = 2.10219e-06, old rate = 2.34519e-05, dx = 3236.22, dgrad = 153945
DEBUG bump the rate down to 0.000001
DEBUG restored weights for redo
DEBUG this is a redo of the last pass
      pass   50, meansq err 0.465862, gradient last 45764.059549 all 45949.788305
         p   50, test: meansq err 0.472959, error rate 0.360737
      pass   51, meansq err 0.464516, gradient last 42174.306093 all 42394.178801
         p   51, test: meansq err 0.472090, error rate 0.359243
DEBUG 1/L = 2.70508e-05, old rate = 1.40146e-06, dx = 194.157, dgrad = 717.749
DEBUG bump the rate up to 0.000003
      pass   52, meansq err 0.463573, gradient last 41646.866137 all 41873.292407
         p   52, test: meansq err 0.471243, error rate 0.354758
DEBUG 1/L = 5.26476e-05, old rate = 2.80292e-06, dx = 202.721, dgrad = 385.053
DEBUG bump the rate up to 0.000006
      pass   53, meansq err 0.462641, gradient last 41309.097192 all 41534.770446
         p   53, test: meansq err 0.469569, error rate 0.346786
DEBUG 1/L = 5.79216e-05, old rate = 5.60583e-06, dx = 401.199, dgrad = 692.659
DEBUG bump the rate up to 0.000011
      pass   54, meansq err 0.460811, gradient last 40671.752966 all 40897.333004
         p   54, test: meansq err 0.466291, error rate 0.330344
DEBUG 1/L = 6.40988e-05, old rate = 1.12117e-05, dx = 794.291, dgrad = 1239.17
DEBUG bump the rate up to 0.000022
      pass   55, meansq err 0.457208, gradient last 39535.210665 all 39761.820873
         p   55, test: meansq err 0.459932, error rate 0.306428
DEBUG 1/L = 5.37449e-05, old rate = 2.24233e-05, dx = 1558.13, dgrad = 2899.12
DEBUG bump the rate up to 0.000036
      pass   56, meansq err 0.450195, gradient last 36997.388140 all 37230.597235
         p   56, test: meansq err 0.448900, error rate 0.246637
DEBUG 1/L = 2.71214e-05, old rate = 3.58299e-05, dx = 3035.18, dgrad = 11191.1
DEBUG bump the rate down to 0.000018
DEBUG restored weights for redo
DEBUG this is a redo of the last pass
      pass   56, meansq err 0.450195, gradient last 36997.388140 all 37230.597235
         p   56, test: meansq err 0.450785, error rate 0.258595
      pass   57, meansq err 0.439945, gradient last 37446.387700 all 37669.920046
         p   57, test: meansq err 0.491093, error rate 0.308421
DEBUG 1/L = 2.92213e-06, old rate = 1.8081e-05, dx = 2323.58, dgrad = 79516.5
DEBUG bump the rate down to 0.000002
DEBUG restored weights for redo
DEBUG this is a redo of the last pass
      pass   57, meansq err 0.439945, gradient last 37446.387700 all 37669.920046
         p   57, test: meansq err 0.448793, error rate 0.260090
      pass   58, meansq err 0.437953, gradient last 33518.020056 all 33775.476289
         p   58, test: meansq err 0.447712, error rate 0.259093
DEBUG 1/L = 3.6782e-05, old rate = 1.94809e-06, dx = 281.762, dgrad = 766.031
DEBUG bump the rate up to 0.000004
      pass   59, meansq err 0.436745, gradient last 33267.228169 all 33510.704658
         p   59, test: meansq err 0.446620, error rate 0.255605
DEBUG 1/L = 6.22338e-05, old rate = 3.89617e-06, dx = 250.538, dgrad = 402.575
DEBUG bump the rate up to 0.000008
      pass   60, meansq err 0.435547, gradient last 32907.868552 all 33153.490621
         p   60, test: meansq err 0.444419, error rate 0.248630
DEBUG 1/L = 7.60022e-05, old rate = 7.79235e-06, dx = 499.891, dgrad = 657.732
DEBUG bump the rate up to 0.000016
      pass   61, meansq err 0.433136, gradient last 32300.579563 all 32548.356969
         p   61, test: meansq err 0.439930, error rate 0.237668
DEBUG 1/L = 7.1914e-05, old rate = 1.55847e-05, dx = 979.558, dgrad = 1362.12
DEBUG bump the rate up to 0.000031
      pass   62, meansq err 0.428257, gradient last 31041.362957 all 31304.399240
         p   62, test: meansq err 0.430860, error rate 0.218236
DEBUG 1/L = 7.95049e-05, old rate = 3.11694e-05, dx = 2004.4, dgrad = 2521.11
DEBUG bump the rate up to 0.000053
      pass   63, meansq err 0.418264, gradient last 29244.792087 all 29506.833870
         p   63, test: meansq err 0.417874, error rate 0.206278
DEBUG 1/L = 1.87546e-05, old rate = 5.30033e-05, dx = 3185.57, dgrad = 16985.5
DEBUG bump the rate down to 0.000013
DEBUG restored weights for redo
DEBUG this is a redo of the last pass
      pass   63, meansq err 0.418264, gradient last 29244.792087 all 29506.833870
         p   63, test: meansq err 0.424095, error rate 0.211261
      pass   64, meansq err 0.410409, gradient last 27261.515295 all 27870.972645
         p   64, test: meansq err 0.431178, error rate 0.194320
DEBUG 1/L = 9.91653e-06, old rate = 1.25031e-05, dx = 3432.25, dgrad = 34611.4
DEBUG bump the rate down to 0.000007
DEBUG restored weights for redo
DEBUG this is a redo of the last pass
      pass   64, meansq err 0.410409, gradient last 27261.515295 all 27870.972645
         p   64, test: meansq err 0.422139, error rate 0.197309
      pass   65, meansq err 0.408465, gradient last 33434.436338 all 34170.826587
         p   65, test: meansq err 0.430915, error rate 0.215745
DEBUG 1/L = 5.37823e-06, old rate = 6.61102e-06, dx = 2147.5, dgrad = 39929.5
DEBUG bump the rate down to 0.000004
DEBUG restored weights for redo
DEBUG this is a redo of the last pass
      pass   65, meansq err 0.408465, gradient last 33434.436338 all 34170.826587
         p   65, test: meansq err 0.418635, error rate 0.202790
      pass   66, meansq err 0.403971, gradient last 27163.471308 all 28236.502323
         p   66, test: meansq err 0.415607, error rate 0.196811
DEBUG 1/L = 8.95818e-06, old rate = 3.58548e-06, dx = 1447.95, dgrad = 16163.5
DEBUG bump the rate up to 0.000006
      pass   67, meansq err 0.400852, gradient last 28467.304105 all 29014.100311
         p   67, test: meansq err 0.413106, error rate 0.193822
DEBUG 1/L = 5.86084e-06, old rate = 5.97212e-06, dx = 787.205, dgrad = 13431.6
DEBUG bump the rate down to 0.000004
DEBUG restored weights for redo
DEBUG this is a redo of the last pass
      pass   67, meansq err 0.400852, gradient last 28467.304105 all 29014.100311
         p   67, test: meansq err 0.413264, error rate 0.194320
      pass   68, meansq err 0.397787, gradient last 25984.971045 all 26948.602490
         p   68, test: meansq err 0.410695, error rate 0.193323
DEBUG 1/L = 9.80454e-06, old rate = 3.90722e-06, dx = 1406.26, dgrad = 14343
DEBUG bump the rate up to 0.000007
      pass   69, meansq err 0.394880, gradient last 27137.553787 all 27779.067858
         p   69, test: meansq err 0.408369, error rate 0.190334
DEBUG 1/L = 6.5284e-06, old rate = 6.53636e-06, dx = 938.079, dgrad = 14369.2
DEBUG bump the rate down to 0.000004
DEBUG restored weights for redo
DEBUG this is a redo of the last pass
      pass   69, meansq err 0.394880, gradient last 27137.553787 all 27779.067858
         p   69, test: meansq err 0.408712, error rate 0.189836
      pass   70, meansq err 0.392207, gradient last 25836.603354 all 27105.944274
         p   70, test: meansq err 0.407150, error rate 0.185850
DEBUG 1/L = 9.48739e-06, old rate = 4.35227e-06, dx = 1844.42, dgrad = 19440.8
DEBUG bump the rate up to 0.000006
      pass   71, meansq err 0.390360, gradient last 27692.576261 all 28873.616515
         p   71, test: meansq err 0.406163, error rate 0.182860
DEBUG 1/L = 7.14137e-06, old rate = 6.32492e-06, dx = 1683.93, dgrad = 23580
DEBUG bump the rate down to 0.000005
      pass   72, meansq err 0.388811, gradient last 27064.591282 all 29259.996320
         p   72, test: meansq err 0.418524, error rate 0.184355
DEBUG 1/L = 9.2702e-06, old rate = 4.76091e-06, dx = 3737.85, dgrad = 40321.2
DEBUG bump the rate up to 0.000006
      pass   73, meansq err 0.402442, gradient last 38927.899529 all 43787.308636
         p   73, test: meansq err 0.428676, error rate 0.176383
DEBUG 1/L = 7.92995e-06, old rate = 6.18013e-06, dx = 5021.52, dgrad = 63323.5
      pass   74, meansq err 0.412884, gradient last 45307.073756 all 54398.318555
         p   74, test: meansq err 0.511935, error rate 0.196811
DEBUG 1/L = 9.91719e-06, old rate = 6.18013e-06, dx = 9966.4, dgrad = 100496
DEBUG bump the rate up to 0.000007
      pass   75, meansq err 0.502923, gradient last 73808.126055 all 86872.073784
         p   75, test: meansq err 0.603145, error rate 0.177379
DEBUG 1/L = 9.03622e-06, old rate = 6.61146e-06, dx = 14989.2, dgrad = 165879
      pass   76, meansq err 0.594955, gradient last 121746.744798 all 150533.184371
         p   76, test: meansq err 0.868434, error rate 0.231689
DEBUG 1/L = 1.44217e-05, old rate = 6.61146e-06, dx = 29788.4, dgrad = 206552
DEBUG bump the rate up to 0.000010
      pass   77, meansq err 0.869680, gradient last 117217.906290 all 146217.859124
         p   77, test: meansq err 0.601898, error rate 0.412058
DEBUG 1/L = 1.47755e-05, old rate = 9.61448e-06, dx = 27968.4, dgrad = 189289
DEBUG bump the rate up to 0.000010
      pass   78, meansq err 0.592387, gradient last 110733.110920 all 125698.424443
         p   78, test: meansq err 0.797555, error rate 0.335825
DEBUG 1/L = 1.51436e-05, old rate = 9.85031e-06, dx = 28024.9, dgrad = 185061
DEBUG bump the rate up to 0.000010
      pass   79, meansq err 0.800243, gradient last 105736.048760 all 128528.319392
         p   79, test: meansq err 0.657395, error rate 0.567015
DEBUG 1/L = 1.61737e-05, old rate = 1.00957e-05, dx = 33185.4, dgrad = 205181
DEBUG bump the rate up to 0.000011
      pass   80, meansq err 0.650430, gradient last 161973.281235 all 182068.217520
         p   80, test: meansq err 1.072384, error rate 0.402093
DEBUG 1/L = 1.3518e-05, old rate = 1.07825e-05, dx = 35606.8, dgrad = 263403
      pass   81, meansq err 1.071768, gradient last 159385.752888 all 183026.960257
         p   81, test: meansq err 0.608781, error rate 0.550075
DEBUG 1/L = 1.59856e-05, old rate = 1.07825e-05, dx = 30330.5, dgrad = 189736
      pass   82, meansq err 0.604173, gradient last 101045.180631 all 102074.445812
         p   82, test: meansq err 0.862957, error rate 0.397110
DEBUG 1/L = 2.34055e-06, old rate = 1.07825e-05, dx = 6695.26, dgrad = 286055
DEBUG bump the rate down to 0.000002
DEBUG restored weights for redo
DEBUG this is a redo of the last pass
      pass   82, meansq err 0.604173, gradient last 101045.180631 all 102074.445812
         p   82, test: meansq err 0.536345, error rate 0.424514
      pass   83, meansq err 0.530808, gradient last 69480.234225 all 71096.313652
         p   83, test: meansq err 0.498381, error rate 0.414549
DEBUG 1/L = 4.70895e-06, old rate = 1.56036e-06, dx = 1000.16, dgrad = 21239.6
DEBUG bump the rate up to 0.000003
      pass   84, meansq err 0.491671, gradient last 52013.306411 all 53406.046966
         p   84, test: meansq err 0.477362, error rate 0.385650
DEBUG 1/L = 5.97996e-06, old rate = 3.12073e-06, dx = 780.906, dgrad = 13058.7
DEBUG bump the rate up to 0.000004
      pass   85, meansq err 0.469635, gradient last 41810.019784 all 42831.283063
         p   85, test: meansq err 0.454783, error rate 0.320877
DEBUG 1/L = 6.95951e-06, old rate = 3.98664e-06, dx = 1124.49, dgrad = 16157.6
DEBUG bump the rate up to 0.000005
      pass   86, meansq err 0.445433, gradient last 31221.020146 all 31781.561597
         p   86, test: meansq err 0.440167, error rate 0.239661
DEBUG 1/L = 9.72315e-06, old rate = 4.63967e-06, dx = 720.731, dgrad = 7412.53
DEBUG bump the rate up to 0.000006
      pass   87, meansq err 0.429680, gradient last 27616.362913 all 28126.233533
         p   87, test: meansq err 0.429789, error rate 0.201295
DEBUG 1/L = 1.37104e-05, old rate = 6.4821e-06, dx = 709.361, dgrad = 5173.89
DEBUG bump the rate up to 0.000009
      pass   88, meansq err 0.418471, gradient last 24508.259910 all 24956.629834
         p   88, test: meansq err 0.422027, error rate 0.205282
DEBUG 1/L = 1.39974e-05, old rate = 9.14027e-06, dx = 817.944, dgrad = 5843.53
DEBUG bump the rate up to 0.000009
      pass   89, meansq err 0.409968, gradient last 22797.158986 all 23188.766766
         p   89, test: meansq err 0.416387, error rate 0.204285
DEBUG 1/L = 1.68113e-05, old rate = 9.33163e-06, dx = 979.948, dgrad = 5829.09
DEBUG bump the rate up to 0.000011
      pass   90, meansq err 0.403356, gradient last 21232.833194 all 21722.378511
         p   90, test: meansq err 0.412729, error rate 0.194320
DEBUG 1/L = 9.54373e-06, old rate = 1.12076e-05, dx = 1458.75, dgrad = 15284.9
DEBUG bump the rate down to 0.000006
DEBUG restored weights for redo
DEBUG this is a redo of the last pass
      pass   90, meansq err 0.403356, gradient last 21232.833194 all 21722.378511
         p   90, test: meansq err 0.412904, error rate 0.196811
      pass   91, meansq err 0.399443, gradient last 23798.670314 all 24205.830388
         p   91, test: meansq err 0.410413, error rate 0.193822
DEBUG 1/L = 5.12268e-06, old rate = 6.36249e-06, dx = 780.008, dgrad = 15226.5
DEBUG bump the rate down to 0.000003
DEBUG restored weights for redo
DEBUG this is a redo of the last pass
      pass   91, meansq err 0.399443, gradient last 23798.670314 all 24205.830388
         p   91, test: meansq err 0.410216, error rate 0.192327
      pass   92, meansq err 0.396202, gradient last 20241.499287 all 20682.706366
         p   92, test: meansq err 0.407965, error rate 0.189337
DEBUG 1/L = 1.46091e-05, old rate = 3.41512e-06, dx = 431.482, dgrad = 2953.51
DEBUG bump the rate up to 0.000007
      pass   93, meansq err 0.393561, gradient last 20443.602626 all 20842.147277
         p   93, test: meansq err 0.405799, error rate 0.186348
DEBUG 1/L = 2.74143e-05, old rate = 6.83025e-06, dx = 341.213, dgrad = 1244.65
DEBUG bump the rate up to 0.000014
      pass   94, meansq err 0.390884, gradient last 19902.534810 all 20327.035510
         p   94, test: meansq err 0.401312, error rate 0.182860
DEBUG 1/L = 6.77251e-05, old rate = 1.36605e-05, dx = 728.842, dgrad = 1076.18
DEBUG bump the rate up to 0.000027
      pass   95, meansq err 0.385463, gradient last 19544.756972 all 19964.862990
         p   95, test: meansq err 0.392450, error rate 0.174888
DEBUG 1/L = 4.09948e-05, old rate = 2.7321e-05, dx = 1342.22, dgrad = 3274.13
DEBUG bump the rate up to 0.000027
      pass   96, meansq err 0.374896, gradient last 18390.572191 all 18857.947408
         p   96, test: meansq err 0.390272, error rate 0.166916
DEBUG 1/L = 1.21111e-05, old rate = 2.73299e-05, dx = 3200.57, dgrad = 26426.7
DEBUG bump the rate down to 0.000008
DEBUG restored weights for redo
DEBUG this is a redo of the last pass
      pass   96, meansq err 0.374896, gradient last 18390.572191 all 18857.947408
         p   96, test: meansq err 0.387893, error rate 0.169407
      pass   97, meansq err 0.369649, gradient last 20136.533252 all 20579.951538
         p   97, test: meansq err 0.386007, error rate 0.173891
DEBUG 1/L = 6.16239e-06, old rate = 8.0741e-06, dx = 974.346, dgrad = 15811.2
DEBUG bump the rate down to 0.000004
DEBUG restored weights for redo
DEBUG this is a redo of the last pass
      pass   97, meansq err 0.369649, gradient last 20136.533252 all 20579.951538
         p   97, test: meansq err 0.385153, error rate 0.170902
      pass   98, meansq err 0.366344, gradient last 17810.974646 all 18278.866247
         p   98, test: meansq err 0.382461, error rate 0.166418
DEBUG 1/L = 1.11324e-05, old rate = 4.10826e-06, dx = 506.766, dgrad = 4552.19
DEBUG bump the rate up to 0.000007
      pass   99, meansq err 0.363449, gradient last 17896.474690 all 18270.494984
         p   99, test: meansq err 0.380073, error rate 0.163428
}

reduced size 8x8:

{
	FloatNeuralNet::LevelSizeVector levels = {64, 64, 32, 10};
	options.weightSaturation_ = 100.;
	TestFloatNn nn(levels, FloatNeuralNet::LEAKY_RELU, &options);
	seed = 1667553859;
	double rate = 0.05;
	double trainRate = 0.;
	double applyRate = rate;
...
DEBUG effective rate = 6.85777e-06
      pass 19831, meansq err 0.069765, gradient last 1291.604484 all 4105.998879
         p 19831, test: meansq err 0.200372, error rate 0.054808
...
DEBUG effective rate = 6.85777e-06
      pass 19999, meansq err 0.073539, gradient last 2532.187868 all 6345.037580
         p 19999, test: meansq err 0.202077, error rate 0.056303

	FloatNeuralNet::LevelSizeVector levels = {64, 64, 10};
	options.weightSaturation_ = 1.;
	options.enableWeightFloor_ = true;
	TestFloatNn nn(levels, FloatNeuralNet::CORNER, &options);
	double rate = 0.1;
	seed = 1667553859;
	double trainRate = 0.;
	double applyRate = rate;
...
DEBUG effective rate = 1.37155e-05
      pass 1719, meansq err 0.135464, gradient last 3887.187399 all 5999.934983
         p 1719, test: meansq err 0.213921, error rate 0.057798
...
DEBUG effective rate = 1.37155e-05
      pass 2476, meansq err 0.124672, gradient last 3984.061903 all 6193.115634
         p 2476, test: meansq err 0.209169, error rate 0.054808
...
DEBUG effective rate = 1.37155e-05
      pass 3270, meansq err 0.116072, gradient last 3752.379300 all 5793.876981
         p 3270, test: meansq err 0.205938, error rate 0.051320
...
DEBUG effective rate = 1.37155e-05
      pass 3274, meansq err 0.115422, gradient last 3585.183999 all 5526.085286
         p 3274, test: meansq err 0.205640, error rate 0.050822
...
DEBUG effective rate = 1.37155e-05
      pass 3748, meansq err 0.109558, gradient last 3291.370011 all 5102.190370
         p 3748, test: meansq err 0.204859, error rate 0.052815
...
DEBUG effective rate = 1.37155e-05
      pass 4348, meansq err 0.102323, gradient last 2797.338009 all 4317.376315
         p 4348, test: meansq err 0.202990, error rate 0.052815
...
DEBUG effective rate = 1.37155e-05
      pass 5768, meansq err 0.092451, gradient last 2506.095110 all 3964.468381
         p 5768, test: meansq err 0.202623, error rate 0.053812
...
DEBUG effective rate = 1.37155e-05
      pass 7619, meansq err 0.085094, gradient last 2468.523019 all 3972.110116
         p 7619, test: meansq err 0.202681, error rate 0.051320
...
DEBUG effective rate = 1.37155e-05
      pass 9711, meansq err 0.082795, gradient last 3398.551523 all 5206.181062
         p 9711, test: meansq err 0.205045, error rate 0.050324
DEBUG effective rate = 1.37155e-05
      pass 9939, meansq err 0.085218, gradient last 4163.138862 all 6052.600950
         p 9939, test: meansq err 0.205786, error rate 0.049826
...
DEBUG effective rate = 1.37155e-05
      pass 9939, meansq err 0.085218, gradient last 4163.138862 all 6052.600950
         p 9939, test: meansq err 0.205786, error rate 0.049826
...
DEBUG effective rate = 1.37155e-05
      pass 14898, meansq err 0.077937, gradient last 4003.889758 all 5751.981767
         p 14898, test: meansq err 0.207242, error rate 0.052317
...
DEBUG effective rate = 1.37155e-05
      pass 19998, meansq err 0.072114, gradient last 3424.909152 all 4735.953653
         p 19998, test: meansq err 0.208376, error rate 0.052317
DEBUG effective rate = 1.37155e-05
      pass 19999, meansq err 0.073429, gradient last 4585.615030 all 5636.877631
         p 19999, test: meansq err 0.208251, error rate 0.052815

	FloatNeuralNet::LevelSizeVector levels = {64, 64, 10};
	options.weightSaturation_ = 100.;
	options.enableWeightFloor_ = true;
	TestFloatNn nn(levels, FloatNeuralNet::CORNER, &options);
	double rate = 0.05;
	seed = 1667553859;
	double trainRate = 0.;
	double applyRate = rate;
	../../../../zipnn/corner_64_64_10_floor100.ckp
...
DEBUG effective rate = 6.85777e-06
      pass 19993 [28365], meansq err 0.151736, gradient last 21258.701764 all 22604.944827
         p 19993, test: meansq err 0.249180, error rate 0.058794
DEBUG effective rate = 6.85777e-06
      pass 19994 [28366], meansq err 0.155205, gradient last 26022.516491 all 27096.656741
         p 19994, test: meansq err 0.248961, error rate 0.061286
DEBUG effective rate = 6.85777e-06
      pass 19995 [28367], meansq err 0.151738, gradient last 21260.744856 all 22606.216138
         p 19995, test: meansq err 0.249177, error rate 0.058794
DEBUG effective rate = 6.85777e-06
      pass 19996 [28368], meansq err 0.155198, gradient last 26017.331486 all 27091.602748
         p 19996, test: meansq err 0.248961, error rate 0.061286
DEBUG effective rate = 6.85777e-06
      pass 19997 [28369], meansq err 0.151735, gradient last 21260.516254 all 22605.378067
         p 19997, test: meansq err 0.249172, error rate 0.058794
DEBUG effective rate = 6.85777e-06
      pass 19998 [28370], meansq err 0.155185, gradient last 26008.867909 all 27082.546033
         p 19998, test: meansq err 0.248961, error rate 0.061286
DEBUG effective rate = 6.85777e-06
      pass 19999 [28371], meansq err 0.151731, gradient last 21261.236461 all 22605.713381
         p 19999, test: meansq err 0.249167, error rate 0.058794
         p 19999, train: meansq err 0.155172, error rate 0.005349

	options.weightSaturation_ = 1.;
	options.enableWeightFloor_ = true;
	FloatNeuralNet::LevelSizeVector levels = {64, 32, 10};
	TestFloatNn nn(levels, FloatNeuralNet::CORNER, &options);
	string checkpoint = fpath + "/corner_64_32_10_floor.ckp";
	seed = 1667553859;
	double rate = 0.1;
	double trainRate = 0.;
	double applyRate = rate;
	../../../../zipnn/corner_64_32_10_floor1.ckp
...
DEBUG effective rate = 1.37155e-05
      pass  156 [ 157], meansq err 0.231601, gradient last 2409.755578 all 4245.133264
         p  156, test: meansq err 0.272486, error rate 0.092177
DEBUG effective rate = 1.37155e-05
      pass  157 [ 158], meansq err 0.231324, gradient last 2396.725155 all 4234.807614
         p  157, test: meansq err 0.272293, error rate 0.091679
DEBUG effective rate = 1.37155e-05
      pass  158 [ 159], meansq err 0.231055, gradient last 2409.991860 all 4246.158177
         p  158, test: meansq err 0.272127, error rate 0.091679
DEBUG effective rate = 1.37155e-05
      pass  159 [ 160], meansq err 0.230780, gradient last 2393.217800 all 4235.715622
         p  159, test: meansq err 0.271930, error rate 0.091181
         p  159, train: meansq err 0.230517, error rate 0.055274
...
DEBUG effective rate = 1.37155e-05
      pass 1119 [1120], meansq err 0.159129, gradient last 2958.261481 all 4778.540915
         p 1119, test: meansq err 0.224768, error rate 0.062282
         p 1119, train: meansq err 0.158991, error rate 0.022356
...
DEBUG effective rate = 1.37155e-05
      pass 19999 [20000], meansq err 0.105095, gradient last 5621.208616 all 7595.810510
         p 19999, test: meansq err 0.223088, error rate 0.062282
         p 19999, train: meansq err 0.103475, error rate 0.006721

	options.weightSaturation_ = 1.;
	FloatNeuralNet::LevelSizeVector levels = {64, 32, 10};
	TestFloatNn nn(levels, FloatNeuralNet::CORNER, &options);
	string checkpoint = fpath + "/corner_64_32_10_nofloor.ckp";
	seed = 1667553859;
	double rate = 0.1;
	double trainRate = 0.;
	double applyRate = rate;
	../../../../zipnn/corner_64_32_10_nofloor1.ckp
...
DEBUG effective rate = 1.37155e-05
      pass  699 [ 700], meansq err 0.177710, gradient last 4315.875741 all 7095.807683
         p  699, test: meansq err 0.237783, error rate 0.067763
         p  699, train: meansq err 0.178248, error rate 0.028665
...
DEBUG effective rate = 1.37155e-05
      pass 19999 [20000], meansq err 0.102839, gradient last 5354.114205 all 6750.998392
         p 19999, test: meansq err 0.220990, error rate 0.061286
         p 19999, train: meansq err 0.102704, error rate 0.005761

	options.weightSaturation_ = 1.;
	FloatNeuralNet::LevelSizeVector levels = {64, 32, 32, 10};
	TestFloatNn nn(levels, FloatNeuralNet::LEAKY_RELU, &options);
	string checkpoint = fpath + "/leaky_64_32_32_10_nofloor.ckp";
	seed = 1667553859;
	double rate = 0.1;
	double trainRate = 0.;
	double applyRate = rate;
	../../../../zipnn/leaky_64_32_32_10_nofloor1.ckp
...
DEBUG effective rate = 1.37155e-05
      pass  699 [ 700], meansq err 0.184612, gradient last 893.578689 all 3376.081822
         p  699, test: meansq err 0.238669, error rate 0.076233
         p  699, train: meansq err 0.184660, error rate 0.030723
...
DEBUG effective rate = 1.37155e-05
      pass 1849 [1850], meansq err 0.145371, gradient last 679.785535 all 2228.722821
         p 1849, test: meansq err 0.220308, error rate 0.066766
         p 1849, train: meansq err 0.145329, error rate 0.017830
...
DEBUG effective rate = 1.37155e-05
      pass 11099 [11100], meansq err 0.101249, gradient last 1124.527997 all 3524.659162
         p 11099, test: meansq err 0.214921, error rate 0.065272
         p 11099, train: meansq err 0.101196, error rate 0.006995
...
DEBUG effective rate = 1.37155e-05
      pass 19999 [20000], meansq err 0.091898, gradient last 1795.316627 all 4674.748496
         p 19999, test: meansq err 0.217704, error rate 0.067763
         p 19999, train: meansq err 0.091711, error rate 0.004938

	options.weightSaturation_ = 1.;
	FloatNeuralNet::LevelSizeVector levels = {64, 32, 32, 10};
	TestFloatNn nn(levels, FloatNeuralNet::CORNER, &options);
	string checkpoint = fpath + "/corner_64_32_32_10_nofloor.ckp";
	seed = 1667553859;
	double rate = 0.1;
	double trainRate = 0.;
	double applyRate = rate;
	../../../../zipnn/corner_64_32_32_10_nofloor1.ckp
...
DEBUG effective rate = 1.37155e-05
      pass  949 [ 950], meansq err 0.148490, gradient last 2974.882919 all 5387.266825
         p  949, test: meansq err 0.218499, error rate 0.066268
         p  949, train: meansq err 0.148158, error rate 0.021808
...
DEBUG effective rate = 1.37155e-05
      pass 5139 [5140], meansq err 0.084411, gradient last 3840.256307 all 5653.026757
         p 5139, test: meansq err 0.207438, error rate 0.059791
         p 5139, train: meansq err 0.084918, error rate 0.005623
...
DEBUG effective rate = 1.37155e-05
      pass 13959 [13960], meansq err 0.064695, gradient last 4804.795417 all 5654.549433
         p 13959, test: meansq err 0.209193, error rate 0.063777
         p 13959, train: meansq err 0.064666, error rate 0.003703
...
DEBUG effective rate = 1.37155e-05
      pass 19999 [20000], meansq err 0.061298, gradient last 5022.824815 all 5878.573831
         p 19999, test: meansq err 0.208971, error rate 0.063279
         p 19999, train: meansq err 0.061535, error rate 0.003566

	options.weightSaturation_ = 1.;
	FloatNeuralNet::LevelSizeVector levels = {64, 64, 64, 10};
	TestFloatNn nn(levels, FloatNeuralNet::CORNER, &options);
	string checkpoint = fpath + "/corner_64_64_64_10_nofloor.ckp";
	seed = 1667553859;
	double rate = 0.1;
	double trainRate = 0.;
	double applyRate = rate;
	../../../../zipnn/corner_64_64_64_10_nofloor1.ckp
...
DEBUG effective rate = 1.37155e-05
      pass 1239 [1240], meansq err 0.124583, gradient last 3591.671483 all 6725.885392
         p 1239, test: meansq err 0.202558, error rate 0.056303
         p 1239, train: meansq err 0.125664, error rate 0.012618
...
DEBUG effective rate = 1.37155e-05
      pass 8349 [8350], meansq err 0.048095, gradient last 2625.852393 all 3108.798046
         p 8349, test: meansq err 0.195341, error rate 0.056801
         p 8349, train: meansq err 0.048107, error rate 0.002880
...
DEBUG effective rate = 1.37155e-05
      pass 11809 [11810], meansq err 0.041934, gradient last 1928.801717 all 2161.140104
         p 11809, test: meansq err 0.195509, error rate 0.057798
         p 11809, train: meansq err 0.041959, error rate 0.002743
...
DEBUG effective rate = 1.37155e-05
      pass 15599 [15600], meansq err 0.039001, gradient last 2099.244579 all 2353.839860
         p 15599, test: meansq err 0.195790, error rate 0.058794
         p 15599, train: meansq err 0.039010, error rate 0.002469
...
DEBUG effective rate = 1.37155e-05
      pass 19209 [19210], meansq err 0.036579, gradient last 1814.458281 all 2082.434341
         p 19209, test: meansq err 0.195441, error rate 0.057299
         p 19209, train: meansq err 0.036576, error rate 0.002332
...
DEBUG effective rate = 1.37155e-05
      pass 19998 [19999], meansq err 0.037149, gradient last 2209.480735 all 2484.449295
         p 19998, test: meansq err 0.195142, error rate 0.057299
DEBUG effective rate = 1.37155e-05
      pass 19999 [20000], meansq err 0.037150, gradient last 2192.094890 all 2469.690955
         p 19999, test: meansq err 0.195461, error rate 0.057299
         p 19999, train: meansq err 0.037159, error rate 0.002332

	options.weightSaturation_ = 1.;
	FloatNeuralNet::LevelSizeVector levels = {64, 64, 10};
	TestFloatNn nn(levels, FloatNeuralNet::CORNER, &options);
	string checkpoint = fpath + "/corner_64_64_10_b2_nofloor.ckp";
	int batchSize = 2;
	seed = 1667553859;
	double rate = 0.1;
	double trainRate = 0.;
	double applyRate = rate;
	../../../../zipnn/corner_64_64_10_b2_nofloor1.ckp
...
DEBUG effective rate = 1.37155e-05
      pass  659 [ 660], meansq err 0.213784, gradient last 4091.881701 all 6886.773251
         p  659, test: meansq err 0.291547, error rate 0.093174
         p  659, train: meansq err 0.253621, error rate 0.061171
...
DEBUG effective rate = 1.37155e-05
      pass 3469 [3470], meansq err 0.180421, gradient last 2931.830868 all 5172.096973
         p 3469, test: meansq err 0.241679, error rate 0.067265
         p 3469, train: meansq err 0.192972, error rate 0.028254
...
DEBUG effective rate = 1.37155e-05
      pass 8749 [8750], meansq err 0.165831, gradient last 2163.470816 all 3586.935132
         p 8749, test: meansq err 0.229162, error rate 0.057798
         p 8749, train: meansq err 0.170871, error rate 0.016733
...
DEBUG effective rate = 1.37155e-05
      pass 14229 [14230], meansq err 0.162525, gradient last 2264.704849 all 3715.485280
         p 14229, test: meansq err 0.227728, error rate 0.058794
         p 14229, train: meansq err 0.166545, error rate 0.013990
...
DEBUG effective rate = 1.37155e-05
      pass 19999 [20000], meansq err 0.155931, gradient last 1976.810443 all 3310.863321
         p 19999, test: meansq err 0.226347, error rate 0.060289
         p 19999, train: meansq err 0.162830, error rate 0.013304

	options.weightSaturation_ = 1.;
	FloatNeuralNet::LevelSizeVector levels = {64, 64, 10};
	TestFloatNn nn(levels, FloatNeuralNet::CORNER, &options);
	string checkpoint = fpath + "/corner_64_64_10_b8_nofloor.ckp";
	int batchSize = 8;
	seed = 1667553859;
	double rate = 0.1;
	double trainRate = 0.;
	double applyRate = rate;
	../../../../zipnn/corner_64_64_10_b8_nofloor1.ckp
...
DEBUG effective rate = 1.37155e-05
      pass 15249 [15250], meansq err 0.123508, gradient last 1842.301795 all 3111.918423
         p 15249, test: meansq err 0.326506, error rate 0.112606
         p 15249, train: meansq err 0.298941, error rate 0.070086
...
DEBUG effective rate = 1.37155e-05
      pass 19997 [19998], meansq err 0.123548, gradient last 2014.530960 all 3533.089548
         p 19997, test: meansq err 0.321072, error rate 0.108122
DEBUG effective rate = 1.37155e-05
      pass 19998 [19999], meansq err 0.123436, gradient last 1986.232109 all 3550.742365
         p 19998, test: meansq err 0.321785, error rate 0.107623
DEBUG effective rate = 1.37155e-05
      pass 19999 [20000], meansq err 0.122613, gradient last 1978.024235 all 3534.257286
         p 19999, test: meansq err 0.321089, error rate 0.107623
         p 19999, train: meansq err 0.292252, error rate 0.064600

	options.isClassifier_ = true;
	options.correctMultiplier_ = 1. / 1000.;
	options.weightSaturation_ = 1.;
	FloatNeuralNet::LevelSizeVector levels = {64, 64, 10};
	TestFloatNn nn(levels, FloatNeuralNet::CORNER, &options);
	string checkpoint = fpath + "/corner_64_64_10_nofl_c1000.ckp";
	int batchSize = 1;
	seed = 1667553859;
	double rate = 0.1;
	double trainRate = 0.;
	double applyRate = rate;
...
DEBUG effective rate = 0.00885975 (adjusted from 1.37155e-05)
      pass 1929 [1930], meansq err 0.241249, gradient last 20.404272 all 34.869908; error rate est 0.000549
         p 1929, test: meansq err 0.276537, error rate 0.055306
         p 1929, train: meansq err 0.229131, error rate 0.000411
...
DEBUG effective rate = 0.0137155 (adjusted from 1.37155e-05)
      pass 3008 [3009], meansq err 0.192364, gradient last 5.415624 all 6.708066; error rate est 0.000000
         p 3008, test: meansq err 0.243637, error rate 0.049327
...
DEBUG effective rate = 0.0120627 (adjusted from 1.37155e-05)
      pass 5363 [5364], meansq err 0.174283, gradient last 9.446452 all 19.620943; error rate est 0.000137
         p 5363, test: meansq err 0.230308, error rate 0.052815
DEBUG effective rate = 0.0120627 (adjusted from 1.37155e-05)
      pass 5364 [5365], meansq err 0.151862, gradient last 7.376246 all 12.314585; error rate est 0.000137
         p 5364, test: meansq err 0.234695, error rate 0.056801
DEBUG effective rate = 0.00469991 (adjusted from 1.37155e-05)
      pass 5365 [5366], meansq err 0.158366, gradient last 89.330886 all 151.219723; error rate est 0.001920
         p 5365, test: meansq err 0.249062, error rate 0.053313
DEBUG effective rate = 0.00341658 (adjusted from 1.37155e-05)
      pass 5366 [5367], meansq err 0.180360, gradient last 170.478190 all 255.131457; error rate est 0.003017
         p 5366, test: meansq err 0.276286, error rate 0.067763
DEBUG effective rate = 0.000891377 (adjusted from 1.37155e-05)
      pass 5367 [5368], meansq err 0.216768, gradient last 560.670072 all 1188.671064; error rate est 0.014401
         p 5367, test: meansq err 0.246856, error rate 0.056801
DEBUG effective rate = 0.0020305 (adjusted from 1.37155e-05)
      pass 5368 [5369], meansq err 0.179043, gradient last 231.071747 all 390.821922; error rate est 0.005761
         p 5368, test: meansq err 0.275862, error rate 0.062282
DEBUG effective rate = 0.00131196 (adjusted from 1.37155e-05)
      pass 5369 [5370], meansq err 0.214880, gradient last 420.423221 all 747.940325; error rate est 0.009464
         p 5369, test: meansq err 0.236753, error rate 0.052815
         p 5369, train: meansq err 0.167191, error rate 0.004938
DEBUG effective rate = 0.00231187 (adjusted from 1.37155e-05)
      pass 5370 [5371], meansq err 0.167191, gradient last 170.388366 all 343.175352; error rate est 0.004938
         p 5370, test: meansq err 0.317280, error rate 0.069258
DEBUG effective rate = 0.00150984 (adjusted from 1.37155e-05)
      pass 5371 [5372], meansq err 0.255190, gradient last 707.444553 all 999.246437; error rate est 0.008092
         p 5371, test: meansq err 0.368880, error rate 0.154459
DEBUG effective rate = 0.000149784 (adjusted from 1.37155e-05)
      pass 5372 [5373], meansq err 0.326492, gradient last 4440.122918 all 4529.566518; error rate est 0.090660
         p 5372, test: meansq err 0.461394, error rate 0.111111
DEBUG effective rate = 0.000360983 (adjusted from 1.37155e-05)
      pass 5373 [5374], meansq err 0.430108, gradient last 5500.735558 all 5569.126622; error rate est 0.037032
         p 5373, test: meansq err 0.628309, error rate 0.064773
DEBUG effective rate = 0.00207254 (adjusted from 1.37155e-05)
      pass 5374 [5375], meansq err 0.592569, gradient last 426.691860 all 631.475208; error rate est 0.005623
         p 5374, test: meansq err 0.408376, error rate 0.100648
DEBUG effective rate = 0.000456456 (adjusted from 1.37155e-05)
      pass 5375 [5376], meansq err 0.355637, gradient last 3229.028562 all 3561.632967; error rate est 0.029077
         p 5375, test: meansq err 0.806753, error rate 0.141006
DEBUG effective rate = 0.000151828 (adjusted from 1.37155e-05)
      pass 5376 [5377], meansq err 0.787682, gradient last 18018.339017 all 20575.877034; error rate est 0.089425
         p 5376, test: meansq err 1.155394, error rate 0.155456
DEBUG effective rate = 0.000133949 (adjusted from 1.37155e-05)
      pass 5377 [5378], meansq err 1.127438, gradient last 26984.063686 all 32667.399565; error rate est 0.101495
         p 5377, test: meansq err 0.724816, error rate 0.760339
DEBUG effective rate = 1.80124e-05 (adjusted from 1.37155e-05)
      pass 5378 [5379], meansq err 0.701944, gradient last 44498.835374 all 74260.959414; error rate est 0.761212
         p 5378, test: meansq err 1.292324, error rate 0.880419
DEBUG effective rate = 1.54302e-05 (adjusted from 1.37155e-05)
      pass 5379 [5380], meansq err 1.295918, gradient last 331432.707540 all 447150.826195; error rate est 0.888767
         p 5379, test: meansq err 0.827101, error rate 0.702541
         p 5379, train: meansq err 0.824675, error rate 0.702647
DEBUG effective rate = 1.95116e-05 (adjusted from 1.37155e-05)
      pass 5380 [5381], meansq err 0.824675, gradient last 148934.435563 all 334968.523678; error rate est 0.702647
         p 5380, test: meansq err 2.857400, error rate 0.303438
DEBUG effective rate = 5.23454e-05 (adjusted from 1.37155e-05)
      pass 5381 [5382], meansq err 2.884521, gradient last 510831.424161 all 556083.948828; error rate est 0.261281
         p 5381, test: meansq err 5.867235, error rate 0.900349
(a meltdown)

	// Same as previous, reduced rate to avoid meltdown.
	options.isClassifier_ = true;
	options.correctMultiplier_ = 1. / 1000.;
	options.weightSaturation_ = 1.;
	FloatNeuralNet::LevelSizeVector levels = {64, 64, 10};
	TestFloatNn nn(levels, FloatNeuralNet::CORNER, &options);
	string checkpoint = fpath + "/corner_64_64_10_nofl_c1000.ckp";
	int batchSize = 1;
	seed = 1667553859;
	double rate = 0.03;
	double trainRate = 0.;
	double applyRate = rate;
...
DEBUG effective rate = 0.00028519 (adjusted from 4.11466e-06)
      pass  749 [ 750], meansq err 0.328470, gradient last 402.837681 all 803.363320; error rate est 0.013441
         p  749, test: meansq err 0.369218, error rate 0.121076
         p  749, train: meansq err 0.349950, error rate 0.069949
...
DEBUG effective rate = 0.00361882 (adjusted from 4.11466e-06)
      pass 1955 [1956], meansq err 0.199321, gradient last 8.171046 all 15.000793; error rate est 0.000137
         p 1955, test: meansq err 0.253842, error rate 0.061784
DEBUG effective rate = 0.00411466 (adjusted from 4.11466e-06)
      pass 1956 [1957], meansq err 0.201337, gradient last 5.605237 all 8.643576; error rate est 0.000000
         p 1956, test: meansq err 0.248642, error rate 0.058296

	// Same as previous, less of a penalty for correct cases.
	options.isClassifier_ = true;
	options.correctMultiplier_ = 1. / 100.;
	options.weightSaturation_ = 1.;
	FloatNeuralNet::LevelSizeVector levels = {64, 64, 10};
	TestFloatNn nn(levels, FloatNeuralNet::CORNER, &options);
	string checkpoint = fpath + "/corner_64_64_10_nofl_c100.ckp";
	int batchSize = 1;
	seed = 1667553859;
	double rate = 0.1;
	double trainRate = 0.;
	double applyRate = rate;
...
DEBUG effective rate = 0.000912909 (adjusted from 1.37155e-05)
      pass  786 [ 787], meansq err 0.198411, gradient last 238.476881 all 479.925948; error rate est 0.005075
         p  786, test: meansq err 0.271986, error rate 0.082212
...
DEBUG effective rate = 0.00123716 (adjusted from 1.37155e-05)
      pass  299 [1088], meansq err 0.139206, gradient last 66.935499 all 124.104602; error rate est 0.001097
         p  299, test: meansq err 0.229249, error rate 0.055805
         p  299, train: meansq err 0.161219, error rate 0.001097
...
DEBUG effective rate = 0.00135318 (adjusted from 1.37155e-05)
      pass  638 [1427], meansq err 0.125191, gradient last 25.061059 all 36.006969; error rate est 0.000137
         p  638, test: meansq err 0.210679, error rate 0.051819
...
DEBUG effective rate = 0.00137155 (adjusted from 1.37155e-05)
      pass  799 [1588], meansq err 0.120253, gradient last 23.099127 all 37.486921; error rate est 0.000000
         p  799, test: meansq err 0.208088, error rate 0.051320
         p  799, train: meansq err 0.117968, error rate 0.000411
...
DEBUG effective rate = 0.00131787 (adjusted from 1.37155e-05)
      pass 1263 [2052], meansq err 0.109301, gradient last 41.272546 all 84.739120; error rate est 0.000411
         p 1263, test: meansq err 0.210160, error rate 0.048829
...
DEBUG effective rate = 0.00135318 (adjusted from 1.37155e-05)
      pass 1780 [2569], meansq err 0.103374, gradient last 29.706979 all 56.741817; error rate est 0.000137
         p 1780, test: meansq err 0.204871, error rate 0.049826
DEBUG effective rate = 0.00131787 (adjusted from 1.37155e-05)
      pass 1781 [2570], meansq err 0.100743, gradient last 37.937703 all 78.796437; error rate est 0.000411
         p 1781, test: meansq err 0.209055, error rate 0.049327
DEBUG effective rate = 0.00131787 (adjusted from 1.37155e-05)
      pass 1782 [2571], meansq err 0.109843, gradient last 43.672084 all 96.952454; error rate est 0.000411
         p 1782, test: meansq err 0.206979, error rate 0.050324
DEBUG effective rate = 0.00126823 (adjusted from 1.37155e-05)
      pass 1783 [2572], meansq err 0.105416, gradient last 67.327195 all 153.532082; error rate est 0.000823
         p 1783, test: meansq err 0.222194, error rate 0.050324
DEBUG effective rate = 0.00119332 (adjusted from 1.37155e-05)
      pass 1784 [2573], meansq err 0.138105, gradient last 111.937402 all 306.722446; error rate est 0.001509
         p 1784, test: meansq err 0.218633, error rate 0.053812
DEBUG effective rate = 0.0008967 (adjusted from 1.37155e-05)
      pass 1785 [2574], meansq err 0.132945, gradient last 284.592486 all 695.641777; error rate est 0.005349
         p 1785, test: meansq err 0.325417, error rate 0.167414
DEBUG effective rate = 9.83884e-05 (adjusted from 1.37155e-05)
      pass 1786 [2575], meansq err 0.291005, gradient last 8943.632535 all 22808.282716; error rate est 0.130709
         p 1786, test: meansq err 0.434445, error rate 0.274041
DEBUG effective rate = 5.62215e-05 (adjusted from 1.37155e-05)
      pass 1787 [2576], meansq err 0.415921, gradient last 12235.311898 all 17024.346975; error rate est 0.236319
(a meltdown)

	// here the below/above logic was not in yet
	options.isClassifier_ = true;
	options.correctMultiplier_ = 1. / 100.;
	options.weightSaturation_ = 1.;
	FloatNeuralNet::LevelSizeVector levels = {64, 64, 64, 10};
	TestFloatNn nn(levels, FloatNeuralNet::CORNER, &options);
	string checkpoint = fpath + "/corner_64_64_64_10_nofl_c100.ckp";
	int batchSize = 1;
	seed = 1667553859;
	double rate = 0.03;
	double trainRate = 0.;
	double applyRate = rate;
	../../../../zipnn/corner_64_64_64_10_nofl_c100_1.ckp
...
DEBUG effective rate = 4.06399e-05 (adjusted from 4.11466e-06)
      pass  199 [ 200], meansq err 0.464075, gradient last 1957.247864 all 2972.819586; error rate est 0.092168
         p  199, test: meansq err 0.476077, error rate 0.137020
         p  199, train: meansq err 0.468229, error rate 0.097792
...
DEBUG effective rate = 0.000400588 (adjusted from 4.11466e-06)
      pass 2069 [2070], meansq err 0.137142, gradient last 23.542346 all 54.794093; error rate est 0.000274
         p 2069, test: meansq err 0.217041, error rate 0.057299
         p 2069, train: meansq err 0.140108, error rate 0.000411
...
DEBUG effective rate = 0.000405954 (adjusted from 4.11466e-06)
      pass 2769 [2770], meansq err 0.113964, gradient last 19.699813 all 34.837325; error rate est 0.000137
         p 2769, test: meansq err 0.205727, error rate 0.054808
         p 2769, train: meansq err 0.114206, error rate 0.000000
...
DEBUG effective rate = 0.000411466 (adjusted from 4.11466e-06)
      pass 3269 [3270], meansq err 0.105586, gradient last 19.160176 all 35.405969; error rate est 0.000000
         p 3269, test: meansq err 0.201634, error rate 0.055805
         p 3269, train: meansq err 0.104399, error rate 0.000137
...
DEBUG effective rate = 0.000411466 (adjusted from 4.11466e-06)
      pass 4109 [4110], meansq err 0.093828, gradient last 17.423165 all 31.227375; error rate est 0.000000
         p 4109, test: meansq err 0.198208, error rate 0.053812
         p 4109, train: meansq err 0.092854, error rate 0.000000
...
DEBUG effective rate = 0.000411466 (adjusted from 4.11466e-06)
      pass 13805 [13806], meansq err 0.045923, gradient last 8.616176 all 13.204259; error rate est 0.000000
         p 13805, test: meansq err 0.193362, error rate 0.055306
...
DEBUG effective rate = 0.000411466 (adjusted from 4.11466e-06)
      pass 15989 [15990], meansq err 0.041599, gradient last 7.934130 all 12.093490; error rate est 0.000000
         p 15989, test: meansq err 0.193368, error rate 0.053812
         p 15989, train: meansq err 0.041578, error rate 0.000137
...
DEBUG effective rate = 0.000411466 (adjusted from 4.11466e-06)
      pass 19994 [19995], meansq err 0.035930, gradient last 6.995607 all 10.824498; error rate est 0.000000
         p 19994, test: meansq err 0.193973, error rate 0.056303
DEBUG effective rate = 0.000411466 (adjusted from 4.11466e-06)
      pass 19995 [19996], meansq err 0.035923, gradient last 6.995114 all 10.819489; error rate est 0.000000
         p 19995, test: meansq err 0.193973, error rate 0.056303
DEBUG effective rate = 0.000411466 (adjusted from 4.11466e-06)
      pass 19996 [19997], meansq err 0.035916, gradient last 6.994504 all 10.818127; error rate est 0.000000
         p 19996, test: meansq err 0.193973, error rate 0.056303
DEBUG effective rate = 0.000411466 (adjusted from 4.11466e-06)
      pass 19997 [19998], meansq err 0.035911, gradient last 6.994160 all 10.809908; error rate est 0.000000
         p 19997, test: meansq err 0.193971, error rate 0.056303
DEBUG effective rate = 0.000405954 (adjusted from 4.11466e-06)
      pass 19998 [19999], meansq err 0.035906, gradient last 16.164770 all 67.841675; error rate est 0.000137
         p 19998, test: meansq err 0.195335, error rate 0.055306
DEBUG effective rate = 0.000411466 (adjusted from 4.11466e-06)
      pass 19999 [20000], meansq err 0.041282, gradient last 16.838851 all 41.517742; error rate est 0.000000
         p 19999, test: meansq err 0.194329, error rate 0.055805
         p 19999, train: meansq err 0.037678, error rate 0.000000
...
	// continuation of the last one's 20K passes,
	// now with below/above logic
...
DEBUG effective rate = 0.000411466 (adjusted from 4.11466e-06)
      pass  679 [20748], meansq err 0.035045, gradient last 6.862434 all 10.641285; est error rt 0.000000 below rt 0.351871
         p  679, test: meansq err 0.194182, error rate 0.055805
         p  679, train: meansq err 0.035041, error rate 0.000000
...
DEBUG effective rate = 0.000411466 (adjusted from 4.11466e-06)
      pass  685 [20754], meansq err 0.035022, gradient last 6.860045 all 10.636940; est error rt 0.000000 below rt 0.930656
         p  685, test: meansq err 0.194174, error rate 0.055805
...
DEBUG effective rate = 2.3905e-05 (adjusted from 4.11466e-06)
      pass 8669 [30796], meansq err 0.048536, gradient last 472.772944 all 583.710409; est error rt 0.000000 below rt 0.163764
         p 8669, test: meansq err 0.198513, error rate 0.061784
         p 8669, train: meansq err 0.049041, error rate 0.000000

	// A fresh run with below/above logic
	string checkpoint = fpath + "/corner_64_64_64_10_nofl_c100_ab.ckp";
...
DEBUG effective rate = 2.37364e-05 (adjusted from 4.11466e-06)
      pass 3929 [10438], meansq err 0.144057, gradient last 389.946863 all 497.961773; est error rt 0.001234 below rt 0.164998
         p 3929, test: meansq err 0.226779, error rate 0.061784
         p 3929, train: meansq err 0.143491, error rate 0.001509
...
DEBUG effective rate = 2.3755e-05 (adjusted from 4.11466e-06)
      pass 10739 [17248], meansq err 0.124926, gradient last 277.286366 all 347.741921; est error rt 0.000823 below rt 0.164861
         p 10739, test: meansq err 0.217689, error rate 0.060289
         p 10739, train: meansq err 0.124794, error rate 0.000960
...
DEBUG effective rate = 2.37923e-05 (adjusted from 4.11466e-06)
      pass 14902 [21411], meansq err 0.117515, gradient last 376.153562 all 465.976671; est error rt 0.000411 below rt 0.164586
         p 14902, test: meansq err 0.214276, error rate 0.061286

	options.weightSaturation_ = 1.;
	options.momentum_ = true;
	FloatNeuralNet::LevelSizeVector levels = {64, 64, 64, 10};
	TestFloatNn nn(levels, FloatNeuralNet::CORNER, &options);
	string checkpoint = fpath + "/corner_64_64_64_10_nofl_m.ckp";
	int batchSize = 1;
	seed = 1667553859;
	double rate = 0.03;
	double trainRate = 0.;
	double applyRate = rate;
...
DEBUG effective rate = 4.11466e-06
      pass  239 [ 240], meansq err 0.109919, gradient last 1727.718056 all 3286.624280; est error rt -nan below rt -nan
         p  239, test: meansq err 0.200862, error rate 0.059292
         p  239, train: meansq err 0.109871, error rate 0.009738
...
DEBUG effective rate = 4.11466e-06
      pass  379 [ 380], meansq err 0.080462, gradient last 1720.992593 all 2903.663843; est error rt -nan below rt -nan
         p  379, test: meansq err 0.195923, error rate 0.055805
         p  379, train: meansq err 0.080228, error rate 0.004526
...
DEBUG effective rate = 4.11466e-06
      pass  879 [ 880], meansq err 0.055057, gradient last 1087.523630 all 3015.615507; est error rt -nan below rt -nan
         p  879, test: meansq err 0.186384, error rate 0.051819
         p  879, train: meansq err 0.054875, error rate 0.000960
...
DEBUG effective rate = 4.11466e-06
      pass  987 [ 988], meansq err 0.050036, gradient last 1289.750252 all 3239.812072; est error rt -nan below rt -nan
         p  987, test: meansq err 0.182951, error rate 0.047833
...
DEBUG effective rate = 4.11466e-06
      pass 1175 [1176], meansq err 0.054792, gradient last 5034.128692 all 5809.115274; est error rt -nan below rt -nan
         p 1175, test: meansq err 0.189199, error rate 0.054808
DEBUG effective rate = 4.11466e-06
      pass 1176 [1177], meansq err 0.065226, gradient last 14778.333526 all 15068.922451; est error rt -nan below rt -nan
         p 1176, test: meansq err 0.194365, error rate 0.055306
DEBUG effective rate = 4.11466e-06
      pass 1177 [1178], meansq err 0.122551, gradient last 43655.884568 all 43851.579676; est error rt -nan below rt -nan
         p 1177, test: meansq err 0.234667, error rate 0.055805
DEBUG effective rate = 4.11466e-06
      pass 1178 [1179], meansq err 0.316783, gradient last 117242.643636 all 119555.104952; est error rt -nan below rt -nan
         p 1178, test: meansq err 0.338155, error rate 0.054808
DEBUG effective rate = 4.11466e-06
      pass 1179 [1180], meansq err 0.513169, gradient last 121474.272072 all 140869.866744; est error rt -nan below rt -nan
         p 1179, test: meansq err 0.386260, error rate 0.101644
         p 1179, train: meansq err 0.319903, error rate 0.047181
(and dissolved)

	// momentum + classifier, with classifier rate boost enabled
	options.isClassifier_ = true;
	options.correctMultiplier_ = 1. / 100.;
	options.weightSaturation_ = 1.;
	options.momentum_ = true;
	FloatNeuralNet::LevelSizeVector levels = {64, 64, 64, 10};
	TestFloatNn nn(levels, FloatNeuralNet::CORNER, &options);
	string checkpoint = fpath + "/corner_64_64_64_10_nofl_c100_m.ckp";
	int batchSize = 1;
	seed = 1667553859;
	double rate = 0.03;
	double trainRate = 0.;
	double applyRate = rate;
...
DEBUG effective rate = 2.05912e-05 (adjusted from 4.11466e-06)
      pass  147 [ 148], meansq err 0.444210, gradient last 4820.580709 all 8204.603062; est error rt 0.191743 below rt 0.640241
         p  147, test: meansq err 0.456574, error rate 0.191829
DEBUG effective rate = 1.87328e-05 (adjusted from 4.11466e-06)
      pass  148 [ 149], meansq err 0.445586, gradient last 5376.815840 all 10475.340563; est error rt 0.211768 below rt 0.648882
         p  148, test: meansq err 0.445082, error rate 0.175386
DEBUG effective rate = 2.3156e-05 (adjusted from 4.11466e-06)
      pass  149 [ 150], meansq err 0.430442, gradient last 5711.629738 all 9765.946207; est error rt 0.169387 below rt 0.583733
         p  149, test: meansq err 0.455754, error rate 0.192825
         p  149, train: meansq err 0.436168, error rate 0.140173
DEBUG effective rate = 1.16476e-05 (adjusted from 4.11466e-06)
      pass  150 [ 151], meansq err 0.470984, gradient last 8831.851628 all 18729.247958; est error rt 0.346729 below rt 0.746948
         p  150, test: meansq err 0.436719, error rate 0.182362
DEBUG effective rate = 1.7704e-05 (adjusted from 4.11466e-06)
      pass  151 [ 152], meansq err 0.463190, gradient last 11550.618010 all 25934.831743; est error rt 0.224661 below rt 0.586614
         p  151, test: meansq err 0.495936, error rate 0.314400
DEBUG effective rate = 1.23855e-05 (adjusted from 4.11466e-06)
      pass  152 [ 153], meansq err 0.531670, gradient last 23704.690723 all 31813.453135; est error rt 0.325470 below rt 0.703470
         p  152, test: meansq err 0.476236, error rate 0.280020
DEBUG effective rate = 1.53228e-05 (adjusted from 4.11466e-06)
      pass  153 [ 154], meansq err 0.474623, gradient last 6556.195066 all 9569.653692; est error rt 0.261144 below rt 0.773419
         p  153, test: meansq err 0.484756, error rate 0.280518
DEBUG effective rate = 1.22801e-05 (adjusted from 4.11466e-06)
      pass  154 [ 155], meansq err 0.501314, gradient last 17917.988266 all 23264.700350; est error rt 0.328350 below rt 0.809628
         p  154, test: meansq err 0.507510, error rate 0.359741
DEBUG effective rate = 8.84593e-06 (adjusted from 4.11466e-06)
      pass  155 [ 156], meansq err 0.550011, gradient last 26561.047473 all 33016.490320; est error rt 0.459745 below rt 0.735016
         p  155, test: meansq err 0.500837, error rate 0.294469
DEBUG effective rate = 1.17107e-05 (adjusted from 4.11466e-06)
      pass  156 [ 157], meansq err 0.503898, gradient last 24024.718632 all 36648.699892; est error rt 0.344809 below rt 0.645316
         p  156, test: meansq err 0.604952, error rate 0.543597
DEBUG effective rate = 5.54833e-06 (adjusted from 4.11466e-06)
      pass  157 [ 158], meansq err 0.837345, gradient last 133972.868743 all 171866.377762; est error rt 0.738993 below rt 0.810588
         p  157, test: meansq err 0.497204, error rate 0.399103
(and dissolved)

	// momentum + classifier, with classifier rate boost disabled
	options.isClassifier_ = true;
	options.correctMultiplier_ = 1. / 100.;
	options.weightSaturation_ = 1.;
	options.momentum_ = true;
	FloatNeuralNet::LevelSizeVector levels = {64, 64, 64, 10};
	TestFloatNn nn(levels, FloatNeuralNet::CORNER, &options);
	string checkpoint = fpath + "/corner_64_64_64_10_nofl_c100_m.ckp";
	int batchSize = 1;
	seed = 1667553859;
	double rate = 0.03;
	double trainRate = 0.;
	double applyRate = rate;
	../../../../zipnn/corner_64_64_64_10_nofl_c100_m_1.ckp
...
DEBUG effective rate = 4.11466e-06
      pass  239 [ 240], meansq err 0.321752, gradient last 109.115295 all 396.827708; est error rt 0.015636 below rt 0.366342
         p  239, test: meansq err 0.349051, error rate 0.076233
         p  239, train: meansq err 0.321755, error rate 0.015773

...
DEBUG effective rate = 4.11466e-06
      pass  749 [ 750], meansq err 0.165081, gradient last 69.730183 all 178.200114; est error rt 0.000960 below rt 0.204636
         p  749, test: meansq err 0.234162, error rate 0.060289
         p  749, train: meansq err 0.164905, error rate 0.000686
...
DEBUG effective rate = 4.11466e-06
      pass 1294 [1295], meansq err 0.120049, gradient last 21.560695 all 70.066354; est error rt 0.000000 below rt 0.178851
         p 1294, test: meansq err 0.212042, error rate 0.055805
DEBUG effective rate = 4.11466e-06
      pass 1295 [1296], meansq err 0.119982, gradient last 21.458471 all 70.345219; est error rt 0.000137 below rt 0.178851
         p 1295, test: meansq err 0.212015, error rate 0.055805
DEBUG effective rate = 4.11466e-06
      pass 1296 [1297], meansq err 0.119908, gradient last 53.400365 all 133.693081; est error rt 0.000411 below rt 0.179125
         p 1296, test: meansq err 0.212011, error rate 0.055805
DEBUG effective rate = 4.11466e-06
      pass 1297 [1298], meansq err 0.119875, gradient last 21.790981 all 70.835425; est error rt 0.000137 below rt 0.178988
         p 1297, test: meansq err 0.211996, error rate 0.055805
DEBUG effective rate = 4.11466e-06
      pass 1298 [1299], meansq err 0.119804, gradient last 29.815473 all 94.322429; est error rt 0.000274 below rt 0.179125
         p 1298, test: meansq err 0.212005, error rate 0.056303
DEBUG effective rate = 4.11466e-06
      pass 1299 [1300], meansq err 0.119793, gradient last 35.167638 all 94.995018; est error rt 0.000274 below rt 0.178851
         p 1299, test: meansq err 0.211996, error rate 0.056303
         p 1299, train: meansq err 0.119775, error rate 0.000137
...
DEBUG effective rate = 4.11466e-06
      pass 2029 [2030], meansq err 0.086841, gradient last 15.913973 all 56.769780; est error rt 0.000137 below rt 0.167330
         p 2029, test: meansq err 0.202314, error rate 0.053812
         p 2029, train: meansq err 0.086836, error rate 0.000137
DEBUG effective rate = 4.11466e-06
      pass 2030 [2031], meansq err 0.086824, gradient last 15.944214 all 56.509843; est error rt 0.000000 below rt 0.167330
         p 2030, test: meansq err 0.202316, error rate 0.053812
...
DEBUG effective rate = 4.11466e-06
      pass 2279 [2280], meansq err 0.078914, gradient last 14.378853 all 52.996023; est error rt 0.000137 below rt 0.165272
         p 2279, test: meansq err 0.201320, error rate 0.050822
         p 2279, train: meansq err 0.078911, error rate 0.000137
...
DEBUG effective rate = 4.11466e-06
      pass 3239 [3240], meansq err 0.056277, gradient last 10.790933 all 41.647001; est error rt 0.000000 below rt 0.164038
         p 3239, test: meansq err 0.199503, error rate 0.054310
         p 3239, train: meansq err 0.056274, error rate 0.000000
...
DEBUG effective rate = 4.11466e-06
      pass 3711 [3712], meansq err 0.048736, gradient last 9.853140 all 38.196868; est error rt 0.000000 below rt 0.164038
         p 3711, test: meansq err 0.199033, error rate 0.055306
 ...
(here tried to enable the classifier mode re-scaling again, it drove the mean error up, I gave up and disabled it again)
...
DEBUG effective rate = 4.11466e-06
      pass    0 [5540], meansq err 0.041704, gradient last 8.967612 all 26.058607; est error rt 0.000000 below rt 0.163764
         p    0, test: meansq err 0.197777, error rate 0.052317
...
DEBUG effective rate = 4.11466e-06
      pass 2039 [7584], meansq err 0.034783, gradient last 7.429027 all 22.512740; est error rt 0.000000 below rt 0.163764
         p 2039, test: meansq err 0.197767, error rate 0.052815
         p 2039, train: meansq err 0.034783, error rate 0.000000
...
DEBUG effective rate = 4.11466e-06
      pass 3670 [9215], meansq err 0.027391, gradient last 5.839088 all 18.828163; est error rt 0.000000 below rt 0.163764
         p 3670, test: meansq err 0.198103, error rate 0.051819
...
DEBUG effective rate = 4.11466e-06
      pass 8255 [13800], meansq err 0.015114, gradient last 3.310046 all 12.502701; est error rt 0.000000 below rt 0.163764
         p 8255, test: meansq err 0.202963, error rate 0.059292
...
DEBUG effective rate = 4.11466e-06
      pass 12508 [18053], meansq err 0.010850, gradient last 2.457237 all 11.043738; est error rt 0.000000 below rt 0.163764
         p 12508, test: meansq err 0.205031, error rate 0.061784
...
DEBUG effective rate = 4.11466e-06
      pass 19999 [25544], meansq err 0.007300, gradient last 1.918746 all 10.057642; est error rt 0.000000 below rt 0.163764
         p 19999, test: meansq err 0.211050, error rate 0.062780
         p 19999, train: meansq err 0.007300, error rate 0.000000

	// Same as above but test vector is included into training vector
	// momentum + classifier, with classifier rate boost disabled
	options.isClassifier_ = true;
	options.correctMultiplier_ = 1. / 100.;
	options.weightSaturation_ = 1.;
	options.momentum_ = true;
	FloatNeuralNet::LevelSizeVector levels = {64, 64, 64, 10};
	TestFloatNn nn(levels, FloatNeuralNet::CORNER, &options);
	string checkpoint = fpath + "/corner_64_64_64_10_nofl_c100_m_inc.ckp";
	int batchSize = 1;
	seed = 1667553859;
	double rate = 0.03;
	double trainRate = 0.;
	double applyRate = rate;
	../../../../zipnn/corner_64_64_64_10_nofl_c100_m_inc1.ckp
...
DEBUG effective rate = 3.2265e-06
      pass   49 [  50], meansq err 0.518367, gradient last 611.756743 all 1051.546019; est error rt 0.120564 below rt 0.994407
         p   49, test: meansq err 0.521303, error rate 0.149477
         p   49, train: meansq err 0.518240, error rate 0.119811
...
DEBUG effective rate = 3.2265e-06
      pass  229 [ 230], meansq err 0.341022, gradient last 175.355839 all 698.377122; est error rt 0.027425 below rt 0.409335
         p  229, test: meansq err 0.350435, error rate 0.042850
         p  229, train: meansq err 0.341072, error rate 0.027103
...
DEBUG effective rate = 3.2265e-06
      pass  599 [ 600], meansq err 0.201633, gradient last 71.884034 all 241.201518; est error rt 0.002581 below rt 0.226608
         p  599, test: meansq err 0.217361, error rate 0.006477
         p  599, train: meansq err 0.201620, error rate 0.002796
...
DEBUG effective rate = 3.2265e-06
      pass  999 [1000], meansq err 0.151506, gradient last 64.962067 all 175.106103; est error rt 0.000860 below rt 0.197462
         p  999, test: meansq err 0.169237, error rate 0.001993
         p  999, train: meansq err 0.151454, error rate 0.000968
...
DEBUG effective rate = 3.2265e-06
      pass 1259 [1260], meansq err 0.133075, gradient last 32.034612 all 99.512262; est error rt 0.000108 below rt 0.187782
         p 1259, test: meansq err 0.150366, error rate 0.000000
         p 1259, train: meansq err 0.133053, error rate 0.000215
...
DEBUG effective rate = 3.2265e-06
      pass 1999 [2000], meansq err 0.098837, gradient last 35.646837 all 80.648087; est error rt 0.000215 below rt 0.172188
         p 1999, test: meansq err 0.116233, error rate 0.000498
         p 1999, train: meansq err 0.098830, error rate 0.000323
...
DEBUG effective rate = 3.2265e-06
      pass 2499 [2500], meansq err 0.083546, gradient last 25.780753 all 66.440388; est error rt 0.000108 below rt 0.169929
         p 2499, test: meansq err 0.098066, error rate 0.000000
         p 2499, train: meansq err 0.083543, error rate 0.000108
...
DEBUG effective rate = 3.2265e-06
      pass 3589 [3590], meansq err 0.062204, gradient last 21.387959 all 52.883467; est error rt 0.000000 below rt 0.167993
         p 3589, test: meansq err 0.072668, error rate 0.000000
         p 3589, train: meansq err 0.062195, error rate 0.000000
...
DEBUG effective rate = 3.2265e-06
      pass 3999 [4000], meansq err 0.055253, gradient last 22.259776 all 49.727757; est error rt 0.000108 below rt 0.167778
         p 3999, test: meansq err 0.065483, error rate 0.000498
         p 3999, train: meansq err 0.055245, error rate 0.000108
...
DEBUG effective rate = 3.2265e-06
      pass 4599 [4600], meansq err 0.047417, gradient last 37.522824 all 54.708509; est error rt 0.000108 below rt 0.167670
         p 4599, test: meansq err 0.056079, error rate 0.000000
         p 4599, train: meansq err 0.047414, error rate 0.000108
...
DEBUG effective rate = 3.2265e-06
      pass 4837 [4838], meansq err 0.045093, gradient last 20.120547 all 44.722513; est error rt 0.000108 below rt 0.167563
         p 4837, test: meansq err 0.053097, error rate 0.000498

	// Same as above but try the original 16x16 input
	// momentum + classifier, with classifier rate boost disabled
	options.isClassifier_ = true;
	options.correctMultiplier_ = 1. / 100.;
	options.weightSaturation_ = 1.;
	options.momentum_ = true;
	FloatNeuralNet::LevelSizeVector levels = {256, 64, 64, 10};
	TestFloatNn nn(levels, FloatNeuralNet::CORNER, &options);
	string checkpoint = fpath + "/corner_256_64_64_10_nofl_c100_m.ckp";
	int batchSize = 1;
	seed = 1667553859;
	double rate = 0.03;
	double trainRate = 0.;
	double applyRate = rate;
...
DEBUG effective rate = 4.11466e-06
      pass  249 [ 250], meansq err 0.233552, gradient last 78.199257 all 195.367852; est error rt 0.002743 below rt 0.257578
         p  249, test: meansq err 0.281640, error rate 0.078226
         p  249, train: meansq err 0.233491, error rate 0.002194
...
DEBUG effective rate = 4.11466e-06
      pass  449 [ 450], meansq err 0.171175, gradient last 47.572093 all 145.035486; est error rt 0.000960 below rt 0.210122
         p  449, test: meansq err 0.244099, error rate 0.069258
         p  449, train: meansq err 0.171153, error rate 0.000411
...
DEBUG effective rate = 4.11466e-06
      pass  749 [ 750], meansq err 0.126913, gradient last 20.204678 all 63.512413; est error rt 0.000000 below rt 0.182280
         p  749, test: meansq err 0.216277, error rate 0.054808
         p  749, train: meansq err 0.126906, error rate 0.000000
...
DEBUG effective rate = 4.11466e-06
      pass  999 [1000], meansq err 0.103908, gradient last 18.016325 all 55.246316; est error rt 0.000000 below rt 0.172130
         p  999, test: meansq err 0.207980, error rate 0.053812
         p  999, train: meansq err 0.103902, error rate 0.000000
...
DEBUG effective rate = 4.11466e-06
      pass 1499 [1500], meansq err 0.072085, gradient last 13.456088 all 42.293852; est error rt 0.000000 below rt 0.164724
         p 1499, test: meansq err 0.202587, error rate 0.054310
         p 1499, train: meansq err 0.072082, error rate 0.000000
...
DEBUG effective rate = 4.11466e-06
      pass 1999 [2000], meansq err 0.051475, gradient last 10.275541 all 32.894576; est error rt 0.000000 below rt 0.163901
         p 1999, test: meansq err 0.201613, error rate 0.050324
         p 1999, train: meansq err 0.051473, error rate 0.000000
...
DEBUG effective rate = 4.11466e-06
      pass 2499 [2500], meansq err 0.038443, gradient last 32.496255 all 106.879723; est error rt 0.000137 below rt 0.163901
         p 2499, test: meansq err 0.202677, error rate 0.054310
         p 2499, train: meansq err 0.038439, error rate 0.000000
...
DEBUG effective rate = 4.11466e-06
      pass 2619 [2620], meansq err 0.036031, gradient last 7.674124 all 24.392312; est error rt 0.000000 below rt 0.163764
         p 2619, test: meansq err 0.202955, error rate 0.054310

	// Use 16x16 and 256-neuron first layer.
	options.isClassifier_ = true;
	options.correctMultiplier_ = 1. / 100.;
	options.weightSaturation_ = 1.;
	options.momentum_ = true;
	FloatNeuralNet::LevelSizeVector levels = {256, 256, 64, 10};
	TestFloatNn nn(levels, FloatNeuralNet::CORNER, &options);
	string checkpoint = fpath + "/corner_256_256_64_10_nofl_c100_m.ckp";
	int batchSize = 1;
	seed = 1667553859;
	double rate = 0.03;
	double trainRate = 0.;
	double applyRate = rate;
	../../../../zipnn/corner_256_256_64_10_nofl_c100_m1.ckp
...
DEBUG effective rate = 4.11466e-06
      pass  659 [ 660], meansq err 0.132476, gradient last 20.351858 all 40.011416; est error rt 0.000137 below rt 0.185571
         p  659, test: meansq err 0.223800, error rate 0.061784
         p  659, train: meansq err 0.132469, error rate 0.000137
...
DEBUG effective rate = 4.11466e-06
      pass  749 [ 750], meansq err 0.120148, gradient last 19.379945 all 36.594953; est error rt 0.000137 below rt 0.178851
         p  749, test: meansq err 0.217176, error rate 0.062282
         p  749, train: meansq err 0.120142, error rate 0.000274
...
DEBUG effective rate = 4.11466e-06
      pass  999 [1000], meansq err 0.093136, gradient last 17.052286 all 30.811225; est error rt 0.000137 below rt 0.168015
         p  999, test: meansq err 0.205872, error rate 0.055805
         p  999, train: meansq err 0.093132, error rate 0.000137
...
DEBUG effective rate = 4.11466e-06
      pass 1169 [1170], meansq err 0.078007, gradient last 18.238288 all 29.875166; est error rt 0.000137 below rt 0.164861
         p 1169, test: meansq err 0.201326, error rate 0.053812
         p 1169, train: meansq err 0.078003, error rate 0.000137
...
DEBUG effective rate = 4.11466e-06
      pass 1399 [1400], meansq err 0.061959, gradient last 13.191574 all 24.108516; est error rt 0.000000 below rt 0.163901
         p 1399, test: meansq err 0.197470, error rate 0.053313
         p 1399, train: meansq err 0.061954, error rate 0.000000
...
DEBUG effective rate = 4.11466e-06
      pass 1769 [1770], meansq err 0.043603, gradient last 23.217728 all 28.020464; est error rt 0.000137 below rt 0.163901
         p 1769, test: meansq err 0.193949, error rate 0.053313
         p 1769, train: meansq err 0.043600, error rate 0.000137
...
DEBUG effective rate = 4.11466e-06
      pass 1999 [2000], meansq err 0.036209, gradient last 9.852824 all 19.407439; est error rt 0.000000 below rt 0.163901
         p 1999, test: meansq err 0.192036, error rate 0.053812
         p 1999, train: meansq err 0.036195, error rate 0.000000
...
DEBUG effective rate = 4.11466e-06
      pass 2099 [2100], meansq err 0.033433, gradient last 8.381547 all 15.435712; est error rt 0.000000 below rt 0.163901
         p 2099, test: meansq err 0.191378, error rate 0.053812
         p 2099, train: meansq err 0.033427, error rate 0.000000
...
DEBUG effective rate = 4.11466e-06
      pass 2299 [2300], meansq err 0.028734, gradient last 6.258293 all 12.626835; est error rt 0.000000 below rt 0.163901
         p 2299, test: meansq err 0.190646, error rate 0.050822
         p 2299, train: meansq err 0.028731, error rate 0.000000
...
DEBUG effective rate = 4.11466e-06
      pass 2495 [2496], meansq err 0.025342, gradient last 26.391406 all 27.996925; est error rt 0.000137 below rt 0.163901
         p 2495, test: meansq err 0.190521, error rate 0.050324
...
DEBUG effective rate = 4.11466e-06
      pass 2499 [2500], meansq err 0.025341, gradient last 26.436135 all 28.127242; est error rt 0.000137 below rt 0.163901
         p 2499, test: meansq err 0.190539, error rate 0.050822
         p 2499, train: meansq err 0.025344, error rate 0.000137
...
DEBUG effective rate = 4.11466e-06
      pass 2699 [2700], meansq err 0.022642, gradient last 4.985129 all 9.611365; est error rt 0.000000 below rt 0.163901
         p 2699, test: meansq err 0.190552, error rate 0.049826
         p 2699, train: meansq err 0.022641, error rate 0.000000
...
DEBUG effective rate = 4.11466e-06
      pass 2999 [3000], meansq err 0.020400, gradient last 9.053858 all 13.061322; est error rt 0.000000 below rt 0.163901
         p 2999, test: meansq err 0.190589, error rate 0.050324
         p 2999, train: meansq err 0.020388, error rate 0.000000
...
DEBUG effective rate = 4.11466e-06
      pass 3099 [3100], meansq err 0.019240, gradient last 6.898407 all 9.704142; est error rt 0.000000 below rt 0.163901
         p 3099, test: meansq err 0.190559, error rate 0.050324
         p 3099, train: meansq err 0.019236, error rate 0.000000
...
DEBUG effective rate = 4.11466e-06
      pass 3299 [3300], meansq err 0.017438, gradient last 3.869742 all 7.966525; est error rt 0.000000 below rt 0.163901
         p 3299, test: meansq err 0.190337, error rate 0.050822
         p 3299, train: meansq err 0.017435, error rate 0.000000
...
DEBUG effective rate = 4.11466e-06
      pass 3539 [3540], meansq err 0.016137, gradient last 3.219762 all 6.639490; est error rt 0.000000 below rt 0.163901
         p 3539, test: meansq err 0.190279, error rate 0.050324
         p 3539, train: meansq err 0.016136, error rate 0.000000
...
DEBUG effective rate = 4.11466e-06
      pass 3849 [3850], meansq err 0.014687, gradient last 2.721611 all 5.591731; est error rt 0.000000 below rt 0.163901
         p 3849, test: meansq err 0.189806, error rate 0.048829
         p 3849, train: meansq err 0.014687, error rate 0.000000
...
DEBUG effective rate = 4.11466e-06
      pass 12499 [12500], meansq err 0.008271, gradient last 1.091569 all 2.023718; est error rt 0.000000 below rt 0.163901
         p 12499, test: meansq err 0.187219, error rate 0.049826
         p 12499, train: meansq err 0.008271, error rate 0.000000
...
DEBUG effective rate = 4.11466e-06
      pass 12504 [12505], meansq err 0.008271, gradient last 1.091967 all 2.024580; est error rt 0.000000 below rt 0.163901
         p 12504, test: meansq err 0.187220, error rate 0.049826

	options.weightSaturation_ = 1.;
	options.momentum_ = true;
	options.autoRate2_ = true;
	FloatNeuralNet::LevelSizeVector levels = {64, 64, 64, 10};
	TestFloatNn nn(levels, FloatNeuralNet::CORNER, &options);
	string checkpoint = fpath + "/corner_64_64_64_10_nofl_c100.ckp";
	int batchSize = 1;
	seed = 1667553859;
	double rate = 0.0001;
	double trainRate = 0.;
	double applyRate = rate;
	// autoRate2 implementation 1
...
DEBUG msq grad L1: 82.946/82.946, L2: 55.9694/55.9694, L3: 169.101/169.101,
DEBUG gradient changed sign in 304 of 9307, zero 77; msq chg 4.147/3.31771 unchg 79.0712/78.3967
DEBUG bumped rate adj   up to 2.14401
DEBUG effective rate = 2.94062e-08 (adjusted by autoRate2 from 1.37155e-08)
      pass   99 [ 100], meansq err 0.315332/0.315332, gradient last 4409.620208 all 7938.492478; est ert -nan below rt -nan
         p   99, test: meansq err 0.342920, error rate 0.153463
         p   99, train: meansq err 0.315315, error rate 0.110547
...
DEBUG msq grad L1: 44.0913/44.0913, L2: 37.5942/37.5942, L3: 57.9978/57.9978,
DEBUG gradient changed sign in 230 of 9220, zero 164; msq chg 0.417385/0.601142 unchg 6.46277/6.41485
DEBUG bumped rate adj   up to 12.7838
DEBUG effective rate = 1.75337e-07 (adjusted by autoRate2 from 1.37155e-08)
      pass  399 [ 400], meansq err 0.133522/0.133522, gradient last 1512.398743 all 4110.794098; est ert -nan below rt -nan
         p  399, test: meansq err 0.208311, error rate 0.059791
         p  399, train: meansq err 0.133520, error rate 0.018790
...
DEBUG msq grad L1: 12.7775/12.7775, L2: 18.2158/18.2158, L3: 19.9423/19.9423,
DEBUG gradient changed sign in 199 of 9277, zero 107; msq chg 0.235553/0.385813 unchg 1.37402/1.38151
DEBUG bumped rate adj   up to 0.558011
DEBUG effective rate = 7.65342e-09 (adjusted by autoRate2 from 1.37155e-08)
      pass 2699 [2700], meansq err 0.055427/0.055427, gradient last 520.031765 all 1557.247192; est ert -nan below rt -nan
         p 2699, test: meansq err 0.195272, error rate 0.057798
         p 2699, train: meansq err 0.055427, error rate 0.004115
...
DEBUG msq grad L1: 10.2054/10.2054, L2: 15.7045/15.7045, L3: 15.1973/15.1973,
DEBUG gradient changed sign in 479 of 9308, zero 76; msq chg 0.977235/0.376659 unchg 1.63105/1.62432
DEBUG bumped rate adj   up to 0.149083
DEBUG effective rate = 2.04476e-09 (adjusted by autoRate2 from 1.37155e-08)
      pass 9419 [9420], meansq err 0.050494/0.050494, gradient last 396.297209 all 1297.557693; est ert -nan below rt -nan
         p 9419, test: meansq err 0.195207, error rate 0.057299
         p 9419, train: meansq err 0.050494, error rate 0.003978
...
DEBUG msq grad L1: 9.34345/9.34345, L2: 14.7413/14.7413, L3: 13.787/13.787,
DEBUG gradient changed sign in 53 of 9340, zero 44; msq chg 0.0858304/0.205562 unchg 3.2977/3.29679
DEBUG bumped rate adj   up to 0.0627961
DEBUG effective rate = 8.61282e-10 (adjusted by autoRate2 from 1.37155e-08)
      pass 14299 [14300], meansq err 0.048939/0.048939, gradient last 359.521391 all 1206.194259; est ert -nan below rt -nan
         p 14299, test: meansq err 0.195829, error rate 0.057299
         p 14299, train: meansq err 0.048939, error rate 0.003566
...
DEBUG msq grad L1: 9.26746/9.26746, L2: 14.6495/14.6495, L3: 13.5802/13.5802,
DEBUG gradient changed sign in 588 of 9331, zero 53; msq chg 0.421153/0.871476 unchg 2.57391/2.59025
DEBUG bumped rate adj Down to 0.0394666
DEBUG effective rate = 5.41306e-10 (adjusted by autoRate2 from 1.37155e-08)
      pass 15899 [15900], meansq err 0.048757/0.048757, gradient last 354.128542 all 1197.142336; est ert -nan below rt -nan
         p 15899, test: meansq err 0.195889, error rate 0.057299
         p 15899, train: meansq err 0.048757, error rate 0.003703
...
DEBUG msq grad L1: 9.17612/9.17612, L2: 14.6069/14.6069, L3: 13.4365/13.4365,
DEBUG gradient changed sign in 153 of 9361, zero 23; msq chg 0.362908/0.284793 unchg 5.62565/5.62808
DEBUG bumped rate adj   up to 0.0399868
DEBUG effective rate = 5.4844e-10 (adjusted by autoRate2 from 1.37155e-08)
      pass 17959 [17960], meansq err 0.048603/0.048603, gradient last 350.381261 all 1190.695706; est ert -nan below rt -nan
         p 17959, test: meansq err 0.195986, error rate 0.057798
         p 17959, train: meansq err 0.048603, error rate 0.003703
...
DEBUG msq grad L1: 9.13372/9.13372, L2: 14.5825/14.5825, L3: 13.2865/13.2865,
DEBUG gradient changed sign in 102 of 9342, zero 42; msq chg 0.267465/0.213833 unchg 3.37421/3.37482
DEBUG bumped rate adj   up to 0.0641233
DEBUG effective rate = 8.79486e-10 (adjusted by autoRate2 from 1.37155e-08)
      pass 19999 [20000], meansq err 0.048471/0.048471, gradient last 346.470119 all 1186.823892; est ert -nan below rt -nan
         p 19999, test: meansq err 0.196084, error rate 0.058794
         p 19999, train: meansq err 0.048471, error rate 0.003703
...
DEBUG msq grad L1: 9.07309/9.07309, L2: 14.58/14.58, L3: 13.2481/13.2481,
DEBUG gradient changed sign in 58 of 9277, zero 107; msq chg 0.766759/1.04437 unchg 0.807965/0.811681
DEBUG bumped rate adj   up to 0.439383
DEBUG effective rate = 6.02637e-09 (adjusted by autoRate2 from 1.37155e-08)
      pass 20499 [20500], meansq err 0.048445/0.048445, gradient last 345.468909 all 1184.373097; est ert -nan below rt -nan
         p 20499, test: meansq err 0.196123, error rate 0.058794
         p 20499, train: meansq err 0.048445, error rate 0.003703

	options.weightSaturation_ = 1.;
	options.momentum_ = true;
	options.autoRate2_ = true;
	FloatNeuralNet::LevelSizeVector levels = {64, 64, 64, 10};
	TestFloatNn nn(levels, FloatNeuralNet::CORNER, &options);
	string checkpoint = fpath + "/corner_64_64_64_10_nofl_c100.ckp";
	int batchSize = 1;
	seed = 1667553859;
	double rate = 0.0001;
	double trainRate = 0.;
	double applyRate = rate;
	// autoRate2 implementation 2
...
DEBUG msq grad L1: 3.64868/3.64868, L2: 7.9566/7.9566, L3: 5.29273/5.29273,
DEBUG gradient changed sign in 1233 of 9279, zero 105; msq chg 0.231182/0.35964 unchg 0.286098/0.311893
DEBUG bumped rate adj   up to 0.935709
DEBUG effective rate = 1.28338e-08 (adjusted by autoRate2 from 1.37155e-08)
      pass 20499 [20500], meansq err 0.042447/0.042447, gradient last 138.017472 all 593.717878; est ert -nan below rt -nan
         p 20499, test: meansq err 0.191676, error rate 0.056801
         p 20499, train: meansq err 0.042447, error rate 0.003429

	// Pulling back from a deeply diverged (to all 1 and -1) case
	options.weightSaturation_ = 1.;
	options.momentum_ = true;
	options.autoRate2_ = true;
	options.tweakRate_ = 0.3;
	FloatNeuralNet::LevelSizeVector levels = {64, 64, 64, 10};
	TestFloatNn nn(levels, FloatNeuralNet::CORNER, &options);
	string checkpoint = fpath + "/corner_64_64_64_10_nofl_c100.ckp";
	int batchSize = 1;
	double rate = 0.0001;
	double trainRate = 0.;
	double applyRate = rate;
	// autoRate2 implementation 2
	../../../../zipnn/corner_64_64_64_10_nofl_c100_bad.ckp
	../../../../zipnn/corner_64_64_64_10_nofl_c100_bad2.ckp
	../../../../zipnn/corner_64_64_64_10_nofl_c100_bad3.ckp
...
DEBUG msq grad L1: 59.2435/59.2435, L2: 130.961/130.961, L3: 109491/109491,
DEBUG gradient changed sign in 401 of 9237, zero 147; msq chg 40.7312/394.231 unchg 4.45459/5.11216
DEBUG bumped rate adj Down to 0.5
DEBUG effective rate = 6.85777e-09 (adjusted by autoRate2 from 1.37155e-08)
      pass    1 [40329], meansq err 0.575100/0.575100, gradient last 2855185.935799 all 2855201.681600; est ert -nan below rt -nan
         p    1, test: meansq err 0.573923, error rate 0.702043
...
DEBUG msq grad L1: 2259.18/2259.18, L2: 1243.93/1243.93, L3: 339.659/339.659,
DEBUG gradient changed sign in 768 of 9209, zero 175; msq chg 2.38237/3.04311 unchg 2.78813/2.96375
DEBUG bumped rate adj   up to 1.1
DEBUG effective rate = 1.50871e-08 (adjusted by autoRate2 from 1.37155e-08)
      pass    1 [182206], meansq err 0.180359/0.180359, gradient last 8857.228859 all 170366.615523; est ert -nan below rt -nan
         p    1, test: meansq err 0.255161, error rate 0.085700
...
DEBUG msq grad L1: 2330.36/2330.36, L2: 1416.95/1416.95, L3: 187.113/187.113,
DEBUG gradient changed sign in 780 of 9238, zero 146; msq chg 8.63405/8.14482 unchg 5.46952/5.39473
DEBUG bumped rate adj   up to 0.0182547
DEBUG effective rate = 2.50373e-10 (adjusted by autoRate2 from 1.37155e-08)
      pass 116809 [299117], meansq err 0.150876/0.150876, gradient last 4879.312485 all 179987.366092; est ert -nan below rt -nan
         p 116809, test: meansq err 0.246730, error rate 0.082212
         p 116809, train: meansq err 0.150876, error rate 0.022768

	// same options as above, including the high tweak rate, but train from random
	options.weightSaturation_ = 1.;
	options.momentum_ = true;
	options.autoRate2_ = true;
	options.tweakRate_ = 0.3;
	FloatNeuralNet::LevelSizeVector levels = {64, 64, 64, 10};
	TestFloatNn nn(levels, FloatNeuralNet::CORNER, &options);
	string checkpoint = fpath + "/corner_64_64_64_10_nofl_c100.ckp";
	int batchSize = 1;
	seed = 1667553859;
	double rate = 0.0001;
	double trainRate = 0.;
	double applyRate = rate;
	// autoRate2 implementation 2
...
DEBUG msq grad L1: 32.2575/32.2575, L2: 33.7611/33.7611, L3: 64.281/64.281,
DEBUG gradient changed sign in 1202 of 9225, zero 159; msq chg 2.72754/2.36293 unchg 6.48657/6.64017
DEBUG bumped rate adj   up to 252.252
DEBUG effective rate = 3.45977e-06 (adjusted by autoRate2 from 1.37155e-08)
      pass  199 [ 200], meansq err 0.135668/0.135668, gradient last 1676.242293 all 3506.951026; est ert -nan below rt -nan
         p  199, test: meansq err 0.210147, error rate 0.063777
         p  199, train: meansq err 0.135636, error rate 0.018927
...
DEBUG msq grad L1: 13.2779/13.2779, L2: 20.9075/20.9075, L3: 28.2421/28.2421,
DEBUG gradient changed sign in 2390 of 9152, zero 232; msq chg 1.2259/1.76153 unchg 1.92153/2.02737
DEBUG bumped rate adj   up to 321.752
DEBUG effective rate = 4.41301e-06 (adjusted by autoRate2 from 1.37155e-08)
      pass  549 [ 550], meansq err 0.067245/0.067245, gradient last 736.463893 all 1792.208722; est ert -nan below rt -nan
         p  549, test: meansq err 0.198877, error rate 0.057798
         p  549, train: meansq err 0.067238, error rate 0.004800
...
DEBUG msq grad L1: 9.04431/9.04431, L2: 16.7629/16.7629, L3: 17.6048/17.6048,
DEBUG gradient changed sign in 1444 of 9125, zero 259; msq chg 0.750103/0.616687 unchg 0.824688/0.793929
DEBUG bumped rate adj   up to 193.579
DEBUG effective rate = 2.65504e-06 (adjusted by autoRate2 from 1.37155e-08)
      pass  999 [1000], meansq err 0.054017/0.054017, gradient last 459.076929 all 1337.771004; est ert -nan below rt -nan
         p  999, test: meansq err 0.200068, error rate 0.056801
         p  999, train: meansq err 0.054016, error rate 0.004115
...
DEBUG msq grad L1: 7.86771/7.86771, L2: 15.6848/15.6848, L3: 14.5733/14.5733,
DEBUG gradient changed sign in 2343 of 9118, zero 266; msq chg 0.819511/0.918753 unchg 0.890002/0.929428
DEBUG bumped rate adj   up to 238.478
DEBUG effective rate = 3.27085e-06 (adjusted by autoRate2 from 1.37155e-08)
      pass 1299 [1300], meansq err 0.051008/0.051008, gradient last 380.025247 all 1218.385357; est ert -nan below rt -nan
         p 1299, test: meansq err 0.199765, error rate 0.055805
         p 1299, train: meansq err 0.051006, error rate 0.004115
...
DEBUG msq grad L1: 5.68254/5.68254, L2: 12.4063/12.4063, L3: 10.7234/10.7234,
DEBUG gradient changed sign in 1696 of 9120, zero 264; msq chg 0.497772/0.507501 unchg 0.409643/0.416605
DEBUG bumped rate adj   up to 36.438
DEBUG effective rate = 4.99767e-07 (adjusted by autoRate2 from 1.37155e-08)
      pass 1999 [2000], meansq err 0.044556/0.044556, gradient last 279.631824 all 942.637851; est ert -nan below rt -nan
         p 1999, test: meansq err 0.199315, error rate 0.056303
         p 1999, train: meansq err 0.044556, error rate 0.003703
...
DEBUG msq grad L1: 5.03781/5.03781, L2: 11.3198/11.3198, L3: 9.42125/9.42125,
DEBUG gradient changed sign in 1699 of 9187, zero 197; msq chg 0.556706/0.637283 unchg 0.322416/0.334935
DEBUG bumped rate adj Down to 5.19356
DEBUG effective rate = 7.12324e-08 (adjusted by autoRate2 from 1.37155e-08)
      pass 2599 [2600], meansq err 0.043518/0.043518, gradient last 245.676084 all 853.500674; est ert -nan below rt -nan
         p 2599, test: meansq err 0.198989, error rate 0.056303
         p 2599, train: meansq err 0.043518, error rate 0.003429
...
DEBUG msq grad L1: 4.748/4.748, L2: 10.7712/10.7712, L3: 8.73316/8.73316,
DEBUG gradient changed sign in 337 of 9265, zero 119; msq chg 0.855933/0.369566 unchg 0.341192/0.340688
DEBUG bumped rate adj   up to 1.7458
DEBUG effective rate = 2.39446e-08 (adjusted by autoRate2 from 1.37155e-08)
      pass 3299 [3300], meansq err 0.043037/0.043037, gradient last 227.732968 all 809.249862; est ert -nan below rt -nan
         p 3299, test: meansq err 0.198758, error rate 0.055805
         p 3299, train: meansq err 0.043037, error rate 0.003292
...
DEBUG msq grad L1: 4.62107/4.62107, L2: 10.5843/10.5843, L3: 8.45653/8.45653,
DEBUG gradient changed sign in 418 of 9274, zero 110; msq chg 0.905996/0.525454 unchg 0.327563/0.328168
DEBUG bumped rate adj   up to 1.20434
DEBUG effective rate = 1.65182e-08 (adjusted by autoRate2 from 1.37155e-08)
      pass 3899 [3900], meansq err 0.042860/0.042860, gradient last 220.519209 all 793.163404; est ert -nan below rt -nan
         p 3899, test: meansq err 0.198663, error rate 0.056303
         p 3899, train: meansq err 0.042860, error rate 0.0034
...
DEBUG msq grad L1: 4.4403/4.4403, L2: 10.3796/10.3796, L3: 8.10768/8.10768,
DEBUG gradient changed sign in 1286 of 9267, zero 117; msq chg 0.348693/0.297658 unchg 0.34402/0.325114
DEBUG bumped rate adj   up to 1.55336
DEBUG effective rate = 2.13052e-08 (adjusted by autoRate2 from 1.37155e-08)
      pass 5399 [5400], meansq err 0.042650/0.042650, gradient last 211.422317 all 774.193863; est ert -nan below rt -nan
         p 5399, test: meansq err 0.198430, error rate 0.056303
         p 5399, train: meansq err 0.042650, error rate 0.003292
...
DEBUG msq grad L1: 4.25999/4.25999, L2: 10.0631/10.0631, L3: 7.71909/7.71909,
DEBUG gradient changed sign in 1122 of 9280, zero 104; msq chg 0.498451/0.565139 unchg 0.355452/0.358174
DEBUG bumped rate adj   up to 0.716778
DEBUG effective rate = 9.83099e-09 (adjusted by autoRate2 from 1.37155e-08)
      pass 8369 [8370], meansq err 0.042429/0.042429, gradient last 201.289251 all 748.470495; est ert -nan below rt -nan
         p 8369, test: meansq err 0.198155, error rate 0.055805
         p 8369, train: meansq err 0.042429, error rate 0.003292
...
DEBUG msq grad L1: 3.32988/3.32988, L2: 8.46926/8.46926, L3: 6.39314/6.39314,
DEBUG gradient changed sign in 128 of 9284, zero 100; msq chg 0.169759/0.147668 unchg 0.347756/0.343093
DEBUG bumped rate adj   up to 0.677332
DEBUG effective rate = 9.28997e-09 (adjusted by autoRate2 from 1.37155e-08)
      pass 22349 [22350], meansq err 0.041305/0.041305, gradient last 166.712634 all 623.065374; est ert -nan below rt -nan
         p 22349, test: meansq err 0.196549, error rate 0.054808
         p 22349, train: meansq err 0.041305, error rate 0.003566

	// bwImage() with cutoff at -0.1
	options.weightSaturation_ = 1.;
	options.momentum_ = true;
	options.autoRate2_ = true;
	options.tweakRate_ = 0.3;
	FloatNeuralNet::LevelSizeVector levels = {256, 64, 64, 10};
	TestFloatNn nn(levels, FloatNeuralNet::CORNER, &options);
	string checkpoint = fpath + "/corner_256_64_64_10_nofl_bw.ckp";
	int batchSize = 1;
	seed = 1667553859;
	double rate = 0.0001;
	double trainRate = 0.;
	double applyRate = rate;
...
DEBUG msq grad L1: 8.99991/8.99991, L2: 20.0613/20.0613, L3: 25.3038/25.3038, 
DEBUG gradient changed sign in 2044 of 21468, zero 204; msq chg 0.53449/0.529434 unchg 1.81483/1.79975
DEBUG bumped rate adj   up to 201.428
DEBUG effective rate = 2.7627e-06 (adjusted by autoRate2 from 1.37155e-08)
      pass  329 [ 385], meansq err 0.068074/0.068074, gradient last 659.842463 all 1880.076303; est ert -nan below rt -nan
         p  329, test: meansq err 0.218084, error rate 0.064275
         p  329, train: meansq err 0.068066, error rate 0.006172
...
DEBUG msq grad L1: 4.77595/4.77595, L2: 12.3213/12.3213, L3: 12.894/12.894,
DEBUG gradient changed sign in 3084 of 21458, zero 214; msq chg 0.227558/0.242686 unchg 0.636272/0.615756
DEBUG bumped rate adj   up to 279.354
DEBUG effective rate = 3.83149e-06 (adjusted by autoRate2 from 1.37155e-08)
      pass  299 [ 685], meansq err 0.055492/0.055492, gradient last 336.234210 all 1073.920942; est ert -nan below rt -nan
         p  299, test: meansq err 0.218860, error rate 0.068261
         p  299, train: meansq err 0.055491, error rate 0.006172
...
DEBUG msq grad L1: 3.02677/3.02677, L2: 8.77634/8.77634, L3: 9.32555/9.32555,
DEBUG gradient changed sign in 4384 of 21475, zero 197; msq chg 0.217608/0.207883 unchg 0.380167/0.380193
DEBUG bumped rate adj   up to 217.435
DEBUG effective rate = 2.98224e-06 (adjusted by autoRate2 from 1.37155e-08)
      pass  619 [1005], meansq err 0.052357/0.052357, gradient last 243.180646 all 739.452959; est ert -nan below rt -nan
         p  619, test: meansq err 0.219237, error rate 0.067265
         p  619, train: meansq err 0.052357, error rate 0.005623

	// Same as last one but not BW
	options.weightSaturation_ = 1.;
	options.momentum_ = true;
	options.autoRate2_ = true;
	options.tweakRate_ = 0.3;
	FloatNeuralNet::LevelSizeVector levels = {256, 64, 64, 10};
	TestFloatNn nn(levels, FloatNeuralNet::CORNER, &options);
	string checkpoint = fpath + "/corner_256_64_64_10_nofl_ar2.ckp";
	int batchSize = 1;
	seed = 1667553859;
	double rate = 0.0001;
	double trainRate = 0.;
	double applyRate = rate;
...
DEBUG msq grad L1: 22.6584/22.6584, L2: 33.8506/33.8506, L3: 56.4973/56.4973, 
DEBUG gradient changed sign in 3150 of 21510, zero 162; msq chg 3.70655/3.38641 unchg 9.4442/8.73529
DEBUG bumped rate adj   up to 213.48
DEBUG effective rate = 2.92799e-06 (adjusted by autoRate2 from 1.37155e-08)
      pass  139 [ 140], meansq err 0.155634/0.155634, gradient last 1473.269953 all 3962.368154; est ert -nan below rt -nan
         p  139, test: meansq err 0.230660, error rate 0.072247
         p  139, train: meansq err 0.155571, error rate 0.024139
...
DEBUG msq grad L1: 10.1914/10.1914, L2: 22.2254/22.2254, L3: 31.3882/31.3882,
DEBUG gradient changed sign in 2313 of 21483, zero 189; msq chg 0.608495/0.525301 unchg 2.65371/2.44195
DEBUG bumped rate adj   up to 201.428
DEBUG effective rate = 2.7627e-06 (adjusted by autoRate2 from 1.37155e-08)
      pass  329 [ 330], meansq err 0.069122/0.069122, gradient last 818.504868 all 2132.603152; est ert -nan below rt -nan
         p  329, test: meansq err 0.197739, error rate 0.057798
         p  329, train: meansq err 0.069106, error rate 0.004663
...
DEBUG msq grad L1: 5.34545/5.34545, L2: 14.583/14.583, L3: 14.9796/14.9796, 
DEBUG gradient changed sign in 6332 of 21440, zero 232; msq chg 0.421405/0.590819 unchg 0.644466/0.696418
DEBUG bumped rate adj   up to 248.147
DEBUG effective rate = 3.40347e-06 (adjusted by autoRate2 from 1.37155e-08)
      pass  629 [ 630], meansq err 0.049098/0.049098, gradient last 390.620400 all 1246.423378; est ert -nan below rt -nan
         p  629, test: meansq err 0.195757, error rate 0.056801
         p  629, train: meansq err 0.049097, error rate 0.004115
...
DEBUG msq grad L1: 3.44293/3.44293, L2: 10.8972/10.8972, L3: 10.1237/10.1237,
DEBUG gradient changed sign in 2846 of 21455, zero 217; msq chg 0.141965/0.134033 unchg 0.312058/0.304509
DEBUG bumped rate adj   up to 193.579
DEBUG effective rate = 2.65504e-06 (adjusted by autoRate2 from 1.37155e-08)
      pass  999 [1000], meansq err 0.045429/0.045429, gradient last 263.993645 all 885.289427; est ert -nan below rt -nan
         p  999, test: meansq err 0.196174, error rate 0.055306
         p  999, train: meansq err 0.045428, error rate 0.004115

	// bwImage() with cutoff at -0.1 and runLenImage()
	options.weightSaturation_ = 1.;
	options.momentum_ = true;
	options.autoRate2_ = true;
	options.tweakRate_ = 0.3;
	FloatNeuralNet::LevelSizeVector levels = {256, 64, 64, 10};
	TestFloatNn nn(levels, FloatNeuralNet::CORNER, &options);
	string checkpoint = fpath + "/corner_256_64_64_10_nofl_bw.ckp";
	int batchSize = 1;
	seed = 1667553859;
	double rate = 0.0001;
	double trainRate = 0.;
	double applyRate = rate;
...
DEBUG msq grad L1: 56.8577/56.8577, L2: 86.5163/86.5163, L3: 55.1453/55.1453,
DEBUG gradient changed sign in 2390 of 12974, zero 8698; msq chg 25.8087/21.2801 unchg 21.9726/17.9571
DEBUG bumped rate adj   up to 52.1181
DEBUG effective rate = 7.14828e-07 (adjusted by autoRate2 from 1.37155e-08)
      pass  199 [ 200], meansq err 0.189204/0.189204, gradient last 1438.013657 all 9404.077999; est ert -nan below rt -nan
         p  199, test: meansq err 0.246015, error rate 0.082711
         p  199, train: meansq err 0.189157, error rate 0.039226
...
DEBUG msq grad L1: 47.4321/47.4321, L2: 59.824/59.824, L3: 48.146/48.146,
DEBUG gradient changed sign in 1107 of 12924, zero 8748; msq chg 3.7852/4.16393 unchg 6.49821/6.43504
DEBUG bumped rate adj   up to 43.0897
DEBUG effective rate = 5.90998e-07 (adjusted by autoRate2 from 1.37155e-08)
      pass  379 [ 380], meansq err 0.125060/0.125060, gradient last 1255.495194 all 7388.397988; est ert -nan below rt -nan
         p  379, test: meansq err 0.222909, error rate 0.075735
         p  379, train: meansq err 0.125052, error rate 0.014264
...
DEBUG msq grad L1: 36.3403/36.3403, L2: 49.1917/49.1917, L3: 35.9798/35.9798,
DEBUG gradient changed sign in 1315 of 12922, zero 8750; msq chg 3.74589/3.4603 unchg 5.02799/4.93067
DEBUG bumped rate adj   up to 23.3046
DEBUG effective rate = 3.19635e-07 (adjusted by autoRate2 from 1.37155e-08)
      pass  629 [ 630], meansq err 0.087573/0.087573, gradient last 938.238958 all 5778.096647; est ert -nan below rt -nan
         p  629, test: meansq err 0.220225, error rate 0.073742
         p  629, train: meansq err 0.087570, error rate 0.007132
...
DEBUG msq grad L1: 29.838/29.838, L2: 44.4804/44.4804, L3: 25.463/25.463, 
DEBUG gradient changed sign in 1499 of 12910, zero 8762; msq chg 3.2494/3.12045 unchg 2.63048/2.66523
DEBUG bumped rate adj   up to 18.7808
DEBUG effective rate = 2.57589e-07 (adjusted by autoRate2 from 1.37155e-08)
      pass  999 [1000], meansq err 0.072638/0.072638, gradient last 663.994865 all 4885.282117; est ert -nan below rt -nan
         p  999, test: meansq err 0.222944, error rate 0.072745
         p  999, train: meansq err 0.072637, error rate 0.006035
...
DEBUG msq grad L1: 26.1817/26.1817, L2: 40.747/40.747, L3: 20.5001/20.5001,
DEBUG gradient changed sign in 1382 of 12931, zero 8741; msq chg 2.24364/2.13508 unchg 1.55164/1.59138
DEBUG bumped rate adj   up to 6.25543
DEBUG effective rate = 8.57966e-08 (adjusted by autoRate2 from 1.37155e-08)
      pass 1459 [1460], meansq err 0.065404/0.065404, gradient last 534.576008 all 4349.471887; est ert -nan below rt -nan
         p 1459, test: meansq err 0.225015, error rate 0.071749
         p 1459, train: meansq err 0.065404, error rate 0.005761

	// bwImage() with cutoff at -0.1 and runLenImage() for 8x8 but a larger first layer
	options.weightSaturation_ = 1.;
	options.momentum_ = true;
	options.autoRate2_ = true;
	options.tweakRate_ = 0.3;
	FloatNeuralNet::LevelSizeVector levels = {64, 256, 64, 10};
	TestFloatNn nn(levels, FloatNeuralNet::CORNER, &options);
	string checkpoint = fpath + "/corner_64_256_64_10_nofl_rl.ckp";
	int batchSize = 1;
	seed = 1667553859;
	double rate = 0.0001;
	double trainRate = 0.;
	double applyRate = rate;
...
DEBUG msq grad L1: 26.8768/26.8768, L2: 29.9754/29.9754, L3: 65.3434/65.3434,
DEBUG gradient changed sign in 2004 of 31215, zero 3513; msq chg 2.54504/2.03337 unchg 12.5542/12.3074
DEBUG bumped rate adj   up to 118.45
DEBUG effective rate = 1.62461e-06 (adjusted by autoRate2 from 1.37155e-08)
      pass  199 [ 200], meansq err 0.215205/0.215205, gradient last 1703.947740 all 5516.320921; est ert -nan below rt -nan
         p  199, test: meansq err 0.281841, error rate 0.121076
         p  199, train: meansq err 0.215145, error rate 0.058154

	// bwImage() with cutoff at -0.1 and runLenImage()
	options.weightSaturation_ = 1.;
	options.momentum_ = true;
	options.autoRate2_ = true;
	options.tweakRate_ = 0.3;
	FloatNeuralNet::LevelSizeVector levels = {256, 256, 64, 10};
	TestFloatNn nn(levels, FloatNeuralNet::CORNER, &options);
	string checkpoint = fpath + "/corner_256_256_64_10_nofl_rl.ckp";
	int batchSize = 1;
	seed = 1667553859;
	double rate = 0.0001;
	double trainRate = 0.;
	double applyRate = rate;
...
DEBUG msq grad L1: 1.71569/1.71569, L2: 3.65917/3.65917, L3: 4.86368/4.86368,
DEBUG gradient changed sign in 7294 of 49497, zero 34383; msq chg 0.44823/0.408903 unchg 0.264732/0.258281
DEBUG bumped rate adj   up to 13.1665
DEBUG effective rate = 1.80586e-07 (adjusted by autoRate2 from 1.37155e-08)
      pass 2029 [2030], meansq err 0.043531/0.043531, gradient last 126.829313 all 659.403469; est ert -nan below rt -nan
         p 2029, test: meansq err 0.213207, error rate 0.067763
         p 2029, train: meansq err 0.043531, error rate 0.003703
...
DEBUG msq grad L1: 0.765684/0.765684, L2: 1.86222/1.86222, L3: 4.25269/4.25269,
DEBUG gradient changed sign in 3618 of 49567, zero 34313; msq chg 0.147616/0.145932 unchg 0.135801/0.135583
DEBUG bumped rate adj   up to 2.02878
DEBUG effective rate = 2.78258e-08 (adjusted by autoRate2 from 1.37155e-08)
      pass 10679 [10680], meansq err 0.039417/0.039417, gradient last 110.896637 all 330.190068; est ert -nan below rt -nan
         p 10679, test: meansq err 0.213033, error rate 0.066766
         p 10679, train: meansq err 0.039417, error rate 0.003017
DEBUG msq grad L1: 0.769682/0.769682, L2: 1.86446/1.86446, L3: 4.25277/4.25277, 
DEBUG gradient changed sign in 6996 of 49559, zero 34321; msq chg 0.20993/0.313482 unchg 0.114636/0.121068
DEBUG bumped rate adj Down to 1.01439
DEBUG effective rate = 1.39129e-08 (adjusted by autoRate2 from 1.37155e-08)
      pass 10680 [10681], meansq err 0.039417/0.039417, gradient last 110.898761 all 331.018810; est ert -nan below rt -nan
         p 10680, test: meansq err 0.213033, error rate 0.066766
DEBUG msq grad L1: 0.768416/0.768416, L2: 1.86347/1.86347, L3: 4.25286/4.25286, 
DEBUG gradient changed sign in 4789 of 49631, zero 34249; msq chg 0.215294/0.169671 unchg 0.214691/0.207427
DEBUG bumped rate adj   up to 1.11583
DEBUG effective rate = 1.53042e-08 (adjusted by autoRate2 from 1.37155e-08)
      pass 10681 [10682], meansq err 0.039417/0.039417, gradient last 110.901003 all 330.731337; est ert -nan below rt -nan
         p 10681, test: meansq err 0.213033, error rate 0.066766

	// bwImage() with cutoff at -0.1 and runLenImage(), now with -1 instead of 0 for unused parts
	options.weightSaturation_ = 1.;
	options.momentum_ = true;
	options.autoRate2_ = true;
	options.tweakRate_ = 0.3;
	FloatNeuralNet::LevelSizeVector levels = {256, 256, 64, 10};
	TestFloatNn nn(levels, FloatNeuralNet::CORNER, &options);
	string checkpoint = fpath + "/corner_256_256_64_10_nofl_rl.ckp";
	int batchSize = 1;
	seed = 1667553859;
	double rate = 0.0001;
	double trainRate = 0.;
	double applyRate = rate;
	../../../../zipnn/corner_256_256_64_10_nofl_rl.ckp
...
DEBUG msq grad L1: 9.80266/9.80266, L2: 20.3907/20.3907, L3: 46.9361/46.9361,
DEBUG gradient changed sign in 6215 of 83504, zero 376; msq chg 0.453124/0.493407 unchg 6.95969/6.98342
DEBUG bumped rate adj   up to 53.8411
DEBUG effective rate = 7.38459e-07 (adjusted by autoRate2 from 1.37155e-08)
      pass  199 [ 200], meansq err 0.155579/0.155579, gradient last 1223.944874 all 3848.700981; est ert -nan below rt -nan
         p  199, test: meansq err 0.223410, error rate 0.071251
         p  199, train: meansq err 0.155553, error rate 0.022219
...
DEBUG msq grad L1: 9.13947/9.13947, L2: 15.2056/15.2056, L3: 41.1683/41.1683,
DEBUG gradient changed sign in 33253 of 83448, zero 432; msq chg 2.68003/2.51968 unchg 4.51333/4.37777
DEBUG bumped rate adj   up to 19.5424
DEBUG effective rate = 2.68034e-07 (adjusted by autoRate2 from 1.37155e-08)
      pass  329 [ 330], meansq err 0.101926/0.101926, gradient last 1073.538258 all 3249.549441; est ert -nan below rt -nan
         p  329, test: meansq err 0.207773, error rate 0.063279
         p  329, train: meansq err 0.101917, error rate 0.006309
...
DEBUG msq grad L1: 8.34725/8.34725, L2: 13.8588/13.8588, L3: 38.4925/38.4925,
DEBUG gradient changed sign in 19895 of 83402, zero 478; msq chg 1.01996/0.965127 unchg 2.80679/2.80904
DEBUG bumped rate adj   up to 44.5141
DEBUG effective rate = 6.10536e-07 (adjusted by autoRate2 from 1.37155e-08)
      pass  379 [ 380], meansq err 0.088249/0.088249, gradient last 1003.762860 all 2973.413303; est ert -nan below rt -nan
         p  379, test: meansq err 0.206702, error rate 0.061784
         p  379, train: meansq err 0.088240, error rate 0.005349
...
DEBUG msq grad L1: 1.41545/1.41545, L2: 2.68539/2.68539, L3: 3.4139/3.4139,
DEBUG gradient changed sign in 4624 of 83594, zero 286; msq chg 0.354035/0.217445 unchg 0.171377/0.173191
DEBUG bumped rate adj   up to 1.75402
DEBUG effective rate = 2.40573e-08 (adjusted by autoRate2 from 1.37155e-08)
      pass 3439 [3440], meansq err 0.040231/0.040231, gradient last 89.023581 all 511.149187; est ert -nan below rt -nan
         p 3439, test: meansq err 0.208344, error rate 0.062282
         p 3439, train: meansq err 0.040231, error rate 0.003292
...
DEBUG msq grad L1: 0.97139/0.97139, L2: 1.97562/1.97562, L3: 3.317/3.317,
DEBUG gradient changed sign in 5922 of 83678, zero 202; msq chg 0.17374/0.380011 unchg 0.193911/0.194012
DEBUG bumped rate adj   up to 0.708437
DEBUG effective rate = 9.7166e-09 (adjusted by autoRate2 from 1.37155e-08)
      pass 8939 [8940], meansq err 0.038908/0.038908, gradient last 86.496651 all 367.742532; est ert -nan below rt -nan
         p 8939, test: meansq err 0.208010, error rate 0.061784
         p 8939, train: meansq err 0.038908, error rate 0.003017

	// The first test in trapeze mode, with reduced layer count
	options.weightSaturation_ = 1.;
	options.momentum_ = true;
	options.autoRate2_ = true;
	options.tweakRate_ = 0.3;
	TrainingFormat format = TF_16X16_TRAPEZE;
	FloatNeuralNet::ActivationFunction activation = FloatNeuralNet::CORNER;
	FloatNeuralNet::LevelSizeVector levels = {0, 256, 10};
	string checkpoint = fpath + "/corner_0_256_10_nofl_tr.ckp";
	int batchSize = 1;
	seed = 1667553859;
	double rate = 0.0001;
	double trainRate = 0.;
	double applyRate = rate;
...
DEBUG msq grad L1: 12.3132/12.3132, L2: 74.3951/74.3951,
DEBUG    L1 gradient changed sign in 18599 of 68244, zero 8812; msq chg 7.44232/6.6065 unchg 13.4164/12.671
DEBUG    L2 gradient changed sign in 625 of 2582, zero 18; msq chg 31.2801/27.4964 unchg 40.4198/38.912
DEBUG gradient changed sign in 19224 of 70826, zero 8830; msq chg 9.2411/8.17357 unchg 15.334/14.5564
DEBUG bumped rate adj   up to 106.958
DEBUG effective rate = 1.46699e-06 (adjusted by autoRate2 from 1.37155e-08)
      pass   99 [ 100], meansq err 0.272483/0.272483, gradient last 3793.421726 all 5106.166363; est ert -nan below rt -nan
         p   99, test: meansq err 0.314302, error rate 0.130045
         p   99, train: meansq err 0.272269, error rate 0.076670
...
DEBUG msq grad L1: 11.5783/11.5783, L2: 39.7796/39.7796, 
DEBUG    L1 gradient changed sign in 24914 of 68088, zero 8968; msq chg 11.0241/5.95958 unchg 11.1767/6.93307
DEBUG    L2 gradient changed sign in 760 of 2587, zero 13; msq chg 26.3732/36.8298 unchg 47.0013/32.1318
DEBUG gradient changed sign in 25674 of 70675, zero 8981; msq chg 11.7696/8.63819 unchg 14.4753/9.38257
DEBUG bumped rate adj   up to 110.991
DEBUG effective rate = 1.5223e-06 (adjusted by autoRate2 from 1.37155e-08)
      pass  199 [ 200], meansq err 0.160241/0.160241, gradient last 2028.370212 all 3800.558623; est ert -nan below rt -nan
         p  199, test: meansq err 0.250538, error rate 0.079721
         p  199, train: meansq err 0.160087, error rate 0.018516
...
DEBUG msq grad L1: 12.1554/12.1554, L2: 28.6318/28.6318,
DEBUG    L1 gradient changed sign in 13007 of 67979, zero 9077; msq chg 1.16865/1.1261 unchg 2.92616/2.82821
DEBUG    L2 gradient changed sign in 336 of 2580, zero 20; msq chg 6.39457/4.2214 unchg 5.47419/5.64963
DEBUG gradient changed sign in 13343 of 70559, zero 9097; msq chg 1.53657/1.29804 unchg 3.06625/2.98947
DEBUG bumped rate adj   up to 107.922
DEBUG effective rate = 1.48021e-06 (adjusted by autoRate2 from 1.37155e-08)
      pass  299 [ 300], meansq err 0.120821/0.120821, gradient last 1459.940790 all 3676.516427; est ert -nan below rt -nan
         p  299, test: meansq err 0.239904, error rate 0.076731
         p  299, train: meansq err 0.120796, error rate 0.011247
...
DEBUG msq grad L1: 10.7401/10.7401, L2: 18.2169/18.2169,
DEBUG    L1 gradient changed sign in 15139 of 67972, zero 9084; msq chg 1.06228/0.942348 unchg 1.6197/1.46003
DEBUG    L2 gradient changed sign in 548 of 2580, zero 20; msq chg 3.19992/2.0879 unchg 3.40418/2.19041
DEBUG gradient changed sign in 15687 of 70552, zero 9104; msq chg 1.2028/1.00463 unchg 1.71914/1.49346
DEBUG bumped rate adj   up to 112.494
DEBUG effective rate = 1.54291e-06 (adjusted by autoRate2 from 1.37155e-08)
      pass  499 [ 500], meansq err 0.085985/0.085985, gradient last 928.885102 all 3122.703285; est ert -nan below rt -nan
         p  499, test: meansq err 0.236828, error rate 0.074240
         p  499, train: meansq err 0.085973, error rate 0.007955
...
DEBUG msq grad L1: 8.03178/8.03178, L2: 8.00466/8.00466,
DEBUG    L1 gradient changed sign in 27452 of 67912, zero 9144; msq chg 0.598737/0.7516 unchg 0.629901/0.702174
DEBUG    L2 gradient changed sign in 834 of 2572, zero 28; msq chg 1.36114/2.17984 unchg 1.43966/1.64373
DEBUG gradient changed sign in 28286 of 70484, zero 9172; msq chg 0.634463/0.829668 unchg 0.682494/0.764213
DEBUG bumped rate adj   up to 154.165
DEBUG effective rate = 2.11446e-06 (adjusted by autoRate2 from 1.37155e-08)
      pass  999 [1000], meansq err 0.065578/0.065578, gradient last 408.159056 all 2266.590713; est ert -nan below rt -nan
         p  999, test: meansq err 0.241825, error rate 0.076233
         p  999, train: meansq err 0.065575, error rate 0.007132


	// Trapeze with 3 layers.
	options.weightSaturation_ = 1.;
	options.momentum_ = true;
	options.autoRate2_ = true;
	options.tweakRate_ = 0.3;
	TrainingFormat format = TF_16X16_TRAPEZE;
	FloatNeuralNet::ActivationFunction activation = FloatNeuralNet::CORNER;
	FloatNeuralNet::LevelSizeVector levels = {0, 256, 64, 10};
	string checkpoint = fpath + "/corner_0_256_64_10_nofl_tr.ckp";
	int batchSize = 1;
	seed = 1667553859;
	double rate = 0.0001;
	double trainRate = 0.;
	double applyRate = rate;
	../../../../zipnn/corner_0_256_64_10_nofl_tr.ckp
...
DEBUG msq grad L1: 17.3883/17.3883, L2: 50.9776/50.9776, L3: 125.4/125.4,
DEBUG    L1 gradient changed sign in 27668 of 68183, zero 8873; msq chg 12.231/15.6992 unchg 17.9393/18.8233
DEBUG    L2 gradient changed sign in 7161 of 16568, zero 72; msq chg 29.1616/34.0246 unchg 23.1686/25.5859
DEBUG    L3 gradient changed sign in 230 of 666, zero 14; msq chg 121.461/156.715 unchg 83.7213/100.287
DEBUG gradient changed sign in 35059 of 85417, zero 8959; msq chg 19.7115/24.3328 unchg 20.4908/22.2358
DEBUG bumped rate adj   up to 192.525
DEBUG effective rate = 2.64059e-06 (adjusted by autoRate2 from 1.37155e-08)
      pass   99 [ 100], meansq err 0.297789/0.297789, gradient last 3270.027873 all 8788.280085; est ert -nan below rt -nan
         p   99, test: meansq err 0.332364, error rate 0.154958
         p   99, train: meansq err 0.297406, error rate 0.111370
...
DEBUG msq grad L1: 13.054/13.054, L2: 33.2863/33.2863, L3: 81.3118/81.3118,
DEBUG    L1 gradient changed sign in 43060 of 68072, zero 8984; msq chg 9.14063/12.7792 unchg 4.8955/5.91418
DEBUG    L2 gradient changed sign in 10927 of 16554, zero 86; msq chg 18.8037/26.2718 unchg 11.127/12.1881
DEBUG    L3 gradient changed sign in 324 of 665, zero 15; msq chg 68.7336/85.9196 unchg 38.7975/41.7574
DEBUG gradient changed sign in 54311 of 85291, zero 9085; msq chg 12.8672/17.6743 unchg 7.64237/8.62634
DEBUG bumped rate adj Down to 87.9047
DEBUG effective rate = 1.20566e-06 (adjusted by autoRate2 from 1.37155e-08)
      pass  199 [ 200], meansq err 0.167557/0.167557, gradient last 2120.352998 all 6005.296286; est ert -nan below rt -nan
         p  199, test: meansq err 0.248606, error rate 0.082212
         p  199, train: meansq err 0.167097, error rate 0.025237
...
DEBUG msq grad L1: 8.41103/8.41103, L2: 20.6367/20.6367, L3: 42.75/42.75,
DEBUG    L1 gradient changed sign in 6203 of 68008, zero 9048; msq chg 0.620187/0.715682 unchg 2.45569/2.46643
DEBUG    L2 gradient changed sign in 1122 of 16533, zero 107; msq chg 1.83756/1.87986 unchg 4.33945/4.27089
DEBUG    L3 gradient changed sign in 43 of 659, zero 21; msq chg 2.69209/1.10243 unchg 7.12445/6.91605
DEBUG gradient changed sign in 7368 of 85200, zero 9176; msq chg 0.938245/0.988151 unchg 2.98646/2.97
DEBUG bumped rate adj   up to 88.2997
DEBUG effective rate = 1.21108e-06 (adjusted by autoRate2 from 1.37155e-08)
      pass  299 [ 300], meansq err 0.125952/0.125952, gradient last 1114.783065 all 3712.228872; est ert -nan below rt -nan
         p  299, test: meansq err 0.235640, error rate 0.077230
         p  299, train: meansq err 0.125929, error rate 0.016184
...
DEBUG msq grad L1: 5.80019/5.80019, L2: 13.3887/13.3887, L3: 23.0166/23.0166,
DEBUG    L1 gradient changed sign in 18402 of 67966, zero 9090; msq chg 0.623625/0.711113 unchg 1.15313/1.20203
DEBUG    L2 gradient changed sign in 3971 of 16522, zero 118; msq chg 1.34614/1.87056 unchg 1.53192/1.66513
DEBUG    L3 gradient changed sign in 136 of 656, zero 24; msq chg 2.13271/3.91248 unchg 2.84464/3.20359
DEBUG gradient changed sign in 22509 of 85144, zero 9232; msq chg 0.815546/1.05981 unchg 1.26082/1.33572
DEBUG bumped rate adj   up to 95.0829
DEBUG effective rate = 1.30411e-06 (adjusted by autoRate2 from 1.37155e-08)
      pass  499 [ 500], meansq err 0.092798/0.092798, gradient last 600.199892 all 2436.269323; est ert -nan below rt -nan
         p  499, test: meansq err 0.232167, error rate 0.074240
         p  499, train: meansq err 0.092790, error rate 0.012344
...
DEBUG msq grad L1: 4.12614/4.12614, L2: 9.4192/9.4192, L3: 13.6235/13.6235,
DEBUG    L1 gradient changed sign in 36201 of 67982, zero 9074; msq chg 0.965393/1.36542 unchg 0.653166/0.881196
DEBUG    L2 gradient changed sign in 8888 of 16539, zero 101; msq chg 1.77041/2.75353 unchg 1.16941/1.7779
DEBUG    L3 gradient changed sign in 254 of 656, zero 24; msq chg 3.88987/7.6272 unchg 6.46154/7.0998
DEBUG gradient changed sign in 45343 of 85177, zero 9199; msq chg 1.20134/1.81674 unchg 1.01212/1.31733
DEBUG bumped rate adj Down to 48.1861
DEBUG effective rate = 6.60898e-07 (adjusted by autoRate2 from 1.37155e-08)
      pass  749 [ 750], meansq err 0.080764/0.080764, gradient last 355.258018 all 1707.165723; est ert -nan below rt -nan
         p  749, test: meansq err 0.234655, error rate 0.075237
         p  749, train: meansq err 0.080758, error rate 0.010972
...
DEBUG msq grad L1: 3.15082/3.15082, L2: 7.71741/7.71741, L3: 11.9179/11.9179,
DEBUG    L1 gradient changed sign in 36735 of 68008, zero 9048; msq chg 1.03064/1.33023 unchg 0.593242/0.722321
DEBUG    L2 gradient changed sign in 10827 of 16541, zero 99; msq chg 1.78531/2.74859 unchg 1.32782/1.63459
DEBUG    L3 gradient changed sign in 334 of 651, zero 29; msq chg 4.25464/6.87137 unchg 2.996/4.21688
DEBUG gradient changed sign in 47896 of 85200, zero 9176; msq chg 1.28896/1.84234 unchg 0.800861/0.998884
DEBUG bumped rate adj Down to 53.7234
DEBUG effective rate = 7.36845e-07 (adjusted by autoRate2 from 1.37155e-08)
      pass  999 [1000], meansq err 0.076661/0.076661, gradient last 310.781712 all 1361.110603; est ert -nan below rt -nan
         p  999, test: meansq err 0.235592, error rate 0.077728
         p  999, train: meansq err 0.076654, error rate 0.010698
...
DEBUG msq grad L1: 0.858271/0.858271, L2: 1.70182/1.70182, L3: 17.673/17.673,
DEBUG    L1 gradient changed sign in 3454 of 68125, zero 8931; msq chg 0.0536892/0.0677342 unchg 0.732382/0.716456
DEBUG    L2 gradient changed sign in 798 of 16573, zero 67; msq chg 0.12359/0.129478 unchg 0.911845/0.875776
DEBUG    L3 gradient changed sign in 32 of 666, zero 14; msq chg 0.290155/0.59142 unchg 1.25097/1.24891
DEBUG gradient changed sign in 4284 of 85364, zero 9012; msq chg 0.0761458/0.0971317 unchg 0.775782/0.755544
DEBUG bumped rate adj   up to 12.1019
DEBUG effective rate = 1.65984e-07 (adjusted by autoRate2 from 1.37155e-08)
      pass 6639 [6640], meansq err 0.061117/0.061117, gradient last 460.854274 all 563.330190; est ert -nan below rt -nan
         p 6639, test: meansq err 0.236829, error rate 0.078226
         p 6639, train: meansq err 0.061116, error rate 0.007681
...
DEBUG msq grad L1: 0.568404/0.568404, L2: 1.50559/1.50559, L3: 17.447/17.447,
DEBUG    L1 gradient changed sign in 13446 of 68116, zero 8940; msq chg 0.0804567/0.0761485 unchg 0.0803721/0.0776092
DEBUG    L2 gradient changed sign in 2391 of 16561, zero 79; msq chg 0.06823/0.0841062 unchg 0.131635/0.117102
DEBUG    L3 gradient changed sign in 101 of 654, zero 26; msq chg 0.0942945/0.107565 unchg 0.282542/0.252397
DEBUG gradient changed sign in 15938 of 85331, zero 9045; msq chg 0.0788409/0.0776304 unchg 0.0962478/0.089739
DEBUG bumped rate adj   up to 60.1958
DEBUG effective rate = 8.25618e-07 (adjusted by autoRate2 from 1.37155e-08)
      pass 7319 [7320], meansq err 0.060686/0.060686, gradient last 454.962535 all 519.235868; est ert -nan below rt -nan
         p 7319, test: meansq err 0.236743, error rate 0.077728
         p 7319, train: meansq err 0.060686, error rate 0.007818
...
DEBUG msq grad L1: 0.588518/0.588518, L2: 1.5286/1.5286, L3: 17.4346/17.4346,
DEBUG    L1 gradient changed sign in 23901 of 68121, zero 8935; msq chg 0.0813305/0.0967906 unchg 0.0795924/0.0775194
DEBUG    L2 gradient changed sign in 3967 of 16559, zero 81; msq chg 0.136474/0.192557 unchg 0.125903/0.098774
DEBUG    L3 gradient changed sign in 175 of 651, zero 29; msq chg 0.212561/0.121786 unchg 0.299873/0.199085
DEBUG gradient changed sign in 28043 of 85331, zero 9045; msq chg 0.0924898/0.115422 unchg 0.0955051/0.0843342
DEBUG bumped rate adj   up to 73.8101
DEBUG effective rate = 1.01234e-06 (adjusted by autoRate2 from 1.37155e-08)
      pass 7479 [7480], meansq err 0.060224/0.060224, gradient last 454.639159 all 521.791928; est ert -nan below rt -nan
         p 7479, test: meansq err 0.236794, error rate 0.076731
         p 7479, train: meansq err 0.060224, error rate 0.007681
...
DEBUG msq grad L1: 0.666918/0.666918, L2: 1.58705/1.58705, L3: 17.0549/17.0549,
DEBUG    L1 gradient changed sign in 13203 of 68069, zero 8987; msq chg 0.109559/0.0971841 unchg 0.227878/0.220884
DEBUG    L2 gradient changed sign in 2768 of 16548, zero 92; msq chg 0.233331/0.204598 unchg 0.340455/0.330831
DEBUG    L3 gradient changed sign in 150 of 647, zero 33; msq chg 0.295489/0.507686 unchg 0.519343/0.520133
DEBUG gradient changed sign in 16121 of 85264, zero 9112; msq chg 0.141389/0.131609 unchg 0.257381/0.249946
DEBUG bumped rate adj   up to 103.773
DEBUG effective rate = 1.4233e-06 (adjusted by autoRate2 from 1.37155e-08)
      pass 7789 [7790], meansq err 0.059946/0.059946, gradient last 444.738043 all 523.427669; est ert -nan below rt -nan
         p 7789, test: meansq err 0.236739, error rate 0.076731
         p 7789, train: meansq err 0.059946, error rate 0.007544
...
DEBUG msq grad L1: 0.493107/0.493107, L2: 1.28614/1.28614, L3: 17.58/17.58,
DEBUG    L1 gradient changed sign in 31206 of 68109, zero 8947; msq chg 0.0858834/0.0855205 unchg 0.0693783/0.0709802
DEBUG    L2 gradient changed sign in 7639 of 16566, zero 74; msq chg 0.126363/0.13535 unchg 0.114417/0.105382
DEBUG    L3 gradient changed sign in 298 of 657, zero 23; msq chg 0.240042/0.305985 unchg 0.25165/0.23561
DEBUG gradient changed sign in 39143 of 85332, zero 9044; msq chg 0.0971348/0.100592 unchg 0.0828736/0.0812595
DEBUG bumped rate adj   up to 85.9888
DEBUG effective rate = 1.17938e-06 (adjusted by autoRate2 from 1.37155e-08)
      pass 8019 [8020], meansq err 0.059720/0.059720, gradient last 458.430226 all 506.379393; est ert -nan below rt -nan
         p 8019, test: meansq err 0.236566, error rate 0.076731
         p 8019, train: meansq err 0.059720, error rate 0.007406

	// same as before but stochastic
	options.weightSaturation_ = 1.;
	TrainingFormat format = TF_16X16_TRAPEZE;
	FloatNeuralNet::ActivationFunction activation = FloatNeuralNet::CORNER;
	FloatNeuralNet::LevelSizeVector levels = {0, 256, 64, 10};
	string checkpoint = fpath + "/corner_0_256_64_10_nofl_st.ckp";
	int batchSize = 1;
	seed = 1667553859;
	double rate = 0.1;
	trainRate = rate;
	applyRate = 0.;
	../../../../zipnn/corner_0_256_64_10_nofl_st.ckp
...
DEBUG msq grad L1: 17.0449/17.0449, L2: 54.6114/54.6114, L3: 109.372/109.372, 
      pass    9 [  49], meansq err 0.312728/0.312728, gradient last 2852.074411 all 8952.567341; est ert -nan below rt -nan
         p    9, test: meansq err 0.345017, error rate 0.159442
         p    9, train: meansq err 0.312360, error rate 0.111507
...
DEBUG msq grad L1: 12.6866/12.6866, L2: 38.6705/38.6705, L3: 59.5748/59.5748,
      pass   99 [ 139], meansq err 0.202207/0.202207, gradient last 1553.519434 all 6300.730187; est ert -nan below rt -nan
         p   99, test: meansq err 0.265049, error rate 0.094669
         p   99, train: meansq err 0.202451, error rate 0.040461
...
DEBUG msq grad L1: 11.3942/11.3942, L2: 28.5185/28.5185, L3: 60.1852/60.1852, 
      pass  269 [ 309], meansq err 0.149162/0.149162, gradient last 1569.439239 all 5099.064728; est ert -nan below rt -nan
         p  269, test: meansq err 0.242651, error rate 0.078724
         p  269, train: meansq err 0.148980, error rate 0.020162
...
DEBUG msq grad L1: 7.23081/7.23081, L2: 17.413/17.413, L3: 25.9131/25.9131, 
      pass  919 [ 959], meansq err 0.091875/0.091875, gradient last 675.730395 all 3087.219819; est ert -nan below rt -nan
         p  919, test: meansq err 0.232310, error rate 0.076731
         p  919, train: meansq err 0.092306, error rate 0.012481
...

	// same as above but stockastic and Leaky RELU
	options.weightSaturation_ = 1.;
	TrainingFormat format = TF_16X16_TRAPEZE;
	FloatNeuralNet::ActivationFunction activation = FloatNeuralNet::LEAKY_RELU;
	FloatNeuralNet::LevelSizeVector levels = {0, 256, 64, 10};
	string checkpoint = fpath + "/leaky_0_256_64_10_nofl_st.ckp";
	int batchSize = 1;
	seed = 1667553859;
	double rate = 0.1;
	trainRate = rate;
	applyRate = 0.;
	../../../../zipnn/leaky_0_256_64_10_nofl_st.ckp
...
DEBUG msq grad L1: 3.58929/3.58929, L2: 5.99799/5.99799, L3: 20.7599/20.7599, 
      pass   99 [ 100], meansq err 0.251449/0.251449, gradient last 529.276687 all 1361.866968; est ert -nan below rt -nan
         p   99, test: meansq err 0.301564, error rate 0.102641
         p   99, train: meansq err 0.251192, error rate 0.051708
...
DEBUG msq grad L1: 2.73612/2.73612, L2: 4.50739/4.50739, L3: 14.7121/14.7121,
      pass  149 [ 150], meansq err 0.224091/0.224091, gradient last 375.085497 all 1022.729804; est ert -nan below rt -nan
         p  149, test: meansq err 0.283650, error rate 0.088690
         p  149, train: meansq err 0.223837, error rate 0.035660
...
DEBUG msq grad L1: 2.2359/2.2359, L2: 3.688/3.688, L3: 11.5961/11.5961,
      pass  199 [ 200], meansq err 0.205463/0.205463, gradient last 295.644966 all 832.168740; est ert -nan below rt -nan
         p  199, test: meansq err 0.272565, error rate 0.081714
         p  199, train: meansq err 0.205199, error rate 0.027020
...
DEBUG msq grad L1: 1.08269/1.08269, L2: 1.84811/1.84811, L3: 5.5593/5.5593,
      pass  559 [ 560], meansq err 0.143529/0.143529, gradient last 141.734924 all 407.054512; est ert -nan below rt -nan
         p  559, test: meansq err 0.245666, error rate 0.069756
         p  559, train: meansq err 0.143279, error rate 0.007269
...

	// Trapeze as before but now with absolute X coordinates
	options.weightSaturation_ = 1.;
	options.momentum_ = true;
	options.autoRate2_ = true;
	options.tweakRate_ = 0.3;
	TrainingFormat format = TF_16X16_TRAPEZE_ABS;
	FloatNeuralNet::ActivationFunction activation = FloatNeuralNet::CORNER;
	FloatNeuralNet::LevelSizeVector levels = {0, 256, 64, 10};
	string checkpoint = fpath + "/corner_0_256_64_10_nofl_tra.ckp";
	int batchSize = 1;
	seed = 1667553859;
	double rate = 0.0001;
	double trainRate = 0.;
	double applyRate = rate;
	../../../../zipnn/corner_0_256_64_10_nofl_tra.x.ckp
...
DEBUG msq grad L1: 26.2978/26.2978, L2: 75.1164/75.1164, L3: 382.427/382.427,
DEBUG    L1 gradient changed sign in 7467 of 67743, zero 9313; msq chg 10.9501/12.6596 unchg 26.2325/29.3828
DEBUG    L2 gradient changed sign in 2448 of 16603, zero 37; msq chg 29.3134/34.6561 unchg 67.5643/73.2061
DEBUG    L3 gradient changed sign in 95 of 666, zero 14; msq chg 145.742/326.205 unchg 176.617/198.516
DEBUG gradient changed sign in 10010 of 85012, zero 9364; msq chg 22.3868/37.7247 unchg 40.6454/44.7803
DEBUG bumped rate adj   up to 52.6646
DEBUG effective rate = 7.22323e-07 (adjusted by autoRate2 from 1.37155e-08)
      pass   49 [  50], meansq err 0.457027/0.457027, gradient last 9972.479090 all 15704.493687; est ert -nan below rt -nan
         p   49, test: meansq err 0.472623, error rate 0.416044
         p   49, train: meansq err 0.456227, error rate 0.371417
...
DEBUG msq grad L1: 22.5836/22.5836, L2: 63.725/63.725, L3: 156.462/156.462,
DEBUG    L1 gradient changed sign in 14440 of 67747, zero 9309; msq chg 23.2707/9.79134 unchg 47.5414/23.8452
DEBUG    L2 gradient changed sign in 3510 of 16576, zero 64; msq chg 63.596/19.7884 unchg 94.343/42.4266
DEBUG    L3 gradient changed sign in 136 of 671, zero 9; msq chg 121.582/104.12 unchg 353.669/161.552
DEBUG gradient changed sign in 18086 of 84994, zero 9382; msq chg 36.4476/15.299 unchg 67.3723/31.831
DEBUG bumped rate adj   up to 23.113
DEBUG effective rate = 3.17007e-07 (adjusted by autoRate2 from 1.37155e-08)
      pass   99 [ 100], meansq err 0.298523/0.298523, gradient last 4080.022895 all 11113.942271; est ert -nan below rt -nan
         p   99, test: meansq err 0.340857, error rate 0.168411
         p   99, train: meansq err 0.298219, error rate 0.114799
...
DEBUG msq grad L1: 15.6998/15.6998, L2: 45.1407/45.1407, L3: 64.9335/64.9335,
DEBUG    L1 gradient changed sign in 7524 of 67507, zero 9549; msq chg 2.49704/2.70052 unchg 7.55774/7.78119
DEBUG    L2 gradient changed sign in 2170 of 16536, zero 104; msq chg 6.03665/6.57742 unchg 9.90666/10.5207
DEBUG    L3 gradient changed sign in 67 of 670, zero 10; msq chg 26.8463/26.3044 unchg 66.339/66.2929
DEBUG gradient changed sign in 9761 of 84713, zero 9663; msq chg 4.22548/4.47087 unchg 9.99641/10.2482
DEBUG bumped rate adj   up to 121.724
DEBUG effective rate = 1.6695e-06 (adjusted by autoRate2 from 1.37155e-08)
      pass  149 [ 150], meansq err 0.238138/0.238138, gradient last 1693.259778 all 7467.734297; est ert -nan below rt -nan
         p  149, test: meansq err 0.297965, error rate 0.127055
         p  149, train: meansq err 0.238007, error rate 0.068166
...
DEBUG msq grad L1: 15.257/15.257, L2: 41.2741/41.2741, L3: 66.1013/66.1013,
DEBUG    L1 gradient changed sign in 3861 of 67488, zero 9568; msq chg 1.17021/1.05106 unchg 5.70224/5.63405
DEBUG    L2 gradient changed sign in 939 of 16531, zero 109; msq chg 2.12365/1.85276 unchg 10.7943/10.5035
DEBUG    L3 gradient changed sign in 24 of 668, zero 12; msq chg 4.51914/3.75534 unchg 58.7112/58.3864
DEBUG gradient changed sign in 4824 of 84687, zero 9689; msq chg 1.44065/1.27379 unchg 8.74354/8.62076
DEBUG bumped rate adj   up to 53.4209
DEBUG effective rate = 7.32697e-07 (adjusted by autoRate2 from 1.37155e-08)
      pass  199 [ 200], meansq err 0.205672/0.205672, gradient last 1723.711382 all 7018.200418; est ert -nan below rt -nan
         p  199, test: meansq err 0.279142, error rate 0.110115
         p  199, train: meansq err 0.205617, error rate 0.045261

	options.weightSaturation_ = 1.;
	options.momentum_ = true;
	options.autoRate2_ = true;
	options.tweakRate_ = 0.3;
	TrainingFormat format = TF_8X8_TRAPEZE_BITMAP;
	FloatNeuralNet::ActivationFunction activation = FloatNeuralNet::CORNER;
	FloatNeuralNet::LevelSizeVector levels = {0, 256, 64, 10};
	string checkpoint = fpath + "/corner_0_256_64_10_nofl_trb.ckp";
	int batchSize = 1;
	seed = 1667553859;
	double rate = 0.0001;
	double trainRate = 0.;
	double applyRate = rate;
	../../../../zipnn/corner_0_256_64_10_nofl_trb..x.ckp
...
DEBUG msq grad L1: 20.8131/20.8131, L2: 31.5844/31.5844, L3: 152.673/152.673, 
DEBUG    L1 gradient changed sign in 5030 of 17251, zero 157; msq chg 15.0148/13.7714 unchg 22.1239/23.0769
DEBUG    L2 gradient changed sign in 5162 of 16599, zero 41; msq chg 22.8306/22.434 unchg 36.7883/34.7741
DEBUG    L3 gradient changed sign in 233 of 671, zero 9; msq chg 68.5804/125.186 unchg 168.994/142.702
DEBUG gradient changed sign in 10425 of 34521, zero 207; msq chg 21.7253/26.2864 unchg 37.5465/34.8456
DEBUG bumped rate adj   up to 1319.96
DEBUG effective rate = 1.8104e-05 (adjusted by autoRate2 from 1.37155e-08)
      pass   49 [  50], meansq err 0.363591/0.363591, gradient last 3981.234559 all 6323.827488; est ert -nan below rt -nan
         p   49, test: meansq err 0.388401, error rate 0.241654
         p   49, train: meansq err 0.361041, error rate 0.195446
...
DEBUG msq grad L1: 6.05477/6.05477, L2: 33.1835/33.1835, L3: 78.4462/78.4462,
DEBUG    L1 gradient changed sign in 1711 of 17242, zero 166; msq chg 1.50502/1.70059 unchg 6.12324/6.15695
DEBUG    L2 gradient changed sign in 2071 of 16552, zero 88; msq chg 1.93687/1.9186 unchg 5.92903/5.88527
DEBUG    L3 gradient changed sign in 101 of 663, zero 17; msq chg 12.6577/14.825 unchg 17.9711/16.2637
DEBUG gradient changed sign in 3883 of 34457, zero 271; msq chg 2.677/2.99237 unchg 6.45235/6.36583
DEBUG bumped rate adj   up to 579.294
DEBUG effective rate = 7.94533e-06 (adjusted by autoRate2 from 1.37155e-08)
      pass   99 [ 100], meansq err 0.244370/0.244370, gradient last 2045.627462 all 4811.014222; est ert -nan below rt -nan
         p   99, test: meansq err 0.314280, error rate 0.141505
         p   99, train: meansq err 0.244180, error rate 0.074201
...
DEBUG msq grad L1: 3.53626/3.53626, L2: 23.1616/23.1616, L3: 58.7722/58.7722,
DEBUG    L1 gradient changed sign in 2704 of 17249, zero 159; msq chg 1.65756/1.33885 unchg 4.09833/3.42318
DEBUG    L2 gradient changed sign in 3066 of 16540, zero 100; msq chg 2.40772/2.04716 unchg 5.24278/4.56268
DEBUG    L3 gradient changed sign in 146 of 652, zero 28; msq chg 8.6518/7.08691 unchg 17.9353/11.1569
DEBUG gradient changed sign in 5916 of 34441, zero 287; msq chg 2.47133/2.05687 unchg 5.22056/4.24461
DEBUG bumped rate adj   up to 111.577
DEBUG effective rate = 1.53033e-06 (adjusted by autoRate2 from 1.37155e-08)
      pass  199 [ 200], meansq err 0.151738/0.151738, gradient last 1532.592155 all 3390.168359; est ert -nan below rt -nan
         p  199, test: meansq err 0.290403, error rate 0.119581
         p  199, train: meansq err 0.151712, error rate 0.025099
...
DEBUG msq grad L1: 3.51694/3.51694, L2: 17.4283/17.4283, L3: 40.2184/40.2184,
DEBUG    L1 gradient changed sign in 8609 of 17224, zero 184; msq chg 2.99588/3.91059 unchg 2.18888/2.62817
DEBUG    L2 gradient changed sign in 10140 of 16527, zero 113; msq chg 3.37888/5.44412 unchg 3.40865/4.06514
DEBUG    L3 gradient changed sign in 393 of 645, zero 35; msq chg 13.9769/22.0987 unchg 4.21518/5.19479
DEBUG gradient changed sign in 19142 of 34396, zero 332; msq chg 3.75435/5.71002 unchg 2.80436/3.35651
DEBUG bumped rate adj Down to 300.736
DEBUG effective rate = 4.12475e-06 (adjusted by autoRate2 from 1.37155e-08)
      pass  299 [ 300], meansq err 0.120995/0.120995, gradient last 1048.767313 all 2523.799609; est ert -nan below rt -nan
         p  299, test: meansq err 0.292182, error rate 0.123568
         p  299, train: meansq err 0.120924, error rate 0.021259
...
DEBUG msq grad L1: 1.04238/1.04238, L2: 8.85389/8.85389, L3: 15.1292/15.1292,
DEBUG    L1 gradient changed sign in 2258 of 17188, zero 220; msq chg 0.346894/0.363706 unchg 0.879947/0.862199
DEBUG    L2 gradient changed sign in 2847 of 16521, zero 119; msq chg 0.518436/0.538123 unchg 1.00579/0.984638
DEBUG    L3 gradient changed sign in 91 of 642, zero 38; msq chg 1.61251/2.04544 unchg 4.85328/4.91846
DEBUG gradient changed sign in 5196 of 34351, zero 377; msq chg 0.495076/0.537981 unchg 1.14722/1.13692
DEBUG bumped rate adj   up to 360.85
DEBUG effective rate = 4.94925e-06 (adjusted by autoRate2 from 1.37155e-08)
      pass  599 [ 600], meansq err 0.099223/0.099223, gradient last 394.520518 all 1216.138811; est ert -nan below rt -nan
         p  599, test: meansq err 0.294111, error rate 0.125062
         p  599, train: meansq err 0.099216, error rate 0.019476
...
DEBUG msq grad L1: 0.700082/0.700082, L2: 5.09347/5.09347, L3: 16.5667/16.5667,
DEBUG    L1 gradient changed sign in 2154 of 17278, zero 130; msq chg 0.210461/0.18191 unchg 0.765247/0.615703
DEBUG    L2 gradient changed sign in 1759 of 16538, zero 102; msq chg 0.419534/0.347418 unchg 1.14477/0.880807
DEBUG    L3 gradient changed sign in 56 of 634, zero 46; msq chg 0.726782/0.505374 unchg 4.71032/3.53084
DEBUG gradient changed sign in 3969 of 34450, zero 278; msq chg 0.330901/0.273961 unchg 1.16047/0.894798
DEBUG bumped rate adj   up to 97.2521
DEBUG effective rate = 1.33386e-06 (adjusted by autoRate2 from 1.37155e-08)
      pass  999 [1000], meansq err 0.093273/0.093273, gradient last 432.005693 all 791.745024; est ert -nan below rt -nan
         p  999, test: meansq err 0.296809, error rate 0.127554
         p  999, train: meansq err 0.093271, error rate 0.018653
...
DEBUG msq grad L1: 0.255631/0.255631, L2: 1.06796/1.06796, L3: 20.703/20.703,
DEBUG    L1 gradient changed sign in 1897 of 17309, zero 99; msq chg 0.057473/0.0478612 unchg 0.189599/0.171083
DEBUG    L2 gradient changed sign in 3222 of 16572, zero 68; msq chg 0.108973/0.0881911 unchg 0.279331/0.229431
DEBUG    L3 gradient changed sign in 173 of 655, zero 25; msq chg 0.344352/0.143935 unchg 0.718529/0.471364
DEBUG gradient changed sign in 5292 of 34536, zero 192; msq chg 0.110863/0.0789543 unchg 0.251144/0.207647
DEBUG bumped rate adj   up to 84.4923
DEBUG effective rate = 1.15886e-06 (adjusted by autoRate2 from 1.37155e-08)
      pass 10379 [10380], meansq err 0.078496/0.078496, gradient last 539.868589 all 558.188405; est ert -nan below rt -nan
         p 10379, test: meansq err 0.306068, error rate 0.130045
         p 10379, train: meansq err 0.078496, error rate 0.011795
...
DEBUG msq grad L1: 0.353178/0.353178, L2: 0.870199/0.870199, L3: 12.2555/12.2555,
DEBUG    L1 gradient changed sign in 1974 of 17311, zero 97; msq chg 0.084592/0.0711754 unchg 0.309263/0.297703
DEBUG    L2 gradient changed sign in 2632 of 16568, zero 72; msq chg 0.134855/0.101402 unchg 0.273991/0.262143
DEBUG    L3 gradient changed sign in 158 of 650, zero 30; msq chg 0.542223/0.405128 unchg 0.251137/0.239618
DEBUG gradient changed sign in 4764 of 34529, zero 199; msq chg 0.150874/0.114993 unchg 0.292358/0.280696
DEBUG bumped rate adj   up to 283.413
DEBUG effective rate = 3.88717e-06 (adjusted by autoRate2 from 1.37155e-08)
      pass 19999 [20000], meansq err 0.061940/0.061940, gradient last 319.584454 all 341.915462; est ert -nan below rt -nan
         p 19999, test: meansq err 0.308820, error rate 0.127554
         p 19999, train: meansq err 0.061939, error rate 0.007955

	//  a continuation offspring of the above after 20K rounds, but with
	options.isClassifier_ = true;
...
DEBUG msq grad L1: 21.6588/14.0443, L2: 22.3439/14.4885, L3: 58.2733/37.7864,
DEBUG effective rate = 8.89363e-09 (adjusted by effective cases from 1.37155e-08)
DEBUG    L1 gradient changed sign in 2136 of 17163, zero 245; msq chg 0.399009/0.410597 unchg 1.49871/1.4786
DEBUG    L2 gradient changed sign in 2996 of 16536, zero 104; msq chg 0.647038/0.590553 unchg 1.29627/1.25273
DEBUG    L3 gradient changed sign in 164 of 631, zero 49; msq chg 2.43324/1.27623 unchg 2.46707/1.91945
DEBUG gradient changed sign in 5296 of 34330, zero 398; msq chg 0.695984/0.561896 unchg 1.4297/1.3866
DEBUG bumped rate adj   up to 112.169
DEBUG effective rate = 9.97589e-07 (adjusted by autoRate2 from 8.89363e-09)
      pass  509 [20613], meansq err 0.033323/0.092336, gradient last 985.348942 all 2810.257206; est ert 0.000274 below rt 0.162392
         p  509, test: meansq err 0.314152, error rate 0.132038,  below 0.206776
         p  509, train: meansq err 0.033318, error rate 0.000274,  below 0.000274
...
DEBUG msq grad L1: 22.1211/14.3441, L2: 20.7248/13.4387, L3: 54.8855/35.5897,
DEBUG effective rate = 8.89363e-09 (adjusted by effective cases from 1.37155e-08)
DEBUG    L1 gradient changed sign in 2785 of 17161, zero 247; msq chg 0.423784/0.353558 unchg 1.15811/1.06078
DEBUG    L2 gradient changed sign in 3256 of 16536, zero 104; msq chg 0.634394/0.434362 unchg 1.17974/1.0367
DEBUG    L3 gradient changed sign in 168 of 629, zero 51; msq chg 2.27087/1.60946 unchg 1.85514/1.20303
DEBUG gradient changed sign in 6209 of 34326, zero 402; msq chg 0.656608/0.474444 unchg 1.18302/1.05199
DEBUG bumped rate adj   up to 68.1841
DEBUG effective rate = 6.06404e-07 (adjusted by autoRate2 from 8.89363e-09)
      pass  689 [20793], meansq err 0.027937/0.091162, gradient last 928.064733 all 2729.140066; est ert 0.000274 below rt 0.162255
         p  689, test: meansq err 0.314773, error rate 0.131041,  below 0.204285
         p  689, train: meansq err 0.027934, error rate 0.000274,  below 0.000274
...
DEBUG msq grad L1: 21.7826/20.6551, L2: 16.6213/15.7609, L3: 40.8775/38.7616,
DEBUG effective rate = 1.30056e-08 (adjusted by effective cases from 1.37155e-08)
DEBUG    L1 gradient changed sign in 5292 of 17143, zero 265; msq chg 0.0662306/0.0832559 unchg 0.191052/0.197362
DEBUG    L2 gradient changed sign in 4947 of 16549, zero 91; msq chg 0.0836942/0.116822 unchg 0.0784143/0.0851968
DEBUG    L3 gradient changed sign in 187 of 640, zero 40; msq chg 0.317016/0.512805 unchg 0.116239/0.116484
DEBUG gradient changed sign in 10426 of 34332, zero 396; msq chg 0.0857478/0.121286 unchg 0.146064/0.151952
DEBUG bumped rate adj   up to 133.494
DEBUG effective rate = 1.73616e-06 (adjusted by autoRate2 from 1.30056e-08)
      pass 3339 [24876], meansq err 0.020559/0.103747, gradient last 1010.777925 all 3547.107894; est ert 0.000274 below 0.000274
         p 3339, test: meansq err 0.318217, error rate 0.130543,  below 0.205780
         p 3339, train: meansq err 0.020554, error rate 0.000274,  below 0.000274
...

	// a continuation offspring from above but added the test data into training data
	// (still with classifier mode)
...
DEBUG msq grad L1: 22.04/8.57262, L2: 57.1766/22.2392, L3: 796.428/309.776,
DEBUG effective rate = 4.18323e-09 (adjusted by effective cases from 1.0755e-08)
DEBUG    L1 gradient changed sign in 6663 of 17302, zero 106; msq chg 9.16309/10.9349 unchg 10.396/10.9824
DEBUG    L2 gradient changed sign in 8410 of 16571, zero 69; msq chg 12.1828/14.8882 unchg 12.9671/14.0457
DEBUG    L3 gradient changed sign in 398 of 650, zero 30; msq chg 30.0049/37.413 unchg 22.9255/29.7717
DEBUG gradient changed sign in 15471 of 34523, zero 205; msq chg 11.8323/14.4221 unchg 11.8038/12.79
DEBUG bumped rate adj   up to 237.897
DEBUG effective rate = 9.95175e-07 (adjusted by autoRate2 from 4.18323e-09)
      pass  279 [25158], meansq err 0.102937/0.192203, gradient last 8077.961377 all 8646.537100; est ert 0.001613 below 0.003119
         p  279, test: meansq err 0.128709, error rate 0.006477,  below 0.011460
         p  279, train: meansq err 0.102869, error rate 0.001613,  below 0.003011
...
DEBUG msq grad L1: 31.8697/29.8907, L2: 38.7344/36.3291, L3: 69.9778/65.6325,
DEBUG effective rate = 1.00872e-08 (adjusted by effective cases from 1.0755e-08)
DEBUG    L1 gradient changed sign in 3510 of 17178, zero 230; msq chg 1.40624/1.41786 unchg 3.13758/3.05801
DEBUG    L2 gradient changed sign in 3621 of 16552, zero 88; msq chg 1.5729/2.18447 unchg 3.21239/2.84912
DEBUG    L3 gradient changed sign in 95 of 624, zero 56; msq chg 7.61335/4.37186 unchg 8.26513/7.45257
DEBUG gradient changed sign in 7226 of 34354, zero 374; msq chg 1.72115/1.90237 unchg 3.34828/3.10869
DEBUG bumped rate adj   up to 88.7755
DEBUG effective rate = 8.95492e-07 (adjusted by autoRate2 from 1.00872e-08)
      pass  609 [25768], meansq err 0.051120/0.124812, gradient last 1711.485329 all 6359.560252; est ert 0.000538 below 0.000968
         p  609, test: meansq err 0.069885, error rate 0.002491,  below 0.004484
         p  609, train: meansq err 0.051158, error rate 0.002043,  below 0.002581
...
DEBUG msq grad L1: 31.585/29.6145, L2: 37.9881/35.6181, L3: 67.5962/63.3789,
DEBUG effective rate = 1.0084e-08 (adjusted by effective cases from 1.0755e-08)
DEBUG    L1 gradient changed sign in 11712 of 17182, zero 226; msq chg 3.79101/6.76143 unchg 3.23652/4.1653
DEBUG    L2 gradient changed sign in 8944 of 16547, zero 93; msq chg 4.4905/9.73585 unchg 1.88682/3.60353
DEBUG    L3 gradient changed sign in 191 of 622, zero 58; msq chg 11.4661/42.8499 unchg 4.58857/8.75724
DEBUG gradient changed sign in 20847 of 34351, zero 377; msq chg 4.23437/9.11993 unchg 2.63049/4.09713
DEBUG bumped rate adj Down to 8.47807
DEBUG effective rate = 8.54929e-08 (adjusted by autoRate2 from 1.0084e-08)
      pass 1049 [26208], meansq err 0.045631/0.122631, gradient last 1652.718927 all 6253.706389; est ert 0.002151 below 0.002474
         p 1049, test: meansq err 0.062153, error rate 0.002491,  below 0.003986
         p 1049, train: meansq err 0.045612, error rate 0.002151,  below 0.002474
...

}

*/
