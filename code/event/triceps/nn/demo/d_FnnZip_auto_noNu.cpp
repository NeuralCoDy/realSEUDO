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

	// Momentum with fast braking on gradient sign change and fixed nu (or eta),
	// and automatic descent rate adjustment.
	options.isClassifier_ = false;
	options.maxMultiplier_ = 100;
	options.weightSaturation_ = 1.;
	options.momentum_ = true;
	options.momentumFixedNu_ = false;
	options.autoRate2_ = true;
	// options.enableWeightFloor_ = true;
	options.tweakRate_ = 0;
#if 0 
	options.autoRate_ = true;
	// initial value
	options.trainingRate_ = 1e-6;
	// options.scaleRatePerLayer_ = true;
	// Also see the adjustment in trainSquareFunction()
	options.trainingRateScale_ = 0.01; // 5e-4;
#endif

	TrainingFormat format = TF_8X8;
	FloatNeuralNet::ActivationFunction activation = FloatNeuralNet::LEAKY_RELU;
	// From the internets, 3 levels should be enough for RELU, and 2 layers
	// seem to work OK for CORNER.
	// The first element gets auto-filled.
	FloatNeuralNet::LevelSizeVector levels = {64, 64, 32, 10};

	string checkpoint; // = fpath + "/leaky_nomom.ckp";
	// batch > 1 doesn't mix with options.isClassifier_
	int batchSize = 1;
	seed = 1667553859;

	double rate = 0.05;
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

	if (!checkpoint.empty()) {
		printf("Checkpointing to %s\n", checkpoint.c_str());
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

	int nPasses = 10000;
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


