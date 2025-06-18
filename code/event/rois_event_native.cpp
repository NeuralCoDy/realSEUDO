#include <stdio.h>
#include <string.h>
#include <string>
#include <memory>
#include <cinttypes>

#include "mex.h"
#include "matrix.h"

#include "event_table.hpp"

// This is a workaround for a symbol defined in the new libstac++ 6.30
// but not in matlab's copy of version 6.28. Comment out when Matlab
// catches up.
namespace std
{

void
__throw_bad_array_new_length(void)
{
	throw std::bad_array_new_length();
}

};

using namespace TRICEPS_NS;

// Prevents the leaks of generated strings on errors.
static std::string errorMsg;

// Load a scalar from Matlab.
// @param descr - description for debugging prints, not NULL enables the prints
// @param src - source Matlab scalar
// @return - the scalar value
double loadScalar(const char *descr, const mxArray *src)
{
	double v = mxGetScalar(src);
	if (descr != NULL) mexPrintf("%s(1, 1) = %f\n", descr, v);
	return v;
}

// Check that the value of an argument is numeric and a scalar, log a Matlab
// error if not (which also exits any current code).
// @param idx - argument index
// @param name - argument name
// @param arg - value to check
Erref checkNumericScalar(int idx, const char *name, const mxArray *arg)
{
	Erref err;

	if (!mxIsNumeric(arg) || mxIsComplex(arg)) {
		err.f("argument %d (%s) must contain a number", idx + 1, name);
	}
	if (mxGetNumberOfElements(arg) != 1) {
		err.f("argument %d (%s) must contain a scalar", idx + 1, name);
	}

	return err;
}

// Load a string from Matlab.
// @param descr - description for debugging prints, not NULL enables the prints
// @param src - source Matlab string or char array
// @return - the string value
char* loadString(const char *descr, const mxArray *src)
{
	char *v = mxArrayToString(src);
	if (descr != NULL) mexPrintf("%s(1, 1) = %s\n", descr, v);
	return v;
}

// Check that the value of an argument is a string, log a Matlab
// error if not (which also exits any current code).
// @param idx - argument index
// @param name - argument name
// @param arg - value to check
Erref checkString(int idx, const char *name, const mxArray *arg)
{
	Erref err;
	auto clid = mxGetClassID(arg);
	if (clid != mxCHAR_CLASS) {
		err.f("argument %d (%s) must contain characters in single quotes, got class %d", idx + 1, name, clid);
	}
	return err;
}

// Inputs:
//   string opcode - the opcode to process
enum InArgIdx {
	InOpcode,
	InFrameId,
	InStableId,
	InBrightness,
	InStableId2,
	InBrightness2,
};

// Table to store the events.
// An actual static object doesn't work (crashes), so make it a shared pointer and
// initialize on first use.
static std::shared_ptr<EventTable> table;

extern "C" {

void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

	if (nrhs < 1) {
		mexErrMsgIdAndTxt(
			"rois_event_native:Usage",
			"Use: rois_event_native(opcode)\n"
			"  opcode - one of ... TODO ..."
			);
		return;
	}

	{
		errorMsg.clear();
		Erref err;

		do {
			err.fAppend(checkString(InOpcode, "opcode", prhs[InOpcode]), "Arg parsing:");
			if (err.hasError()) break;

			char* cmd = loadString(nullptr /*"opcode"*/, prhs[InOpcode]);

			if (table == nullptr) {
				table = std::make_shared<EventTable>();
				err.fAppend(table->lastError(), "Event table creation:");
				if (err.hasError()) break;
			}

			EventTable::Opcode op;
			if (!strcmp(cmd, "clear")) {
				op = EventTable::OP_CLEAR;
			} else if (!strcmp(cmd, "get")) {
				op = EventTable::OP_GET;
			} else if (!strcmp(cmd, "get_trace")) {
				op = EventTable::OP_GET_TRACE;
			} else if (!strcmp(cmd, "get_last_frame")) {
				op = EventTable::OP_GET_LAST_FRAME;
			} else if (!strcmp(cmd, "set")) {
				op = EventTable::OP_SET;
			} else if (!strcmp(cmd, "size")) {
				op = EventTable::OP_SIZE;
			} else if (!strcmp(cmd, "detect")) {
				op = EventTable::OP_DETECT;
			} else if (!strcmp(cmd, "detect_early")) {
				op = EventTable::OP_DETECT_EARLY;
			} else if (!strcmp(cmd, "mod_merge")) {
				op = EventTable::OP_MOD_MERGE;
			} else if (!strcmp(cmd, "mod_subtract")) {
				op = EventTable::OP_MOD_SUBTRACT;
			} else if (!strcmp(cmd, "mod_shard")) {
				op = EventTable::OP_MOD_SHARD;
			} else {
				err.f("Invalid opcode '%s'", cmd);
				break;
			}

			switch(op) {
			case EventTable::OP_CLEAR: {
				err = table->clear();
			} break;
			case EventTable::OP_GET: {
				int64_t sz;
				err.fAppend(table->size(sz), "Getting table size:");
				if (err.hasError())
					break;

				if (nlhs > 0) {
					mxArray *result = mxCreateCellMatrix(sz, 6);
					if (result == nullptr) {
						err.f("Failed to create the return cell matrix of %" PRId64 "x%d", sz, 6);
						break;
					}

					plhs[0] = result;

					err = table->get([&](
						size_t index,
						EventTable::Opcode op,
						int64_t frame_id,
						int64_t stable_id,
						double brightness,
						int64_t stable_id2,
						double brightness2) -> Erref {
							Erref err;
							if (index >= sz) {
								err.f("Too many rows");
								return err;
							}

							// Matlab has dimensions in backwards order
							mwIndex pos = index;

							const char *opstr = nullptr;
							switch(op) {
							case EventTable::OP_DETECT:
								opstr = "detect";
								break;
							case EventTable::OP_DETECT_EARLY:
								opstr = "detect_early";
								break;
							case EventTable::OP_MOD_MERGE:
								opstr = "mod_merge";
								break;
							case EventTable::OP_MOD_SUBTRACT:
								opstr = "mod_subtract";
								break;
							case EventTable::OP_MOD_SHARD:
								opstr = "mod_shard";
								break;
							default:
								err.f("Unexpected opcode %d", (int)op);
								return err;
							}

							mxSetCell(result, pos, mxCreateString(opstr));
							pos += sz;
							mxSetCell(result, pos, mxCreateDoubleScalar(frame_id));
							pos += sz;
							mxSetCell(result, pos, mxCreateDoubleScalar(stable_id));
							pos += sz;
							mxSetCell(result, pos, mxCreateDoubleScalar(brightness));
							pos += sz;
							mxSetCell(result, pos, mxCreateDoubleScalar(stable_id2));
							pos += sz;
							mxSetCell(result, pos, mxCreateDoubleScalar(brightness2));
							pos += sz;

							return err;
						});

				}
			} break;
			case EventTable::OP_GET_TRACE: {
				Erref clde = new Errors();
				clde->absorb(checkNumericScalar(InFrameId, "frame_id/last_frame", prhs[InFrameId]));
				clde->absorb(checkNumericScalar(InStableId, "stable_id", prhs[InStableId]));
				clde->absorb(checkNumericScalar(InBrightness, "brightness/coeff", prhs[InBrightness]));
				clde->absorb(checkNumericScalar(InStableId2, "stable_id2/norm_mode", prhs[InStableId2]));
				err.fAppend(clde, "Arg parsing:");
				if (err.hasError()) break;

				int64_t last_frame = (int64_t)loadScalar(nullptr /*"frame_id"*/, prhs[InFrameId]);
				int64_t stable_id = (int64_t)loadScalar(nullptr /*"stable_id"*/, prhs[InStableId]);
				double coeff = loadScalar(nullptr /*"brightness"*/, prhs[InBrightness]);
				int norm_mode = (int)loadScalar(nullptr /*"stable_id2"*/, prhs[InStableId2]);

				int64_t last_event_frame = table->getLastFrame();
				if (last_frame < last_event_frame)
					last_frame = last_event_frame;

				if (nlhs > 0) {
					plhs[0] = mxCreateDoubleMatrix(1, last_frame, mxREAL);
					if (plhs[0] == nullptr) {
						err.f("Failed to allocate a matrix of 1x%zd", (size_t) last_frame);
						break;
					}

					if (last_frame > 0) {
						err = table->getTrace((double *)mxGetData(plhs[0]), last_frame, stable_id, coeff, norm_mode);
					}
				}
			} break;
			case EventTable::OP_GET_LAST_FRAME: {
				int64_t frame = table->getLastFrame();
				if (nlhs > 0) {
					plhs[0] = mxCreateDoubleScalar(frame);
				}
			} break;
			case EventTable::OP_SET: {
				if (nrhs < 2) {
					err.f("The opcode '%s' requires an additional argument\n", cmd);
					break;
				}
				auto cells = prhs[1];
				if (!mxIsCell(cells)) {
					err.f("The opcode '%s' requires an additional argument to be a cell array\n", cmd);
					break;
				}
				mwIndex ht = mxGetM(cells);
				mwIndex wd = mxGetN(cells);

				if (ht == 0)
					break; // nothing to do

				if (wd != 6) {
					err.f("The opcode '%s' requires the cell array argument to be 6 wide, got %d\n", cmd, (int)wd);
					break;
				}

				err = table->clear();
				if (err.hasError())
					break;

				for (mwIndex row = 0; row < ht; row++) {
					mxArray *op_c = mxGetCell(cells, InOpcode * ht + row);
					err.fAppend(checkString(InOpcode, "opcode", op_c), "Row %zd parsing:", (size_t)row + 1);
					if (err.hasError()) break;

					char* op_str = loadString(nullptr /*"opcode"*/, op_c);
					EventTable::Opcode op_code;
					if (!strcmp(op_str, "detect")) {
						op_code = EventTable::OP_DETECT;
					} else if (!strcmp(op_str, "detect_early")) {
						op_code = EventTable::OP_DETECT_EARLY;
					} else if (!strcmp(op_str, "mod_merge")) {
						op_code = EventTable::OP_MOD_MERGE;
					} else if (!strcmp(op_str, "mod_subtract")) {
						op_code = EventTable::OP_MOD_SUBTRACT;
					} else if (!strcmp(op_str, "mod_shard")) {
						op_code = EventTable::OP_MOD_SHARD;
					} else {
						err.f("Row %zd parsing: Invalid opcode '%s'", (size_t)row + 1, op_str);
						break;
					}

					mxArray *frame_id_c = mxGetCell(cells, InFrameId * ht + row);
					err.fAppend(checkNumericScalar(InOpcode, "frame_id", frame_id_c), "Row %zd parsing:", (size_t)row + 1);
					if (err.hasError()) break;

					mxArray *stable_id_c = mxGetCell(cells, InStableId * ht + row);
					err.fAppend(checkNumericScalar(InOpcode, "stable_id", stable_id_c), "Row %zd parsing:", (size_t)row + 1);
					if (err.hasError()) break;

					mxArray *brightness_c = mxGetCell(cells, InBrightness * ht + row);
					err.fAppend(checkNumericScalar(InOpcode, "brightness", brightness_c), "Row %zd parsing:", (size_t)row + 1);
					if (err.hasError()) break;

					mxArray *stable_id2_c = mxGetCell(cells, InStableId2 * ht + row);
					err.fAppend(checkNumericScalar(InOpcode, "stable_id2", stable_id2_c), "Row %zd parsing:", (size_t)row + 1);
					if (err.hasError()) break;

					mxArray *brightness2_c = mxGetCell(cells, InBrightness2 * ht + row);
					err.fAppend(checkNumericScalar(InOpcode, "brightness2", brightness2_c), "Row %zd parsing:", (size_t)row + 1);
					if (err.hasError()) break;

					err.fAppend(
						table->insert(op_code,
							(int64_t)loadScalar(nullptr /*"frame_id"*/, frame_id_c),
							(int64_t)loadScalar(nullptr /*"stable_id"*/, stable_id_c),
							loadScalar(nullptr /*"brightness"*/, brightness_c),
							(int64_t)loadScalar(nullptr /*"stable_id2"*/, stable_id2_c),
							loadScalar(nullptr /*"brightness2"*/, brightness2_c)),
						"Table insert of row %zd:", (size_t)row + 1);
					if (err.hasError()) break;
				}

			} break;
			case EventTable::OP_SIZE: {
				int64_t sz;
				err.fAppend(table->size(sz), "Getting table size:");
				if (!err.hasError() && nlhs > 0) {
					plhs[0] = mxCreateDoubleScalar(sz);
				}
			} break;
			case EventTable::OP_DETECT:
			case EventTable::OP_DETECT_EARLY:
			case EventTable::OP_MOD_MERGE:
			case EventTable::OP_MOD_SUBTRACT:
			case EventTable::OP_MOD_SHARD: {
				if (nrhs != 6) {
					err.f("The opcode '%s' requires 6 arguments, got %d\n", cmd, nrhs);
					break;
				}

				Erref clde = new Errors();
				clde->absorb(checkNumericScalar(InFrameId, "frame_id", prhs[InFrameId]));
				clde->absorb(checkNumericScalar(InStableId, "stable_id", prhs[InStableId]));
				clde->absorb(checkNumericScalar(InBrightness, "brightness", prhs[InBrightness]));
				clde->absorb(checkNumericScalar(InStableId2, "stable_id2", prhs[InStableId2]));
				clde->absorb(checkNumericScalar(InBrightness2, "brightness2", prhs[InBrightness2]));
				err.fAppend(clde, "Arg parsing:");
				if (err.hasError()) break;

				err.fAppend(
					table->insert(op,
						(int64_t)loadScalar(nullptr /*"frame_id"*/, prhs[InFrameId]),
						(int64_t)loadScalar(nullptr /*"stable_id"*/, prhs[InStableId]),
						loadScalar(nullptr /*"brightness"*/, prhs[InBrightness]),
						(int64_t)loadScalar(nullptr /*"stable_id2"*/, prhs[InStableId2]),
						loadScalar(nullptr /*"brightness2"*/, prhs[InBrightness2])),
					"Table insert:");
			} break;
			default:
				break;
			}
		} while(false);

		if (err.hasError()) {
			errorMsg = err->print();
		}
	}

	if (!errorMsg.empty()) {
		mexErrMsgIdAndTxt(
			"rois_event_native:Error",
			"%s", errorMsg.c_str());
		return;
	}

	if (nlhs > 0 && plhs[0] == nullptr) {
		// Make an empty array to make the error shut up.
		plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
	}
}

};
