#include <functional>
#include "triceps/pw/ptwrap.h"
#include "triceps/common/Strprintf.h"
#include "triceps/type/AllTypes.h"
#include "triceps/table/Table.h"

using namespace TRICEPS_NS;

// The storage table with all the Triceps infrastructure around it,
// starting with a Unit.
class EventTable {
public:

	// Not all the opcodes get stored in the table, some get processed immediately.
	enum Opcode {
		OP_CLEAR,
		OP_GET,
		OP_GET_TRACE,
		OP_GET_LAST_FRAME,
		OP_SET,
		OP_SIZE,
		OP_DETECT,
		OP_DETECT_EARLY,
		OP_MOD_MERGE,
		OP_MOD_SUBTRACT,
		OP_MOD_SHARD,
	};

	// Type of table storing the data and its components.
	enum FieldId {
		OPCODE,
		FRAME_ID,
		STABLE_ID,
		BRIGHTNESS,
		STABLE_ID2,
		BRIGHTNESS2,
		MAX_FIELD_ID
	};

	// Initializes the table.
	EventTable();

	// Get the last error.
	Erref lastError()
	{
		return initError_;
	}

	// Clear the table contents.
	Erref clear();

	// Insert an event into the table
	Erref insert(
		Opcode op,
		int64_t frame_id,
		int64_t stable_id,
		double brightness,
		int64_t stable_id2,
		double brightness2);

	// Return the current table size
	// @param sz - the place to return the size (unless an error is returned).
	Erref size(int64_t &sz);

	// Get the data row by row.
	// @param callback - function where the contents will be fed row by row.
	//   Index is the row index, starting from 0.
	//   If the function returns an error, no further calls will be made,
	//   and that error will be returned back.
	Erref get(std::function<Erref(
		size_t index,
		Opcode op,
		int64_t frame_id,
		int64_t stable_id,
		double brightness,
		int64_t stable_id2,
		double brightness2)> callback);

	// Get the highest frame id.
	int64_t getLastFrame();

	// Get the trace for a single ROI.
	// @param data - place to return the trace
	// @param datasz - number of frames allocated in data
	// @param stable_id - stable ID of the ROI to trace
	// @param coeff - brightness coefficient, meaning depends on norm_mode
	// @param norm_mode - normalization mode, see rois_params.m
	Erref getTrace(double *data, int64_t datasz, int64_t stable_id, double coeff, int norm_mode);

protected:
	// Global lock that synchronizes access to the table
	mutable pw::pmutex glock_;

	// Triceps model storing the data.
	Autoref<Unit> unit_;
	Autoref<RowType> row_type_;
	Autoref<TableType> tab_type_;
	Autoref<Table> table_;

	// Error from initialization.
	Erref initError_;

	// The last reported frame.
	int64_t lastFrame_ = 0;

	// Initializes the table from scratch.
	Erref initTable();

	// Internal implementation of getTrace() that gets called recursively for
	// processing of each split or merge.
	// @param data - place to return the trace
	// @param filled - mark with true whenever an OP_DETECT if filled for that frame
	//   (but not OP_DETECT_EARLY)
	// @param min_frame - the lowest frame id to include
	// @param max_frame - the highest frame id to include
	// @param stable_id - stable ID of the ROI to trace
	// @param coeff - brightness coefficient, meaning depends on norm_mode
	// @param norm_mode - normalization mode, see rois_params.m
	Erref fillTrace(double *data, std::vector<bool> &filled, int64_t min_frame, int64_t max_frame,
		int64_t stable_id, double coeff, int norm_mode);
};
