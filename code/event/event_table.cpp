#include <vector>
#include "event_table.hpp"
#include "triceps/mem/Rhref.h"

// #include "mex.h"

class SortOnFrameId : public SortedIndexCondition
{
public:
	// no internal configuration, all copies are the same
	SortOnFrameId()
	{ }
	SortOnFrameId(const SortOnFrameId *other, Table *t) :
		SortedIndexCondition(other, t)
	{ }
	virtual TreeIndexType::Less *tableCopy(Table *t) const
	{
		return new SortOnFrameId(this, t);
	}
	virtual bool equals(const SortedIndexCondition *sc) const
	{
		return true;
	}
	virtual bool match(const SortedIndexCondition *sc) const
	{
		return true;
	}
	virtual void printTo(string &res, const string &indent = "", const string &subindent = "  ") const
	{
		res.append("SortOnFrameId()");
	}
	virtual SortedIndexCondition *copy() const
	{
		return new SortOnFrameId(*this);
	}

	virtual bool operator() (const RowHandle *r1, const RowHandle *r2) const
	{
		int32_t a = rt_->getInt64(r1->getRow(), EventTable::FRAME_ID);
		int32_t b = rt_->getInt64(r2->getRow(), EventTable::FRAME_ID);
		return (a < b);
	}
};

EventTable::EventTable()
{
	initTable();
}

Erref EventTable::initTable()
{
	initError_ = nullptr;

	RowType::FieldVec fields(MAX_FIELD_ID);
	fields[OPCODE] = RowType::Field("opcode", Type::r_uint8);
	fields[FRAME_ID] = RowType::Field("frame_id", Type::r_int64);
	fields[STABLE_ID] = RowType::Field("stable_id", Type::r_int64);
	fields[BRIGHTNESS] = RowType::Field("brightness", Type::r_float64);
	fields[STABLE_ID2] = RowType::Field("stable_id2", Type::r_int64);
	fields[BRIGHTNESS2] = RowType::Field("brightness2", Type::r_float64);

	unit_ = new Unit("event_unit");
	row_type_ = new CompactRowType(fields);
	if (initError_.fAppend(row_type_->getErrors(), "In row type:"))
		return initError_;

	tab_type_ = TableType::make(row_type_)
		->addSubIndex("onStableId", HashedIndexType::make(
				(new NameSet())->add("stable_id")
			)
#if 0
			->addSubIndex("onFrameId", SortedIndexType::make(new SortOnFrameId()))
#endif
			->addSubIndex("reverse", FifoIndexType::make()->setReverse(true))
		)
		->addSubIndex("AllSequential", FifoIndexType::make())
		;
	tab_type_->initialize();
	if (initError_.fAppend(tab_type_->getErrors(), "In table type:"))
		return initError_;
	
	table_ = tab_type_->makeTable(unit_, "events");

	return initError_;
}

Erref EventTable::clear()
{
	pw::lockmutex lm(glock_);

	if (initError_)
		return initError_;

	Erref err;
	try {
		table_->clear();
	} catch (Exception &e) {
		err.f("%s", e.getErrors()->print().c_str());
	}
	return err;
}
 
Erref EventTable::insert(
	Opcode op,
	int64_t frame_id,
	int64_t stable_id,
	double brightness,
	int64_t stable_id2,
	double brightness2)
{
	pw::lockmutex lm(glock_);

	if (initError_)
		return initError_;

	if (frame_id > lastFrame_)
		lastFrame_ = frame_id;

	Erref err;

	try {
		uint8_t opx = (uint8_t) op;
		FdataVec fd;
		fd.emplace_back(Fdata(true, &opx, sizeof(opx)));
		fd.emplace_back(Fdata(true, &frame_id, sizeof(frame_id)));
		fd.emplace_back(Fdata(true, &stable_id, sizeof(stable_id)));
		fd.emplace_back(Fdata(true, &brightness, sizeof(brightness)));
		fd.emplace_back(Fdata(true, &stable_id2, sizeof(stable_id2)));
		fd.emplace_back(Fdata(true, &brightness2, sizeof(brightness2)));

		Rowref row(row_type_, row_type_->makeRow(fd));

		if (!table_->insertRow(row)) {
			err.f("Table insert failed");
		}
	} catch (Exception &e) {
		err.f("%s", e.getErrors()->print().c_str());
	}
	return err;
}

Erref EventTable::size(int64_t &sz)
{
	pw::lockmutex lm(glock_);

	if (initError_)
		return initError_;

	Erref err;

	sz = (int64_t) table_->size();

	return err;
}

Erref EventTable::get(std::function<Erref(
	size_t index,
	Opcode op,
	int64_t frame_id,
	int64_t stable_id,
	double brightness,
	int64_t stable_id2,
	double brightness2)> callback)
{
	pw::lockmutex lm(glock_);

	if (initError_)
		return initError_;

	Erref err;

	try {
		IndexType *seqidx = tab_type_->findSubIndex("AllSequential");
		if (seqidx == NO_INDEX_TYPE) {
			err.f("Cannot find the index AllSequential");
			return err;
		}

		size_t count = 0;
		for (RowHandle *rh = table_->beginIdx(seqidx); rh != nullptr; rh = table_->nextIdx(seqidx, rh), count++) {
			const Row *r = rh->getRow();
			
			err.fAppend(
				callback(
					count,
					(Opcode)row_type_->getUint8(r, OPCODE),
					row_type_->getInt64(r, FRAME_ID),
					row_type_->getInt64(r, STABLE_ID),
					row_type_->getFloat64(r, BRIGHTNESS),
					row_type_->getInt64(r, STABLE_ID2),
					row_type_->getFloat64(r, BRIGHTNESS2)),
				"On row %zd:", count);
			if (err.hasError())
				break;
		}
	} catch (Exception &e) {
		err.f("%s", e.getErrors()->print().c_str());
	}

	return err;
}

int64_t EventTable::getLastFrame()
{
	pw::lockmutex lm(glock_);

	return lastFrame_;
}

Erref EventTable::getTrace(double *data, int64_t datasz, int64_t stable_id, double coeff, int norm_mode)
{
	pw::lockmutex lm(glock_);

	if (initError_)
		return initError_;

	Erref err;

	// To mark, which frames have already been filled.
	std::vector<bool> filled(datasz, false);

	try {
		err.fAppend(fillTrace(data, filled, 1, datasz, stable_id, coeff, norm_mode),
			"When processing stable_id %zd:", (size_t) stable_id);
	} catch (Exception &e) {
		err.f("%s", e.getErrors()->print().c_str());
	}

	return err;
}

Erref EventTable::fillTrace(double *data, std::vector<bool> &filled,
	int64_t min_frame, int64_t max_frame, int64_t stable_id, double coeff, int norm_mode)
{
	Erref err;

	IndexType *id_idx = tab_type_->findSubIndex("onStableId");
	if (id_idx == NO_INDEX_TYPE) {
		err.f("Cannot find the index onStableId");
		return err;
	}
	IndexType *rev_idx = id_idx->findSubIndex("reverse");
	if (rev_idx == NO_INDEX_TYPE) {
		err.f("Cannot find the index reverse");
		return err;
	}

	FdataVec fd(MAX_FIELD_ID);
	fd[STABLE_ID].setPtr(true, &stable_id, sizeof(stable_id));

	Rowref proto_row(row_type_, row_type_->makeRow(fd));
	Rhref proto_rh(table_, table_->makeRowHandle(proto_row));

	RowHandle *iter = table_->findIdx(id_idx, proto_rh);
	RowHandle *last = nullptr;
	if (iter != nullptr) {
		last = table_->nextGroupIdx(rev_idx, iter);
	}

	for (; iter != last; iter = table_->nextIdx(rev_idx, iter)) {
		const Row *r = iter->getRow();

		Opcode op = (Opcode)row_type_->getUint8(r, OPCODE);
		int64_t stable_id_row = row_type_->getInt64(r, STABLE_ID);
		int64_t frame_id = row_type_->getInt64(r, FRAME_ID);
		double b1 = row_type_->getFloat64(r, BRIGHTNESS);
		int64_t stable_id2 = row_type_->getInt64(r, STABLE_ID2);
		double b2 = row_type_->getFloat64(r, BRIGHTNESS2);

		// mexPrintf("Got ROI %zd frame %zd\n", (size_t)stable_id_row, (size_t)frame_id);

		if (frame_id < min_frame)
			break;
		if (frame_id > max_frame)
			continue;

		double brightness;

		// Index in the data array
		int64_t idx = frame_id - 1;

		switch (op) {
		case OP_DETECT: {
			if (norm_mode == 0) {
				brightness = b1 / b2 * coeff;
			} else {
				brightness = b1 * b2 / coeff;
			}

			if (!filled[idx]) {
				data[idx] = brightness;
				filled[idx] = true;
			} else if (data[idx] < brightness) {
				data[idx] = brightness;
			}
		} break;
		case OP_DETECT_EARLY: {
			if (!filled[idx]) {
				if (norm_mode == 0) {
					brightness = b1 / b2 * coeff;
				} else {
					brightness = b1 * b2 / coeff;
				}

				if (data[idx] < brightness) {
					data[idx] = brightness;
				}
			}
		} break;
		case OP_MOD_MERGE: {
			if (stable_id2 >= 0) {
				double c2 = coeff;
				if (norm_mode == 1) {
					c2 *= b2;
				}

				err.fAppend(fillTrace(data, filled, min_frame, frame_id, stable_id2, c2, norm_mode),
					"When processing stable_id %zd:", (size_t) stable_id2);
			}
			if (norm_mode == 1) {
				coeff *= b1;
			}
		} break;
		case OP_MOD_SUBTRACT: {
			if (norm_mode == 1) {
				coeff *= b1;
			}
		} break;
		case OP_MOD_SHARD: {
			if (stable_id2 >= 0) {
				double c2 = coeff;
				if (norm_mode == 1) {
					c2 *= b1;
				}

				err.fAppend(fillTrace(data, filled, min_frame, frame_id, stable_id2, c2, norm_mode),
					"When processing stable_id %zd:", (size_t) stable_id2);
			}
		} break;
		}
	}

	return err;
}
