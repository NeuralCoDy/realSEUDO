//
// (C) Copyright 2022 Sergey A. Babkin.
// This file is a part of Triceps.
// See the file COPYRIGHT for the copyright notice and license information
//
//
// The memory-based storage device.

#ifndef __Triceps_MemDevice_h__
#define __Triceps_MemDevice_h__

#include <store/Device.h>

namespace TRICEPS_NS {

// Just points to the right place in the greater memory buffer representing the
// whole device.
class MemDevBuffer : public DevBuffer
{
public:
	MemDevBuffer(void *data, uint64_t size) :
		DevBuffer(data, size)
	{ }

	virtual ~MemDevBuffer();
};

// A pseudo-device in memory, for convenience of testing.
class MemDevice : public Device
{
public:
	// Leaves the contents of the "device" uninitialized, with random garbage.
	//
	// @param block_shift - shift for size of the basic allocation block, the
	//   I/O block gets set to the same size (can be overriden later). A block should
	//   normally not be smaller than a memory page.
	// @param size_blocks - the current size in allocation blocks.
	// @param max_size_blocks - the maximal size in allocation blocks (if not
	//   resizeable, must be the same as size_blocks).
	// @param resizeable - true if the storage can be resized (grown), such as if it
	//   represents a file.
	MemDevice(uint64_t block_shift, uint64_t size_blocks, uint64_t max_size_blocks, bool resizeable);
	virtual ~MemDevice();
	
	// From Device.
	virtual Erref resize(uint64_t size_blocks);
	virtual Erref trim(uint64_t offset, uint64_t size);
	virtual Erref wipeTrim(uint64_t offset, uint64_t size);
	virtual Erref read(uint64_t offset, uint64_t size, Autoref<DevBuffer> &result);
	virtual Erref writePrepare(uint64_t offset, uint64_t size, Autoref<DevBuffer> &result);
	virtual Erref write(uint64_t offset, uint64_t size, const Autoref<DevBuffer> &buffer);
	virtual Erref sync();

protected:
	uint8_t *data_;
};

}; // TRICEPS_NS

#endif // __Triceps_MemDevice_h__
