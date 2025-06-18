//
// (C) Copyright 2022 Sergey A. Babkin.
// This file is a part of Triceps.
// See the file COPYRIGHT for the copyright notice and license information
//
//
// The base classes for a storage device.

#ifndef __Triceps_Device_h__
#define __Triceps_Device_h__

#include <inttypes.h>
#include <common/Errors.h>
#include <mem/Mtarget.h>
#include <mem/Autoref.h>

namespace TRICEPS_NS {

// Encapsulation of a device buffer. Virtual mainly for the sake of
// destructor, to allow different buffer freeing patterns.
// A buffer may not be used if the underlying device has been destroyed.
class DevBuffer : public Mtarget
{
public:
	DevBuffer(void *data, uint64_t size) :
		data_(data),
		size_(size)
	{ }
	virtual ~DevBuffer();

	template <typename T>
	T *data() const
	{
		return (T *)data_;
	}

	// A shorter name for a common case.
	uint8_t *data8() const
	{
		return (uint8_t *) data_;
	}

	uint64_t size() const
	{
		return size_;
	}

protected:
	void *data_;
	uint64_t size_;
};

// Representation of a storage device.
// Right now it has no internal synchronization, so Mtarget is just a wishful
// thinking, but if all goes well, it will be made internally synchronized
// eventually.
class Device : public Mtarget
{
public:
	// @param block_shift - shift for size of the basic allocation block, the
	//   I/O block gets set to the same size (can be overriden later). A block should
	//   normally not be smaller than a memory page.
	// @param size_blocks - the current size in allocation blocks.
	// @param max_size_blocks - the maximal size in allocation blocks (if not
	//   resizeable, must be the same as size_blocks).
	// @param resizeable - true if the storage can be resized (grown), such as if it
	//   represents a file.
	Device(uint64_t block_shift, uint64_t size_blocks, uint64_t max_size_blocks, bool resizeable) :
		block_shift_(block_shift),
		io_block_shift_(block_shift),
		size_blocks_(size_blocks),
		max_size_blocks_(max_size_blocks),
		resizeable_(resizeable)
	{ }
	virtual ~Device();
	
	// Resize the device if possible.
	// @param size_blocks - the new size in allocation blocks.
	virtual Erref resize(uint64_t size_blocks) = 0;

	// Trim/discard the contents. Depending on the device, on read these blocks
	// will return either all-zeroes or all-ones, or if the device doesn't support trim
	// then the contents is unchanged.
	// @param offset - byte offset, must be block-aligned
	// @param size - byte size, must be block-aligned
	virtual Erref trim(uint64_t offset, uint64_t size) = 0;

	// Wipe out the contents of a block that was previously trimmed. If the device
	// supports trim, does nothing (since the data was already erased by the
	// preceding trim). Otherwise writes zeroes. Use when the expectation of the
	// block being empty is important (i.e. when it doesn't get immediately overwritten
	// but will get filled later).
	// @param offset - byte offset, must be block-aligned
	// @param size - byte size, must be block-aligned
	virtual Erref wipeTrim(uint64_t offset, uint64_t size) = 0;

	// Read data.
	// @param offset - byte offset, must be block-aligned
	// @param size - byte size, must be block-aligned
	// @param result - the read data buffer, the caller should release the reference
	//   once done with the data. Note that writes at the same offset may
	//   change the buffer in case if the device implements it as memory-mapped
	//   I/O.  Multiple reads and writes may receive the same buffer and/or the
	//   same underlying data pointer, so writes and reads must be synchronized
	//   by the caller.
	virtual Erref read(uint64_t offset, uint64_t size, Autoref<DevBuffer> &result) = 0;

	// Get a buffer for a subsequent write.
	// @param offset - byte offset, must be io-block-aligned.
	// @param size - byte size, must be io-block-aligned.
	// @param result - the write data buffer, the caller should fill the buffer and
	//   then call write(). Depending on the device implementation, the buffer might
	//   be memory-mapped, then any changes to it would directly apply and write()
	//   would do nothing, which means that the writes and reads must be synchronized
	//   by the caller.
	virtual Erref writePrepare(uint64_t offset, uint64_t size, Autoref<DevBuffer> &result) = 0;

	// Write data.
	// @param offset - byte offset, must be io-block-aligned and the same as prepared.
	// @param size - byte size, must be io-block-aligned and the same as prepared.
	// @param buffer - buffer returned by writePrepare().
	virtual Erref write(uint64_t offset, uint64_t size, const Autoref<DevBuffer> &buffer) = 0;

	// Sync the written data.
	virtual Erref sync() = 0;

	uint64_t block_shift() const
	{
		return block_shift_;
	}

	uint64_t block_size() const
	{
		return 1 << block_shift_;
	}

	// Mask for the intra-block offsets.
	uint64_t block_mask() const
	{
		return block_size() - 1;
	}

	uint64_t io_block_shift() const
	{
		return io_block_shift_;
	}

	uint64_t io_block_size() const
	{
		return 1 << io_block_shift_;
	}

	// Mask for the intra-block offsets.
	uint64_t io_block_mask() const
	{
		return io_block_size() - 1;
	}

	// not const, may be resized!
	uint64_t device_size_blocks()
	{
		return size_blocks_;
	}

	// not const, may be resized!
	uint64_t device_size()
	{
		return size_blocks_ << block_shift_;
	}

	// not const, may be resized!
	uint64_t max_device_size_blocks()
	{
		return max_size_blocks_;
	}

	// not const, may be resized!
	uint64_t max_device_size()
	{
		return max_size_blocks_ << block_shift_;
	}

	bool isResizeable() const
	{
		return resizeable_;
	}

	// Check that the offset and size are proper.
	Erref checkOffset(uint64_t offset, uint64_t size);

protected:
	// Shift (i.e. log2) of a basic allocation block.
	uint64_t block_shift_;

	// Shift (i.e. log2) of a smallest I/O block. Could be smaller than
	// the allocation block.
	uint64_t io_block_shift_;

	// Size in allocation blocks.
	uint64_t size_blocks_;

	// Maximal size in allocation blocks.
	uint64_t max_size_blocks_;

	// True if the storage can be resized (grown).
	bool resizeable_;
};

}; // TRICEPS_NS

#endif // __Triceps_Device_h__
