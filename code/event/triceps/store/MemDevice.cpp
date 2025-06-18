//
// (C) Copyright 2022 Sergey A. Babkin.
// This file is a part of Triceps.
// See the file COPYRIGHT for the copyright notice and license information
//
//
// The memory-based storage device.

#include <stdio.h>
#include <string.h>
#include <store/MemDevice.h>

namespace TRICEPS_NS {

MemDevBuffer::~MemDevBuffer()
{ }

MemDevice::MemDevice(uint64_t block_shift, uint64_t size_blocks, uint64_t max_size_blocks, bool resizeable) :
	Device(block_shift, size_blocks, max_size_blocks, resizeable),
	data_(new uint8_t[size_blocks << block_shift])
{ }

MemDevice::~MemDevice()
{
	delete[] data_;
}

// From Device.
Erref
MemDevice::resize(uint64_t size_blocks)
{
	Erref err;
	err.f("Resizing a MemDevice is not supported yet.");
	return err;
}

Erref
MemDevice::trim(uint64_t offset, uint64_t size)
{
	return Erref();
}

Erref
MemDevice::wipeTrim(uint64_t offset, uint64_t size)
{
	Erref err = checkOffset(offset, size);
	if (err.hasError()) {
		return err;
	}

	// Write 0s.
	Autoref<DevBuffer> buf;
	err.fAppend(writePrepare(offset, size, buf), "In writePrepare():");
	if (err.hasError()) {
		return err;
	}
	memset(buf->data8(), 0, size);
	err.fAppend(write(offset, size, buf), "In write():");

	return err;
}

Erref
MemDevice::read(uint64_t offset, uint64_t size, Autoref<DevBuffer> &result)
{
	Erref err = checkOffset(offset, size);
	if (err.hasError()) {
		return err;
	}
	result = new MemDevBuffer(data_ + offset, size);
	return err;
}

Erref
MemDevice::writePrepare(uint64_t offset, uint64_t size, Autoref<DevBuffer> &result)
{
	Erref err = checkOffset(offset, size);
	if (err.hasError()) {
		return err;
	}
	result = new MemDevBuffer(data_ + offset, size);
	return err;
}

Erref
MemDevice::write(uint64_t offset, uint64_t size, const Autoref<DevBuffer> &buffer)
{
	return Erref();
}

Erref MemDevice::sync()
{
	return Erref();
}

}; // TRICEPS_NS

