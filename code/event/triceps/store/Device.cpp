//
// (C) Copyright 2022 Sergey A. Babkin.
// This file is a part of Triceps.
// See the file COPYRIGHT for the copyright notice and license information
//
//
// The base classes for a storage device.

#include <stdio.h>
#include <store/Device.h>

namespace TRICEPS_NS {

DevBuffer::~DevBuffer()
{ }

Device::~Device()
{ }

Erref
Device::checkOffset(uint64_t offset, uint64_t size)
{
	Erref err;
	uint64_t devsz = device_size();
	// Includes the check for overflow.
	if (offset >= devsz || offset + size < offset || offset + size > devsz) {
		err.f("Offset 0x%" PRIx64 " size 0x%" PRIx64 " refers past the device size 0x%" PRIx64, offset, size, devsz);
		return err;
	}
	uint64_t mask = block_mask();
	if (offset & mask) {
		err.f("Offset 0x%" PRIx64 " is not aligned to block size 0x%" PRIx64, offset, mask + 1);
		return err;
	}
	if (size == 0) {
		err.f("Size must not be 0");
		return err;
	}
	if (size & mask) {
		err.f("Size 0x%" PRIx64 " is not aligned to block size 0x%" PRIx64, size, mask + 1);
		return err;
	}

	return err;
}

}; // TRICEPS_NS
