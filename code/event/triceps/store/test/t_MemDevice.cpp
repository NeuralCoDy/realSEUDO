//
// (C) Copyright 2022 Sergey A. Babkin.
// This file is a part of Triceps.
// See the file COPYRIGHT for the copyright notice and license information
//
//
// Test of the memory-based device, and common device methods.

#include <utest/Utest.h>
#include <store/MemDevice.h>
#include <string.h>

class TestDevice : public MemDevice
{
public:
	TestDevice(uint64_t block_shift, uint64_t size_blocks, uint64_t max_size_blocks, bool resizeable) :
		MemDevice(block_shift, size_blocks, max_size_blocks, resizeable)
	{ }

	uint8_t *data()
	{
		return data_;
	}
};

UTESTCASE ops(Utest *utest)
{
	TestDevice dev(12, 10, 10, false);

	UT_NOERROR(dev.trim(1 << dev.block_shift(), 1 << dev.block_shift()));

	Autoref<DevBuffer> buf;

	UT_NOERROR(dev.writePrepare(1 << dev.block_shift(), 1 << dev.block_shift(), buf));
	UT_IS(buf->size(), 1 << dev.block_shift());
	*buf->data<uint64_t>() = 0x12345678;
	buf->data8()[8] = 0x9A;
	// This is a direct pointer, so it shows right away.
	UT_IS(*(uint64_t*)(dev.data() + dev.block_size()), 0x12345678);
	UT_IS(dev.data()[dev.block_size() + 8], 0x9A);

	// A no-op, just check that it works.
	UT_NOERROR(dev.write(1 << dev.block_shift(), 1 << dev.block_shift(), buf));

	buf = nullptr;
	UT_NOERROR(dev.read(1 << dev.block_shift(), 1 << dev.block_shift(), buf));
	UT_IS(*buf->data<uint64_t>(), 0x12345678);
	UT_IS(buf->data8()[8], 0x9A);

	UT_NOERROR(dev.wipeTrim(1 << dev.block_shift(), 1 << dev.block_shift()));

	buf = nullptr;
	UT_NOERROR(dev.read(1 << dev.block_shift(), 1 << dev.block_shift(), buf));
	UT_IS(*buf->data<uint64_t>(), 0);
	UT_IS(buf->data8()[8], 0);

	UT_NOERROR(dev.sync());
}

UTESTCASE bad_ops(Utest *utest)
{
	Erref err;
	TestDevice dev(12, 10, 10, false);

	// Not supported yet.
	err = dev.resize(20);
	UT_IS(err.print(), "Resizing a MemDevice is not supported yet.\n");

	Autoref<DevBuffer> buf;

	err = (dev.writePrepare(dev.block_size() + 1, dev.block_size(), buf));
	UT_IS(err.print(), "Offset 0x1001 is not aligned to block size 0x1000\n");
	err = (dev.writePrepare(dev.block_size(), dev.block_size() + 1, buf));
	UT_IS(err.print(), "Size 0x1001 is not aligned to block size 0x1000\n");
	err = (dev.writePrepare(dev.block_size(), -1, buf));
	UT_IS(err.print(), "Offset 0x1000 size 0xffffffffffffffff refers past the device size 0xa000\n");
	err = (dev.writePrepare(0, dev.block_size() * 11, buf));
	UT_IS(err.print(), "Offset 0x0 size 0xb000 refers past the device size 0xa000\n");

	err = (dev.read(dev.block_size() + 1, dev.block_size(), buf));
	UT_IS(err.print(), "Offset 0x1001 is not aligned to block size 0x1000\n");
	err = (dev.read(dev.block_size(), dev.block_size() + 1, buf));
	UT_IS(err.print(), "Size 0x1001 is not aligned to block size 0x1000\n");
	err = (dev.read(dev.block_size(), -1, buf));
	UT_IS(err.print(), "Offset 0x1000 size 0xffffffffffffffff refers past the device size 0xa000\n");
	err = (dev.read(0, dev.block_size() * 11, buf));
	UT_IS(err.print(), "Offset 0x0 size 0xb000 refers past the device size 0xa000\n");

	err = (dev.wipeTrim(dev.block_size() + 1, dev.block_size()));
	UT_IS(err.print(), "Offset 0x1001 is not aligned to block size 0x1000\n");
	err = (dev.wipeTrim(dev.block_size(), dev.block_size() + 1));
	UT_IS(err.print(), "Size 0x1001 is not aligned to block size 0x1000\n");
	err = (dev.wipeTrim(dev.block_size(), -1));
	UT_IS(err.print(), "Offset 0x1000 size 0xffffffffffffffff refers past the device size 0xa000\n");
	err = (dev.wipeTrim(0, dev.block_size() * 11));
	UT_IS(err.print(), "Offset 0x0 size 0xb000 refers past the device size 0xa000\n");
}
