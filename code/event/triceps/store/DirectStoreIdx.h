//
// (C) Copyright 2022 Sergey A. Babkin.
// This file is a part of Triceps.
// See the file COPYRIGHT for the copyright notice and license information
//
//
// A simple index that maps uint64 to uint64.

#ifndef __Triceps_DirectStoreIdx_h__
#define __Triceps_DirectStoreIdx_h__

#include <map>
#include <store/Device.h>

namespace TRICEPS_NS {

// A simple index that maps uint64 to uint64.
class DirectStoreIdx : public Mtarget
{
	// Level of block in the index. Level 0 contains the actual entries,
	// higher levels contain references to the next-lower-level index blocks.
	typedef uint16_t BlockLevel;

	// Header for on-disk representation of a store block.
	// Size rounded to uint64_t granularity.
	struct Block : public StoreBlock
	{
		// Level of block in the index.
		BlockLevel level_;
		uint16_t dsib_reserved_;
		// Count of valid entries in block.
		uint32_t count_;
	};

	// A key-value pair of index.
	struct KeyVal
	{
		uint64_t key_;
		uint64_t value_;
	};

	// Representation of an index block loaded in memory.
	class LoadedBlock : public Mtarget
	{
	public:
		SBXXX TODO;

	protected:
		// First key contained in this block.
		uint64_t first_key_;
		// Address of block on the device.
		uint64_t dev_addr_;

		// Size of device block in bytes.
		uint64_t block_size_;

		// Count of entries (contains the most current value between
		// buffer_ and tree_).
		size_t count_;

		// Refers to a copy of on-disk representation.
		Autoref<DevBuffer> buffer_;

		// An in-memory representation used to build and modify the block.
		// The tree representation is valid when buffer_ is NULL (since
		// as soon as we start modifying the block, its old serialized
		// contents stops being interesting).
		std::map<uint64_t, uint64_t> tree_;
	};

	SBXXX TODO;
}; // TRICEPS_NS

#endif // __Triceps_DirectStoreIdx_h__

