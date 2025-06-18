//
// (C) Copyright 2022 Sergey A. Babkin.
// This file is a part of Triceps.
// See the file COPYRIGHT for the copyright notice and license information
//
//
// The general structures of a data store.

#ifndef __Triceps_Store_h__
#define __Triceps_Store_h__

#include <inttypes.h>
#include <common/Errors.h>
#include <mem/Mtarget.h>
#include <mem/Autoref.h>

namespace TRICEPS_NS {

// Unique identifier of an entity (like an index or table) in store.
typedef uint32_t StoreEntity;

// Used for both offsets and sequential enumerations.
typedef uint64_t StoreOffset;

// Header of a block in store, allows to verify integrity.
// Size rounded to uint64_t granularity.
struct StoreBlock
{
	StoreEntity entity_;
	uint32_t sb_reserved_;
};

// On-disk contents of a superblock.
struct StoreSuperblock
{
	uint64_t magic_;
	uint32_t version_;
	// Size of valid data, on a uint64 granularity, with the 64-bit checksum following
	// afterwards.
	uint32_t data_size_;

	// Offset of the root block of the free block index.
	uint64_t free_idx_root_;
	// Offset of the root block of the second free block index. This index contains
	// the blocks that have been freed but cannot be reused before checkpointing,
	// because the previous checkpoint refers to them. They get moved to the main
	// free index right after checkpointing.
	uint64_t free_idx2_root_;

	SBXX TODO;
};

}; // TRICEPS_NS

#endif // __Triceps_Store_h__
