// Simple test helpers.

#include <stdio.h>
#include <string.h>
#include "test.hpp"

bool verbose = false;

int TestDriver::runTest(const char *name, TestFunction *function)
{
	// -v runs all the tests in verbose mode
	if (argc_ > 1 && strcmp("-v", argv_[1])) {
		bool run = false;
		for (int i = 1; i < argc_; i++) {
			if (!strcmp(name, argv_[i])) {
				run = true;
				break;
			}
		}
		if (!run)
			return 0;
	}
	printf("  ==== %s =====\n", name);
	return function(name);
}
