// Test of EventTable.

#include <stdio.h>
#include <math.h>
#include "test.hpp"
#include "event_table.hpp"

int testInit(const char *tname)
{
	EventTable et;
	Erref err = et.lastError();
	if (err.hasError()) {
		printf("Errors:\n%s\n", err->print().c_str());
		return 1;
	}
	return 0;
}


// -------------------------------------------------------------------

int
main(int argc, char **argv)
{
	TestDriver td(argc, argv);
	int result = 0;

	RUN_TEST(result, td, testInit);

	return result;
}
