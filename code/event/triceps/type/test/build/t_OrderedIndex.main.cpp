
#include <utest/Utest.h>

int main(int ac, char **av)
{
	Utest driver(ac, av);


	extern Utest::Testcase orderedIndex;
	driver.addcase(orderedIndex, "orderedIndex");

	extern Utest::Testcase orderedNested;
	driver.addcase(orderedNested, "orderedNested");

	extern Utest::Testcase orderedBadField;
	driver.addcase(orderedBadField, "orderedBadField");

	extern Utest::Testcase orderedIndexScalar;
	driver.addcase(orderedIndexScalar, "orderedIndexScalar");

	extern Utest::Testcase orderedIndexArray;
	driver.addcase(orderedIndexArray, "orderedIndexArray");

	return driver.run();
}
