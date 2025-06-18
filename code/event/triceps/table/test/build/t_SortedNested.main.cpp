
#include <utest/Utest.h>

int main(int ac, char **av)
{
	Utest driver(ac, av);


	extern Utest::Testcase primaryIndex;
	driver.addcase(primaryIndex, "primaryIndex");

	extern Utest::Testcase uninitialized;
	driver.addcase(uninitialized, "uninitialized");

	extern Utest::Testcase tableops;
	driver.addcase(tableops, "tableops");

	return driver.run();
}
