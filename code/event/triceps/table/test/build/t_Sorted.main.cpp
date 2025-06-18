
#include <utest/Utest.h>

int main(int ac, char **av)
{
	Utest driver(ac, av);


	extern Utest::Testcase primaryIndex;
	driver.addcase(primaryIndex, "primaryIndex");

	extern Utest::Testcase tableops;
	driver.addcase(tableops, "tableops");

	extern Utest::Testcase tableops_exception;
	driver.addcase(tableops_exception, "tableops_exception");

	return driver.run();
}
