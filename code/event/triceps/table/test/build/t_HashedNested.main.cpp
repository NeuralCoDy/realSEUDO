
#include <utest/Utest.h>

int main(int ac, char **av)
{
	Utest driver(ac, av);


	extern Utest::Testcase primaryIndex;
	driver.addcase(primaryIndex, "primaryIndex");

	extern Utest::Testcase uninitialized;
	driver.addcase(uninitialized, "uninitialized");

	extern Utest::Testcase withError;
	driver.addcase(withError, "withError");

	extern Utest::Testcase tableops;
	driver.addcase(tableops, "tableops");

	extern Utest::Testcase queuing;
	driver.addcase(queuing, "queuing");

	return driver.run();
}
