
#include <utest/Utest.h>

int main(int ac, char **av)
{
	Utest driver(ac, av);


	extern Utest::Testcase fifoIndex;
	driver.addcase(fifoIndex, "fifoIndex");

	extern Utest::Testcase fifoIndexLimit;
	driver.addcase(fifoIndexLimit, "fifoIndexLimit");

	extern Utest::Testcase fifoIndexJumping;
	driver.addcase(fifoIndexJumping, "fifoIndexJumping");

	extern Utest::Testcase fifoIndexLimitReplace;
	driver.addcase(fifoIndexLimitReplace, "fifoIndexLimitReplace");

	extern Utest::Testcase fifoIndexLimitNoReplace;
	driver.addcase(fifoIndexLimitNoReplace, "fifoIndexLimitNoReplace");

	extern Utest::Testcase deepNested;
	driver.addcase(deepNested, "deepNested");

	return driver.run();
}
