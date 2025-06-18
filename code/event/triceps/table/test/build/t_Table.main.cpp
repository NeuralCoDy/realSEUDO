
#include <utest/Utest.h>

int main(int ac, char **av)
{
	Utest driver(ac, av);


	extern Utest::Testcase preLabel;
	driver.addcase(preLabel, "preLabel");

	extern Utest::Testcase exceptions;
	driver.addcase(exceptions, "exceptions");

	extern Utest::Testcase groupSize;
	driver.addcase(groupSize, "groupSize");

	extern Utest::Testcase clear;
	driver.addcase(clear, "clear");

	extern Utest::Testcase dumpAll;
	driver.addcase(dumpAll, "dumpAll");

	extern Utest::Testcase inputRef;
	driver.addcase(inputRef, "inputRef");

	return driver.run();
}
