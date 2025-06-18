
#include <utest/Utest.h>

int main(int ac, char **av)
{
	Utest driver(ac, av);


	extern Utest::Testcase stringConst;
	driver.addcase(stringConst, "stringConst");

	extern Utest::Testcase badName;
	driver.addcase(badName, "badName");

	extern Utest::Testcase tableops;
	driver.addcase(tableops, "tableops");

	extern Utest::Testcase tableops_fret;
	driver.addcase(tableops_fret, "tableops_fret");

	extern Utest::Testcase bad_fret;
	driver.addcase(bad_fret, "bad_fret");

	extern Utest::Testcase aggLast;
	driver.addcase(aggLast, "aggLast");

	extern Utest::Testcase aggBasicSum;
	driver.addcase(aggBasicSum, "aggBasicSum");

	extern Utest::Testcase aggSum;
	driver.addcase(aggSum, "aggSum");

	extern Utest::Testcase aggSumBad;
	driver.addcase(aggSumBad, "aggSumBad");

	return driver.run();
}
