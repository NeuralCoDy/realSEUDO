
#include <utest/Utest.h>

int main(int ac, char **av)
{
	Utest driver(ac, av);


	extern Utest::Testcase mkshort;
	driver.addcase(mkshort, "mkshort");

	extern Utest::Testcase mkvshort;
	driver.addcase(mkvshort, "mkvshort");

	extern Utest::Testcase mklong;
	driver.addcase(mklong, "mklong");

	extern Utest::Testcase mkvlong;
	driver.addcase(mkvlong, "mkvlong");

	return driver.run();
}
