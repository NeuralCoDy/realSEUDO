
#include <utest/Utest.h>

int main(int ac, char **av)
{
	Utest driver(ac, av);


	extern Utest::Testcase rowset;
	driver.addcase(rowset, "rowset");

	return driver.run();
}
