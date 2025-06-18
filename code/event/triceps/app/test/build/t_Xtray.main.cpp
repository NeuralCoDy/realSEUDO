
#include <utest/Utest.h>

int main(int ac, char **av)
{
	Utest driver(ac, av);


	extern Utest::Testcase xtray;
	driver.addcase(xtray, "xtray");

	return driver.run();
}
