
#include <utest/Utest.h>

int main(int ac, char **av)
{
	Utest driver(ac, av);


	extern Utest::Testcase tray;
	driver.addcase(tray, "tray");

	return driver.run();
}
