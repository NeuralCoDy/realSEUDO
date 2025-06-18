
#include <utest/Utest.h>

int main(int ac, char **av)
{
	Utest driver(ac, av);


	extern Utest::Testcase zipcodes;
	driver.addcase(zipcodes, "zipcodes");

	return driver.run();
}
