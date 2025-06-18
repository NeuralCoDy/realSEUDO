
#include <utest/Utest.h>

int main(int ac, char **av)
{
	Utest driver(ac, av);


	extern Utest::Testcase findSimpleType;
	driver.addcase(findSimpleType, "findSimpleType");

	extern Utest::Testcase print;
	driver.addcase(print, "print");

	return driver.run();
}
