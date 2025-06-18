
#include <utest/Utest.h>

int main(int ac, char **av)
{
	Utest driver(ac, av);


	extern Utest::Testcase throw_catch;
	driver.addcase(throw_catch, "throw_catch");

	extern Utest::Testcase abort;
	driver.addcase(abort, "abort");

	return driver.run();
}
