
#include <utest/Utest.h>

int main(int ac, char **av)
{
	Utest driver(ac, av);


	extern Utest::Testcase ops;
	driver.addcase(ops, "ops");

	extern Utest::Testcase bad_ops;
	driver.addcase(bad_ops, "bad_ops");

	return driver.run();
}
