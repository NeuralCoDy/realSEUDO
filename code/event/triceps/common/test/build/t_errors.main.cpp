
#include <utest/Utest.h>

int main(int ac, char **av)
{
	Utest driver(ac, av);


	extern Utest::Testcase simple;
	driver.addcase(simple, "simple");

	extern Utest::Testcase nested;
	driver.addcase(nested, "nested");

	extern Utest::Testcase absorb;
	driver.addcase(absorb, "absorb");

	extern Utest::Testcase errefAppend;
	driver.addcase(errefAppend, "errefAppend");

	return driver.run();
}
