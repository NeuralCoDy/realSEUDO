
#include <utest/Utest.h>

int main(int ac, char **av)
{
	Utest driver(ac, av);


	extern Utest::Testcase nullref;
	driver.addcase(nullref, "nullref");

	extern Utest::Testcase construct;
	driver.addcase(construct, "construct");

	extern Utest::Testcase factory;
	driver.addcase(factory, "factory");

	extern Utest::Testcase assign;
	driver.addcase(assign, "assign");

	extern Utest::Testcase swap;
	driver.addcase(swap, "swap");

	extern Utest::Testcase onceref;
	driver.addcase(onceref, "onceref");

	extern Utest::Testcase onceref_casts;
	driver.addcase(onceref_casts, "onceref_casts");

	return driver.run();
}
