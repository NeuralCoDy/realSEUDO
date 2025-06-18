
#include <utest/Utest.h>

int main(int ac, char **av)
{
	Utest driver(ac, av);


	extern Utest::Testcase mkgadget;
	driver.addcase(mkgadget, "mkgadget");

	extern Utest::Testcase scheduling;
	driver.addcase(scheduling, "scheduling");

	return driver.run();
}
