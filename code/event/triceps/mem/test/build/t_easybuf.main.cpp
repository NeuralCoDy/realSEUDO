
#include <utest/Utest.h>

int main(int ac, char **av)
{
	Utest driver(ac, av);


	extern Utest::Testcase buf;
	driver.addcase(buf, "buf");

	return driver.run();
}
