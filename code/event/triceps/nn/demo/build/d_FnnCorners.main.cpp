
#include <utest/Utest.h>

int main(int ac, char **av)
{
	Utest driver(ac, av);


	extern Utest::Testcase train_corners1_from_random;
	driver.addcase(train_corners1_from_random, "train_corners1_from_random");

	extern Utest::Testcase train_corners1_xor;
	driver.addcase(train_corners1_xor, "train_corners1_xor");

	return driver.run();
}
