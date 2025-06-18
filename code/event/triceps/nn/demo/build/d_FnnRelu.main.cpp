
#include <utest/Utest.h>

int main(int ac, char **av)
{
	Utest driver(ac, av);


	extern Utest::Testcase train_relu_from_random;
	driver.addcase(train_relu_from_random, "train_relu_from_random");

	return driver.run();
}
