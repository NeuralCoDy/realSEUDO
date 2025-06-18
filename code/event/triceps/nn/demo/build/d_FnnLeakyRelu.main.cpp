
#include <utest/Utest.h>

int main(int ac, char **av)
{
	Utest driver(ac, av);


	extern Utest::Testcase train_leaky_relu_from_random;
	driver.addcase(train_leaky_relu_from_random, "train_leaky_relu_from_random");

	return driver.run();
}
