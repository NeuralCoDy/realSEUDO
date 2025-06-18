
#include <utest/Utest.h>

int main(int ac, char **av)
{
	Utest driver(ac, av);


	extern Utest::Testcase construct;
	driver.addcase(construct, "construct");

	extern Utest::Testcase construct_options;
	driver.addcase(construct_options, "construct_options");

	extern Utest::Testcase setGetWeights;
	driver.addcase(setGetWeights, "setGetWeights");

	extern Utest::Testcase computeForTraining;
	driver.addcase(computeForTraining, "computeForTraining");

	extern Utest::Testcase reclaim;
	driver.addcase(reclaim, "reclaim");

	return driver.run();
}
