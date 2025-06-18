
#include <utest/Utest.h>

int main(int ac, char **av)
{
	Utest driver(ac, av);


	extern Utest::Testcase sortedIndexInt32;
	driver.addcase(sortedIndexInt32, "sortedIndexInt32");

	extern Utest::Testcase sortedIndexMultiInt32;
	driver.addcase(sortedIndexMultiInt32, "sortedIndexMultiInt32");

	extern Utest::Testcase sortedIndexSeq;
	driver.addcase(sortedIndexSeq, "sortedIndexSeq");

	return driver.run();
}
