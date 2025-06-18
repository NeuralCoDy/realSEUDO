
#include <utest/Utest.h>

int main(int ac, char **av)
{
	Utest driver(ac, av);


	extern Utest::Testcase mkunit;
	driver.addcase(mkunit, "mkunit");

	extern Utest::Testcase mklabel;
	driver.addcase(mklabel, "mklabel");

	extern Utest::Testcase rowop;
	driver.addcase(rowop, "rowop");

	extern Utest::Testcase scheduling;
	driver.addcase(scheduling, "scheduling");

	extern Utest::Testcase chaining;
	driver.addcase(chaining, "chaining");

	extern Utest::Testcase clearing1;
	driver.addcase(clearing1, "clearing1");

	extern Utest::Testcase clearing2;
	driver.addcase(clearing2, "clearing2");

	extern Utest::Testcase frameMarks;
	driver.addcase(frameMarks, "frameMarks");

	extern Utest::Testcase markLoop;
	driver.addcase(markLoop, "markLoop");

	extern Utest::Testcase exceptions;
	driver.addcase(exceptions, "exceptions");

	extern Utest::Testcase label_exceptions;
	driver.addcase(label_exceptions, "label_exceptions");

	return driver.run();
}
