
#include <utest/Utest.h>

int main(int ac, char **av)
{
	Utest driver(ac, av);


	extern Utest::Testcase drain_reader;
	driver.addcase(drain_reader, "drain_reader");

	extern Utest::Testcase drain_writer;
	driver.addcase(drain_writer, "drain_writer");

	return driver.run();
}
