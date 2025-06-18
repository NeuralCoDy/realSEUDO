
#include <utest/Utest.h>

int main(int ac, char **av)
{
	Utest driver(ac, av);


	extern Utest::Testcase warmup;
	driver.addcase(warmup, "warmup");

	extern Utest::Testcase readAlignedDirect;
	driver.addcase(readAlignedDirect, "readAlignedDirect");

	extern Utest::Testcase readAlignedMemcpy;
	driver.addcase(readAlignedMemcpy, "readAlignedMemcpy");

	extern Utest::Testcase readUnalignedMemcpy;
	driver.addcase(readUnalignedMemcpy, "readUnalignedMemcpy");

	extern Utest::Testcase readAlignedTemplate;
	driver.addcase(readAlignedTemplate, "readAlignedTemplate");

	extern Utest::Testcase readUnalignedTemplate;
	driver.addcase(readUnalignedTemplate, "readUnalignedTemplate");

	return driver.run();
}
