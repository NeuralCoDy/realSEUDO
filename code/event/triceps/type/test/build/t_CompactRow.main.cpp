
#include <utest/Utest.h>

int main(int ac, char **av)
{
	Utest driver(ac, av);


	extern Utest::Testcase rowtype;
	driver.addcase(rowtype, "rowtype");

	extern Utest::Testcase x_fields;
	driver.addcase(x_fields, "x_fields");

	extern Utest::Testcase x_data;
	driver.addcase(x_data, "x_data");

	extern Utest::Testcase parse_err;
	driver.addcase(parse_err, "parse_err");

	extern Utest::Testcase mkrow;
	driver.addcase(mkrow, "mkrow");

	extern Utest::Testcase mkrowshort;
	driver.addcase(mkrowshort, "mkrowshort");

	extern Utest::Testcase mkrowover;
	driver.addcase(mkrowover, "mkrowover");

	extern Utest::Testcase equal;
	driver.addcase(equal, "equal");

	extern Utest::Testcase hold_row_types;
	driver.addcase(hold_row_types, "hold_row_types");

	return driver.run();
}
