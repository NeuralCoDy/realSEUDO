
#include <utest/Utest.h>

int main(int ac, char **av)
{
	Utest driver(ac, av);


	extern Utest::Testcase make_facet;
	driver.addcase(make_facet, "make_facet");

	extern Utest::Testcase export_import;
	driver.addcase(export_import, "export_import");

	extern Utest::Testcase mknexus;
	driver.addcase(mknexus, "mknexus");

	extern Utest::Testcase import_queues;
	driver.addcase(import_queues, "import_queues");

	extern Utest::Testcase queue_fill;
	driver.addcase(queue_fill, "queue_fill");

	extern Utest::Testcase dynamic_add_del;
	driver.addcase(dynamic_add_del, "dynamic_add_del");

	extern Utest::Testcase pass_data;
	driver.addcase(pass_data, "pass_data");

	extern Utest::Testcase pass_begin_end;
	driver.addcase(pass_begin_end, "pass_begin_end");

	return driver.run();
}
