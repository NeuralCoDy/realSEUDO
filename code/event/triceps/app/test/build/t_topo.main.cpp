
#include <utest/Utest.h>

int main(int ac, char **av)
{
	Utest driver(ac, av);


	extern Utest::Testcase mkgraph;
	driver.addcase(mkgraph, "mkgraph");

	extern Utest::Testcase reduce;
	driver.addcase(reduce, "reduce");

	extern Utest::Testcase check;
	driver.addcase(check, "check");

	extern Utest::Testcase reduce_check_graph;
	driver.addcase(reduce_check_graph, "reduce_check_graph");

	extern Utest::Testcase check_loops_diamond_horiz;
	driver.addcase(check_loops_diamond_horiz, "check_loops_diamond_horiz");

	extern Utest::Testcase check_loops_diamond_vert;
	driver.addcase(check_loops_diamond_vert, "check_loops_diamond_vert");

	extern Utest::Testcase check_loops_touching;
	driver.addcase(check_loops_touching, "check_loops_touching");

	extern Utest::Testcase check_loops_twigs;
	driver.addcase(check_loops_twigs, "check_loops_twigs");

	extern Utest::Testcase check_loops_twodir;
	driver.addcase(check_loops_twodir, "check_loops_twodir");

	extern Utest::Testcase check_ready;
	driver.addcase(check_ready, "check_ready");

	extern Utest::Testcase check_dead;
	driver.addcase(check_dead, "check_dead");

	return driver.run();
}
