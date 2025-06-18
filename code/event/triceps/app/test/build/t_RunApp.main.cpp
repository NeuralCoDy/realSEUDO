
#include <utest/Utest.h>

int main(int ac, char **av)
{
	Utest driver(ac, av);


	extern Utest::Testcase create_basics_die;
	driver.addcase(create_basics_die, "create_basics_die");

	extern Utest::Testcase shutdown_loop;
	driver.addcase(shutdown_loop, "shutdown_loop");

	extern Utest::Testcase drain_loop;
	driver.addcase(drain_loop, "drain_loop");

	extern Utest::Testcase drain_unready;
	driver.addcase(drain_unready, "drain_unready");

	extern Utest::Testcase drain_frag;
	driver.addcase(drain_frag, "drain_frag");

	extern Utest::Testcase drain_after_frag;
	driver.addcase(drain_after_frag, "drain_after_frag");

	extern Utest::Testcase drain_except;
	driver.addcase(drain_except, "drain_except");

	extern Utest::Testcase drain_parallel;
	driver.addcase(drain_parallel, "drain_parallel");

	extern Utest::Testcase interrupt_fd_open;
	driver.addcase(interrupt_fd_open, "interrupt_fd_open");

	extern Utest::Testcase interrupt_fd_loop;
	driver.addcase(interrupt_fd_loop, "interrupt_fd_loop");

	extern Utest::Testcase interrupt_fd_close;
	driver.addcase(interrupt_fd_close, "interrupt_fd_close");

	extern Utest::Testcase shutdown_on_abort;
	driver.addcase(shutdown_on_abort, "shutdown_on_abort");

	extern Utest::Testcase shutdown_disconnects;
	driver.addcase(shutdown_disconnects, "shutdown_disconnects");

	extern Utest::Testcase flush_after_abort;
	driver.addcase(flush_after_abort, "flush_after_abort");

	extern Utest::Testcase nextXtray_timeout;
	driver.addcase(nextXtray_timeout, "nextXtray_timeout");

	extern Utest::Testcase interrupt_once;
	driver.addcase(interrupt_once, "interrupt_once");

	extern Utest::Testcase dispose_once;
	driver.addcase(dispose_once, "dispose_once");

	extern Utest::Testcase drain_frame_simple;
	driver.addcase(drain_frame_simple, "drain_frame_simple");

	extern Utest::Testcase drain_frame;
	driver.addcase(drain_frame, "drain_frame");

	return driver.run();
}
