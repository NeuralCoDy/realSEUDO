
#include <utest/Utest.h>

int main(int ac, char **av)
{
	Utest driver(ac, av);


	extern Utest::Testcase statics;
	driver.addcase(statics, "statics");

	extern Utest::Testcase empty_is_ready;
	driver.addcase(empty_is_ready, "empty_is_ready");

	extern Utest::Testcase basic_trieads;
	driver.addcase(basic_trieads, "basic_trieads");

	extern Utest::Testcase basic_frags;
	driver.addcase(basic_frags, "basic_frags");

	extern Utest::Testcase basic_pthread_join;
	driver.addcase(basic_pthread_join, "basic_pthread_join");

	extern Utest::Testcase find_triead_success;
	driver.addcase(find_triead_success, "find_triead_success");

	extern Utest::Testcase find_triead_immed_fail;
	driver.addcase(find_triead_immed_fail, "find_triead_immed_fail");

	extern Utest::Testcase basic_abort;
	driver.addcase(basic_abort, "basic_abort");

	extern Utest::Testcase timeout_find;
	driver.addcase(timeout_find, "timeout_find");

	extern Utest::Testcase find_deadlock_catch_pthread;
	driver.addcase(find_deadlock_catch_pthread, "find_deadlock_catch_pthread");

	extern Utest::Testcase basic_pthread_assert;
	driver.addcase(basic_pthread_assert, "basic_pthread_assert");

	extern Utest::Testcase any_abort;
	driver.addcase(any_abort, "any_abort");

	extern Utest::Testcase define_join;
	driver.addcase(define_join, "define_join");

	extern Utest::Testcase join_throw;
	driver.addcase(join_throw, "join_throw");

	extern Utest::Testcase find_errors;
	driver.addcase(find_errors, "find_errors");

	extern Utest::Testcase store_fd;
	driver.addcase(store_fd, "store_fd");

	return driver.run();
}
