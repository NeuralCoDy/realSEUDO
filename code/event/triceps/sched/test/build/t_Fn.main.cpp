
#include <utest/Utest.h>

int main(int ac, char **av)
{
	Utest driver(ac, av);


	extern Utest::Testcase fn_return;
	driver.addcase(fn_return, "fn_return");

	extern Utest::Testcase fn_binding;
	driver.addcase(fn_binding, "fn_binding");

	extern Utest::Testcase call_bindings;
	driver.addcase(call_bindings, "call_bindings");

	extern Utest::Testcase tray_bindings;
	driver.addcase(tray_bindings, "tray_bindings");

	extern Utest::Testcase xtray;
	driver.addcase(xtray, "xtray");

	extern Utest::Testcase fn_binding_memory;
	driver.addcase(fn_binding_memory, "fn_binding_memory");

	extern Utest::Testcase fn_return_memory;
	driver.addcase(fn_return_memory, "fn_return_memory");

	return driver.run();
}
