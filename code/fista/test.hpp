// Simple test helpers.

#ifndef __TEST_HPP__
#define __TEST_HPP__

extern bool verbose;

class TestDriver
{
public:
	TestDriver(int argc, char **argv)
		: argc_(argc), argv_(argv)
	{
		verbose = argc > 1;
	}

	// Returns 0 on success, 1 on failure.
	typedef int TestFunction(const char *tname);

	// Run a test if it's requested by argc/argv.
	// If the test is not requested, returns false.
	//
	// The requesting is done by one of the follwing ways:
	//   * no arguments - run everything, not verbose
	//   * first argument is "-v" - run everything verbose
	//   * list the names of tests to run as arguments
	//
	// @param name - name of the test
	// @param function - function implementing the test.
	int runTest(const char *name, TestFunction *function);

protected:
	int argc_;
	char **argv_;
};

// Run a test, collecting the error indication.
#define RUN_TEST(errvar, driver, name) \
	(errvar |= driver.runTest(#name, name))

#endif // __TEST_HPP__
