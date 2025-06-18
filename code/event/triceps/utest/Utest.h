//
// (C) Copyright 2011-2018 Sergey A. Babkin.
// This file is a part of Triceps.
// See the file COPYRIGHT for the copyright notice and license information
//
//
// Definition of the simple unit test infrastructure.

#ifndef __Triceps_Utest_h__
#define __Triceps_Utest_h__

#include <cmath>
#include <string>
#include <map>
#include <vector>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <common/Conf.h>
#include <common/Strprintf.h>
#include <common/Errors.h>

using namespace std;
using namespace TRICEPS_NS;
// NOT in TRICEPS_NS!!!

// this is for the script to detect test cases
#define UTESTCASE void

#define UT_FAIL() utest->fail(__FILE__, __LINE__)
#define UT_FAILMSG(msg) utest->fail(__FILE__, __LINE__, msg)
#define UT_ASSERT(expr) ((expr)? false : utest->fail(__FILE__, __LINE__, "Failed assertion: " #expr))
#define UT_IS(expr, cexpr) (((expr) == (cexpr))? false : \
	(utest->fail(__FILE__, __LINE__, "Failed assertion: " #expr " == " #cexpr), \
		(cout << "      Has value: \"" << (expr) << "\"" << endl << flush), true))
// For floating-point numbers, check that the values are near within eps.
#define UT_NEAR(expr, cexpr, eps) ((abs((expr) - (cexpr)) < (eps))? false : \
	(utest->fail(__FILE__, __LINE__, "Failed assertion: " #expr " near " #cexpr), \
		(cout << "      Has value: \"" << (expr) << "\"" << endl << flush), true))
#define UT_NOERROR(expr) utest->noerror(__FILE__, __LINE__, expr)

class Utest {
public:
	// Initialize with the command-line arguments
	Utest(int ac, char **av);

	// each test case is in a function of this type
	typedef void Testcase(Utest *utest);

	// Add a test case to the list.
	// These calls are normally done by the auto-generated main().
	// The test cases will run in the order they are registered.
	// @param fn - test case function
	// @param name - test case name
	void addcase(Testcase *fn, const char *name = 0);

	// Request a prerequisite test case to be executed.
	// This is a way to define a set of connected cases, where the
	// state left by one is used as the initialization for the next one.
	// If the prerequisite was just executed, or already failed, it won't be re-run.
	// Otherwise it will be re-run (even it was already executed earlier).
	// @param pre - the prerequisite case
	// @returm - false on success, true if that case failed: then
	//           the caller must immediately return and not try to 
	//           proceed further. It will be automatically marked as 
	//           skipped.
	bool require(Testcase *pre);

	// The current test case reporting that it has failed.
	// @param file - source file defining the test
	// @param line - line number in source file defining the test
	// @param msg - custom error message
	// @return - always true, to pass through the failure indication
	bool fail(const char *file, int line, string msg = "see the source code");

	// Check that the Erref contains no error.
	// @param file - source file defining the test
	// @param line - line number in source file defining the test
	// @param error - error object to check
	// @return - true if an error was detected.
	bool noerror(const char *file, int line, Erref error);
	
	// Get the error code.
	// @return - 0 if all tests succeeded, 1 if any failed
	int getrc()
	{
		return rc_;
	}

	// Run all the registered test cases (or not all, as selected by ac/av).
	// Not reentrant (for the same instance).
	// @return - 0 if all tests succeeded, 1 if any failed
	int run();

protected:
	int rc_; // return code, the summary result of all the cases
	Testcase *lastcase_; // last case that was run
	Testcase *curcase_; // the current running case

	enum CaseStatus {
		CS_OK,
		CS_FAIL,
		CS_SKIP,
		CS_UNKNOWN // not run yet
	};
	struct CaseInfo {
		CaseStatus status_; // status of executed case
		string name_;
	};
	typedef vector<Testcase *> TcV;
	TcV cases_; // all the registered cases, in order registered
	typedef map<Testcase *, CaseInfo> TcInfo;
	TcInfo info_; // information about the cases

private:
	Utest();
	Utest(const Utest &);
	void operator=(const Utest &);
}; 

#endif // __Triceps_Utest_h__
