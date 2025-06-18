// Test of FISTA code.

#include <stdio.h>
#include <math.h>
#include "fista.hpp"
#include "test.hpp"

// for convenience
inline double pow2(double x)
{
	return x * x;
}

// Minimizer for 
//   f(x) = (x_0 - 10)^2 + (x_1 - 10)^2
//   g(x) = |x_0| + |x_1|
// (the offset in f() from (0, 0) is selected to have its minimizer different
// than for g()). See t_fista_v1.cpp for a more historic version, and how this one
// was developed from that one.
//
// Then the gradient for f(x):
//   grad_f(x) = [
//     df/dx_0 = 2*(x_0 - 10),
//     df/dx_1 = 2*(x_1 - 10),
//   ]
// The gradient for f(x) + g(x):
//   grad_fg(x) = [
//     df/dx_0 = 2*(x_0 - 10) + (x_0 >= 0? 1 : -1),
//     df/dx_1 = 2*(x_1 - 10) + (x_1 >= 0? 1 : -1),
//   ]
// Then if we denote M = (y - (1/L)grad_f(y)),
// p_L(M) = argmin(x){ |x_0| + |x_1| - L/2 * ( (x_0 - M_0)^2 + (x_1 - M_1)^2 ) }
//        = argmin(x){ |x_0| + |x_1| + L/2 * ( - x_0^2 + 2*M_0*x_0 - M_0^2 - x_1^2 + 2*M_1*x_1 - M_1^2 ) }
//        = argmin(x){ |x_0| + |x_1| + L/2 * ( - x_0^2 + 2*M_0*x_0 - x_1^2 + 2*M_1*x_1 ) }
// then to find the minimizer
// grad_p(x) = [
//   dp/dx_0 = {
//     x_0 >= 0: 1 + L/2 * ( -2*x_0 + 2*M_0 ) = -L*x_0 + L*M_0 + 1,
//     x_0 < 0: -1 + L/2 * ( -2*x_0 + 2*M_0 ) = -L*x_0 + L*M_0 - 1,
//   } = -L*x_0 + L*M_0 + (x_0 >= 0? 1 : -1),
//   dp/dx_1 = {
//     x_1 >= 0: 1 + L/2 * ( -2*x_1 + 2*M_1 ) = -L*x_1 + L*M_1 + 1,
//     x_1 < 0: -1 + L/2 * ( -2*x_1 + 2*M_1 ) = -L*x_1 + L*M_1 - 1,
//   } = -L*x_1 + L*M_1 + (x_1 >= 0? 1 : -1),
// ]
// for grad_p(x) = 0, we have the point
//   x_0 = M_0 - (x_0 >= 0? 1 : -1)/L
//       = y_0 - (1/L)grad_f(y_0) - (1/L)grad_g(y_0)
//       = y_0 - (1/L)grad_fg(y_0)
//   x_1 = M_1 - (x_1 >= 0? 1 : -1)/L
//       = y_1 - (1/L)grad_f(y_1) - (1/L)grad_g(y_1)
//       = y_1 - (1/L)grad_fg(y_1)
//
// L = max( ||grad_f(x) - grad_f(y)|| / ||x - y|| ) = 2
// But then if we take this value of L, we will get:
//   M = y - (1/2)(2y - [10, 10]) = y - y + [5, 5] = [5, 5]
// and then potential x = [ 5 +- 1/2, 5 +- 1/2], and optimal x = [4.5, 4.5].
// Which is reasonable, since [10, 10] is the minimum of f and [0, 0] is the minimum of g,
// but it's not interesting to run multiple steps, since the optimim will always be
// reached in 1 step. Instead of that let's pick L = 10.
class SquareGradient : public Fista::ScaledGradient
{
public:
	// from ScaledGradient
	virtual void computeGradient(const Fista::Vector &x, Fista::Vector &grad, int step)
	{
		static const double inv_L = 1./10.;

		for (int i = 0; i < 2; i++) {
			// g represents grad_fg()
			double g = 2.* (x[i] - 10.); // grad_f part
			if (x[i] >= 0.) // grad_g part
				g += 1.;
			else
				g -= 1.;
			grad[i] = inv_L * g;
		}

		// printf("   p_L(%f, %f) -> (1/L)g=(%f, %f)\n", x[0], x[1], grad[0], grad[1]);
	}
};

int testSteps(const char *tname)
{
	// Basic stepping and resetting.

	Fista::Vector init_x(2);
	init_x[0] = -2.;
	init_x[1] = -1.;
	int n = 25;

	// This eps is stricter than in the tests of v1, because it includes algorithm's step too.
	// eps is 0.01 instead of 0.1 because here the gradient dimensions get multiplied by (1/L) before
	// comparing them, which is (1/10).
	Fista::Run run(std::make_shared<SquareGradient>(), Fista::NoLimiter, init_x, /*diffEps*/ 0.01);
	
	// This is a copy of repeatedSteps() with printing.
	bool stop;

	for (int i = 0; i < n; i++) {
		stop = run.oneStep();
		verbose && printf("step %2d: (%f, %f) stop=%d\n", run.step_, run.x_[0], run.x_[1], stop);
	}

	if (fabs(run.x_[0] - 9.5) > 0.1
	|| fabs(run.x_[1] - 9.5) > 0.1) {
		printf("%s: Too far from the expected result!\n", tname);
		return 1;
	}

	// Do the whole iteration in one function.
	run.reset(init_x);
	int done_steps = run.repeatedSteps(n);
	verbose && printf("repeated, after %2d steps: (%f, %f)\n", done_steps, run.x_[0], run.x_[1]);

	if (fabs(run.x_[0] - 9.5) > 0.1
	|| fabs(run.x_[1] - 9.5) > 0.1) {
		printf("%s: Too far from the expected result!\n", tname);
		return 1;
	}
	if (done_steps > n) {
		printf("%s: The stop condition not met!\n", tname);
		return 1;
	}

	// Do the whole iteration again, from point (1, 1).
	run.reset(1.0);
	done_steps = run.repeatedSteps(n);
	verbose && printf("repeated, after %2d steps: (%f, %f)\n", done_steps, run.x_[0], run.x_[1]);

	if (fabs(run.x_[0] - 9.5) > 0.1
	|| fabs(run.x_[1] - 9.5) > 0.1) {
		printf("%s: Too far from the expected result!\n", tname);
		return 1;
	}
	if (done_steps > n) {
		printf("%s: The stop condition not met!\n", tname);
		return 1;
	}

	return 0;
}

int testStepsStopNorm(const char *tname)
{
	// Basic stepping, with stop on a norm condition.

	Fista::Vector init_x(2);
	init_x[0] = -2.;
	init_x[1] = -1.;
	int n = 25;

	// This eps is stricter than in the tests of v1, because it includes algorithm's step too.
	// eps is 0.01 instead of 0.1 because here the gradient dimensions get multiplied by (1/L) before
	// comparing them, which is (1/10).
	Fista::Run run(std::make_shared<SquareGradient>(), Fista::NoLimiter, init_x, /*diffEps*/ 0.01);
	run.stopping_ = Fista::Run::StopEpsNorm2;
	
	// This is a copy of repeatedSteps() with printing.
	bool stop;

	for (int i = 0; i < n; i++) {
		stop = run.oneStep();
		verbose && printf("step %2d: (%f, %f) stop=%d\n", run.step_, run.x_[0], run.x_[1], stop);
	}

	if (fabs(run.x_[0] - 9.5) > 0.1
	|| fabs(run.x_[1] - 9.5) > 0.1) {
		printf("%s: Too far from the expected result!\n", tname);
		return 1;
	}

	if (!stop) {
		printf("%s: The stop condition not met!\n", tname);
		return 1;
	}

	return 0;
}

int testStepsStopNormRel(const char *tname)
{
	// Basic stepping, with stop on a relative norm condition.

	Fista::Vector init_x(2);
	init_x[0] = -2.;
	init_x[1] = -1.;
	int n = 16; // stops earlier, because of larger tolerance at large X

	// This eps is stricter than in the tests of v1, because it includes algorithm's step too.
	// eps is 0.01 instead of 0.1 because here the gradient dimensions get multiplied by (1/L) before
	// comparing them, which is (1/10).
	Fista::Run run(std::make_shared<SquareGradient>(), Fista::NoLimiter, init_x, /*diffEps*/ 0.01);
	run.stopping_ = Fista::Run::StopEpsNorm2Rel;
	
	// This is a copy of repeatedSteps() with printing.
	bool stop;

	for (int i = 0; i < n; i++) {
		stop = run.oneStep();
		verbose && printf("step %2d: (%f, %f) stop=%d\n", run.step_, run.x_[0], run.x_[1], stop);
	}

	// Relative stopping gives less precision.
	if (fabs(run.x_[0] - 9.5) > 0.2
	|| fabs(run.x_[1] - 9.5) > 0.2) {
		printf("%s: Too far from the expected result!\n", tname);
		return 1;
	}

	if (!stop) {
		printf("%s: The stop condition not met!\n", tname);
		return 1;
	}

	return 0;
}

int testRestart(const char *tname)
{
	// Test the restart point.

	Fista::Vector init_x(2);
	init_x[0] = -2.;
	init_x[1] = -1.;
	int n = 25;
	const int nrestart = 5;

	// This eps is stricter than in the tests of v1, because it includes algorithm's step too.
	// eps is 0.01 instead of 0.1 because here the gradient dimensions get multiplied by (1/L) before
	// comparing them, which is (1/10).
	Fista::Run run(std::make_shared<SquareGradient>(), Fista::NoLimiter, init_x, /*diffEps*/ 0.01);

	// Run with restart.
	Fista::Run run_r(std::make_shared<SquareGradient>(), Fista::NoLimiter, init_x, /*diffEps*/ 0.01);
	run_r.setRestartEvery(nrestart);

	bool stop, stop_r;
	
	// Go to the restart point.
	for (int i = 0; i < nrestart; i++) {
		stop = run.oneStep();
		stop_r = run_r.oneStep();

		verbose && printf("step %2d: (%f, %f) stop=%d | (%f, %f) stop=%d\n",
			run.step_, run.x_[0], run.x_[1], stop, run_r.x_[0], run_r.x_[1], stop_r);

		if (run_r.x_[0] != run.x_[0] || run_r.x_[1] != run.x_[1]) {
			printf("%s: premature effect from restart on step %d: (%f, %f) != (%f, %f)\n", tname,
				run.step_, run.x_[0], run.x_[1], run_r.x_[0], run_r.x_[1]);
			return 1;
		}
	}

	// the first restart step
	stop = run.oneStep();
	stop_r = run_r.oneStep();

	verbose && printf("restart step %2d: (%f, %f) stop=%d | (%f, %f) stop=%d\n",
		run.step_, run.x_[0], run.x_[1], stop, run_r.x_[0], run_r.x_[1], stop_r);
	verbose && printf("t=%f | t=%f\n", run.t_, run_r.t_);

	// Compare the step sizes.
	double sz = sqrt( pow2(run.diff_[0]) + pow2(run.diff_[1]) );
	double sz_r = sqrt( pow2(run_r.diff_[0]) + pow2(run_r.diff_[1]) );

	verbose && printf("step size: %f | %f\n", sz, sz_r);
	if (sz_r >= sz) {
		printf("%s: Step on restart is too big!\n", tname);
		return 1;
	}
	if (run.t_ == run_r.t_) {
		printf("%s: t unchanged on restart\n", tname);
		return 1;
	}

	return 0;
}

int
main(int argc, char **argv)
{
	TestDriver td(argc, argv);
	int result = 0;

	RUN_TEST(result, td, testSteps);
	RUN_TEST(result, td, testStepsStopNorm);
	RUN_TEST(result, td, testStepsStopNormRel);
	RUN_TEST(result, td, testRestart);

	return result;
}
