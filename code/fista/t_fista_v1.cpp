// Test of FISTA code.

#include <stdio.h>
#include <math.h>
#include "fista_v1.hpp"
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
// than for g()).
//
// Then:
//   grad_f(x) = [
//     df/dx_0 = 2*(x_0 - 10),
//     df/dx_1 = 2*(x_1 - 10),
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
//   },
//   dp/dx_1 = {
//     x_1 >= 0: 1 + L/2 * ( -2*x_1 + 2*M_1 ) = -L*x_1 + L*M_1 + 1,
//     x_1 < 0: -1 + L/2 * ( -2*x_1 + 2*M_1 ) = -L*x_1 + L*M_1 - 1,
//   },
// ]
// for grad_p(x) = 0, we have 4 potential points:
//   x_0 = M_0 +- 1/L
//   x_1 = M_1 +- 1/L
//
// L = max( ||grad_f(x) - grad_f(y)|| / ||x - y|| ) = 2
// But then if we take this value of L, we will get:
//   M = y - (1/2)(2y - [10, 10]) = y - y + [5, 5] = [5, 5]
// and then potential x = [ 5 +- 1/2, 5 +- 1/2], and optimal x = [4.5, 4.5].
// Which is reasonable, since [10, 10] is the minimum of f and [0, 0] is the minimum of g,
// but it's not interesting to run multiple steps, since the optimim will always be
// reached in 1 step. Instead of that let's pick L = 10.
class SquareMinimizer1 : public Fista::LipschitzMinimizer
{
public:
	// Compute the value of minimized function at [m0 + adj0, m1 + adj1]
	static double minfunc(double m0, double adj0, double m1, double adj1)
	{
		static const double inv_L = 1./10.;
		return fabs(m0 + adj0) + fabs(m1 + adj1) - inv_L*( adj0*adj0 + adj1*adj1 );
	}

	bool compute(Fista::Vector &y, Fista::Vector &x, int step)
	{
		static const double inv_L = 1./10.;

		double m0 = y[0] - inv_L * 2.* (y[0] - 10.);
		double m1 = y[1] - inv_L * 2.* (y[1] - 10.);

		// Start with one alternative.
		x[0] = m0 - inv_L;
		x[1] = m1 - inv_L;
		double minval = minfunc(m0, -inv_L, m1, -inv_L);

		// Then check all other alternatives.
		double v;

		v = minfunc(m0, inv_L, m1, -inv_L);
		if (v < minval) {
			minval = v;
			x[0] = m0 + inv_L;
			x[1] = m1 - inv_L;
		}

		v = minfunc(m0, -inv_L, m1, inv_L);
		if (v < minval) {
			minval = v;
			x[0] = m0 - inv_L;
			x[1] = m1 + inv_L;
		}

		v = minfunc(m0, inv_L, m1, inv_L);
		if (v < minval) {
			// minval won't be used any more, so skip updating it
			x[0] = m0 + inv_L;
			x[1] = m1 + inv_L;
		}
		// printf("   p_L(%f, %f) -> (%f, %f); m=(%f, %f)\n", y[0], y[1], x[0], x[1], m0, m1);

		// True if length (norm) of the gradient is 0.1 or less.
		double grad0 = 2.* (x[0] - 10.) + (x[0] >= 0? 1 : -1);
		double grad1 = 2.* (x[1] - 10.) + (x[0] >= 0? 1 : -1);
		return (grad0*grad0 + grad1*grad1) <= (0.1*0.1);
	};
};

// Another version of the same minimizer but done slightly differently, in 2 ways:
//
// 1. In the last step, instead of looking at every point +-1/L in every dimension,
//    we can look at the gradient in every dimension separately, and use it to
//    decide the direction of the step, for a positive partial gradient, take a
//    negative step with -1/L, for a negative gradient use +1/L.
//
// 2. Since +-1/L represents a step by the gradient of g(x), and we do a step by
//    gradient of f(x) before that, we can merge both steps and do a single step
//    by the gradient of (f(x) + g(x)). It will be slightly different than in
//    the original version, because when computing the sign of the step by g(x)
//    the orignal version uses the value of x _after_ it has been stepped by
//    f(x), and the new version uses the same original value of x as the point
//    for computation of both parts. But it should be close enough. Moreover,
//    if we get into optimizing with condition x > 0 (as many problems are), the
//    part for g(x) will be constant, and it won't matter at all.
//
// So here is the new version:
// 
// Minimizer for 
//   f(x) = (x_0 - 10)^2 + (x_1 - 10)^2
//   g(x) = |x_0| + |x_1|
// (the offset in f() from (0, 0) is selected to have its minimizer different
// than for g()).
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
class SquareMinimizer2 : public Fista::LipschitzMinimizer
{
public:
	bool compute(Fista::Vector &y, Fista::Vector &x, int step)
	{
		static const double inv_L = 1./10.;

		// g0, g1 represent grad_fg()
		double g0 = 2.* (y[0] - 10.); // grad_f part
		if (y[0] >= 0.) // grad_g part
			g0 += 1.;
		else
			g0 -= 1.;
		x[0] = y[0] - inv_L * g0;

		double g1 = 2.* (y[1] - 10.); // grad_f part
		if (y[1] >= 0.) // grad_g part
			g1 += 1.;
		else
			g1 -= 1.;
		x[1] = y[1] - inv_L * g1;

		// printf("   p_L(%f, %f) -> (%f, %f); g=(%f, %f)\n", y[0], y[1], x[0], x[1], g0, g1);

		// True if length (norm) of the gradient is 0.1 or less.
		return (g0*g0 + g1*g1) <= (0.1*0.1);
	};
};

int testSteps(const char *tname)
{
	// Basic stepping with a simple minimizer.
	
	SquareMinimizer1 pl;
	
	const int n = 20;
	const int nrestart = 5;

	Fista::Vector x(2);
	x[0] = -2.;
	x[1] = -1.;
	Fista::Vector x_r(x); // runs side-by-side the second copy with restart

	// This is a copy of repeatedSteps() with printing.
	double t = 1., t_r = 1.;
	bool stop, stop_r;
	Fista::Vector x_next(x.size());
	stop = pl.compute(x, x_next, 0); // computes x_1 from x_0
	Fista::Vector x_next_r(x_next); // with restart
	verbose && printf("step %2d: (%f, %f) stop=%d\n", 1, x_next[0], x_next[1], stop);

	Fista::Vector y;
	// the loop goes from x_2 and up to restart point
	for (int i = 1; i < nrestart; i++) {
		stop = Fista::oneStep(pl, t, x, x_next, y, i, 0);
		stop_r = Fista::oneStep(pl, t_r, x_r, x_next_r, y, i, nrestart);

		verbose && printf("step %2d: (%f, %f) stop=%d | (%f, %f) stop=%d\n",
			i + 1, x_next[0], x_next[1], stop, x_next_r[0], x_next_r[1], stop_r);

		if (x_next_r[0] != x_next[0] || x_next_r[1] != x_next[1]) {
			printf("%s: premature effect from restart on step %d: (%f, %f) != (%f, %f)\n", tname,
				i + 1, x_next[0], x_next[1], x_next_r[0], x_next_r[1]);
			return 1;
		}
	}

	// the first restart step
	stop = Fista::oneStep(pl, t, x, x_next, y, nrestart, 0);
	stop_r = Fista::oneStep(pl, t_r, x_r, x_next_r, y, nrestart, nrestart);

	verbose && printf("step %2d: (%f, %f) stop=%d | (%f, %f) stop=%d\n",
		nrestart + 1, x_next[0], x_next[1], stop, x_next_r[0], x_next_r[1], stop_r);
	verbose && printf("t=%f | t=%f\n", t, t_r);

	// Compare the step sizes.
	double sz = sqrt( pow2(x_next[0] - x[0]) + pow2(x_next[1] - x[1]) );
	double sz_r = sqrt( pow2(x_next_r[0] - x_r[0]) + pow2(x_next_r[1] - x_r[1]) );

	verbose && printf("step size: %f | %f\n", sz, sz_r);
	if (sz_r >= sz) {
		printf("%s: Step on restart is too big!\n", tname);
		return 1;
	}
	if (t == t_r) {
		printf("%s: t unchanged on restart\n", tname);
		return 1;
	}

	for (int i = nrestart + 1; i < n; i++) {
		stop = Fista::oneStep(pl, t, x, x_next, y, i, 0);
		stop_r = Fista::oneStep(pl, t_r, x_r, x_next_r, y, i, nrestart);

		verbose && printf("step %2d: (%f, %f) stop=%d | (%f, %f) stop=%d\n",
			i + 1, x_next[0], x_next[1], stop, x_next_r[0], x_next_r[1], stop_r);
	}

	if (fabs(x_next[0] - 9.5) > 0.1
	|| fabs(x_next[1] - 9.5) > 0.1) {
		printf("%s: Too far from the expected result!\n", tname);
		return 1;
	}

	if (fabs(x_next_r[0] - 9.5) > 0.1
	|| fabs(x_next_r[1] - 9.5) > 0.1) {
		printf("%s: Too far from the expected result with restart!\n", tname);
		return 1;
	}

	// Do the whole iteration in one function.
	x[0] = -2.;
	x[1] = -1.;
	stop = Fista::repeatedSteps(pl, x, n);
	// it's really less than n steps because it stops as soon as stop=true
	verbose && printf("after %2d steps: (%f, %f) stop=%d\n", n, x[0], x[1], stop);

	if (fabs(x[0] - 9.5) > 0.1
	|| fabs(x[1] - 9.5) > 0.1) {
		printf("%s: Too far from the expected result!\n", tname);
		return 1;
	}
	if (!stop) {
		printf("%s: The stop condition not met!\n", tname);
		return 1;
	}

	// Do the whole iteration in one function with restart.
	x[0] = -2.;
	x[1] = -1.;
	stop = Fista::repeatedSteps(pl, x, n, nrestart);
	// it's really less than n steps because it stops as soon as stop=true
	verbose && printf("after %2d steps with restart: (%f, %f) stop=%d\n", n, x[0], x[1], stop);

	if (fabs(x[0] - 9.5) > 0.1
	|| fabs(x[1] - 9.5) > 0.1) {
		printf("%s: Too far from the expected result with restart!\n", tname);
		return 1;
	}
	if (!stop) {
		printf("%s: The stop condition not met with restart!\n", tname);
		return 1;
	}

	return 0;
}

int testSteps2(const char *tname)
{
	// Basic stepping with the second version of the minimizer, that
	// works through the gradient computation.
	// This allows to compare the implementations side-by-side.
	// The version 2 requires slightly more steps but each step is simpler.

	SquareMinimizer2 pl;
	
	Fista::Vector x(2);
	x[0] = -2.;
	x[1] = -1.;
	int n = 25;

	// This is a copy of repeatedSteps() with printing.
	double t = 1.;
	bool stop;
	Fista::Vector x_next(x.size());
	stop = pl.compute(x, x_next, 0); // computes x_1 from x_0
	verbose && printf("step %2d: (%f, %f) stop=%d\n", 1, x_next[0], x_next[1], stop);

	Fista::Vector y;
	// the loop goes from x_2 and up
	for (int i = 1; i < n; i++) {
		stop = Fista::oneStep(pl, t, x, x_next, y, i, 0);
		verbose && printf("step %2d: (%f, %f) stop=%d\n", i + 1, x_next[0], x_next[1], stop);
	}

	if (fabs(x_next[0] - 9.5) > 0.1
	|| fabs(x_next[1] - 9.5) > 0.1) {
		printf("%s: Too far from the expected result!\n", tname);
		return 1;
	}

	// Do the whole iteration in one function.
	x[0] = -2.;
	x[1] = -1.;
	stop = Fista::repeatedSteps(pl, x, n);
	verbose && printf("after %2d steps: (%f, %f) stop=%d\n", n, x[0], x[1], stop);

	if (fabs(x[0] - 9.5) > 0.1
	|| fabs(x[1] - 9.5) > 0.1) {
		printf("%s: Too far from the expected result!\n", tname);
		return 1;
	}
	if (!stop) {
		printf("%s: The stop condition not met!\n", tname);
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
	RUN_TEST(result, td, testSteps2);

	return result;
}
