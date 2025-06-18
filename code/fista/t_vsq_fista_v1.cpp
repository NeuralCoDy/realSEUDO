// Test of FISTA code.

#include <stdio.h>
#include <math.h>
#include "fista_v1.hpp"
#include "test.hpp"

// Since the vector multiplication A*x can be represented ultimately as
//   sum( k_i * x_i)
// try to represent a simple case for x = [ x_0, x_1 ]
//
// Minimizer for 
//   f(x) = (k_0*x_0 - b0)^2 + (k_1*x_1 - b_1)^2
//   g(x) = la_0*|x_0| + la_1*|x_1|
//
// Then:
//   grad_f(x) = [
//     df/dx_0 = 2*k_0*(k_0*x_0 - b_0),
//     df/dx_1 = 2*k_1*(k_1*x_1 - b_1),
//   ]
// Then if we denote M = (y - (1/L)grad_f(y)),
// p_L(M) = argmin(x){ la_0*|x_0| + la_1*|x_1| - L/2 * ( (x_0 - M_0)^2 + (x_1 - M_1)^2 ) }
//        = argmin(x){ la_0*|x_0| + la_1*|x_1| + L/2 * ( - x_0^2 + 2*M_0*x_0 - M_0^2 - x_1^2 + 2*M_1*x_1 - M_1^2 ) }
//        = argmin(x){ la_0*|x_0| + la_1*|x_1| + L/2 * ( - x_0^2 + 2*M_0*x_0 - x_1^2 + 2*M_1*x_1 ) }
// then to find the minimizer
// grad_p(x) = [
//   dp/dx_0 = {
//     x_0 >= 0: la_0 + L/2 * ( -2*x_0 + 2*M_0 ) = -L*x_0 + L*M_0 + la_0,
//     x_0 < 0: -la_0 + L/2 * ( -2*x_0 + 2*M_0 ) = -L*x_0 + L*M_0 - la_0,
//   },
//   dp/dx_1 = {
//     x_1 >= 0: la_1 + L/2 * ( -2*x_1 + 2*M_1 ) = -L*x_1 + L*M_1 + la_1,
//     x_1 < 0: -la_1 + L/2 * ( -2*x_1 + 2*M_1 ) = -L*x_1 + L*M_1 - la_1,
//   },
// ]
// for grad_p(x) = 0, we have 4 potential points:
//   x_0 = M_0 +- la_0/L
//   x_1 = M_1 +- la_1/L
// and we can select each dimension independently based on the value of
// that dimension in grad_f(x): if grad_f_i(x) >= 0, select -(la_i/L),
// otherwise +(la_i/L).
//
// L = max( ||grad_f(x) - grad_f(y)|| / ||x - y|| ) = 2*max(k_0^2, k_1^2)
//
// This gets easily generalized to N dimensions.
//
// And also limit x >= 0, since this is assumed in SEUDO.
class PosVecSquareMinimizer : public Fista::LipschitzMinimizer
{
public:
	// See the meaning of arguments in the comment to the class.
	// The sizes of all vectors must be equal.
	// @param k - coefficients for x
	// @param b - target values
	// @param lambda - LASSO coefficients
	PosVecSquareMinimizer(
		const Fista::Vector &k,
		const Fista::Vector &b,
		const Fista::Vector &lambda)
		: k_(k), b_(b), lambda_(lambda)
	{
		double l = 0;
		for (size_t i = 0; i < k.size(); i++) {
			double v = k[i] * k[i];
			if (v > l)
				l = v;
		}
		if (l < 0.00001) // avoid division by 0
			l = 0.00001;
		// 1 / (2*L)
		inv_L_ = 0.5 / l;
	}

	bool compute(Fista::Vector &y, Fista::Vector &x, int step)
	{
		double gradnorm = 0.;
		for (size_t i = 0; i < k_.size(); i++) {
			// The tentative x[i]
			double v = y[i] - 2. * inv_L_ * k_[i] * (k_[i] * y[i] - b_[i]);
			// Compute df/dx in this dimension, to determine the LASSO adjustment
			double df = 2. * k_[i] * (k_[i] * v - b_[i]);
			// Fold dg/dx into it too. 
			// Since we assume always x>0, we could keep only one alternative,
			// but do it generically for now to allow negative experiments.
			if (v >= 0.) {
				df += lambda_[i];
			} else {
				df -= lambda_[i];
			}
			// printf("   p_L[%zd] (%f -> draft %f) df=%f\n", i, y[i], v, df);

			// Now shift using the derivative.
			if (df >= 0) {
				v -= lambda_[i] * inv_L_;
			} else {
				v += lambda_[i] * inv_L_;
			}
			// printf("   p_L[%zd] (%f -> %f)\n", i, y[i], v);

			// Compute this dimension of the gradient norm.
			// The computation of the partial derivative is the same as for df
			// but at the new point.
			double gd = 2 * k_[i] * (k_[i] * v - b_[i]) + (v >= 0.? lambda_[i] : -lambda_[i]);
			if (v <= 0. && gd > 0.)  {
				// If we've hit the boundary of x>0 and the gradient points
				// further downwards, we can't go there, so consider this
				// equivalent to gradient dimension becoming 0.
				gd = 0.;
			}
			gradnorm += gd*gd;

			// And can also enforce x >= 0.
			if (v < 0.) v = 0.;
			x[i] = v;
		}
		// True if length (norm) of the gradient is 0.1 or less.
		return gradnorm <= (0.1*0.1);
	};
protected:
	// Value of 1/L
	double inv_L_;
	Fista::Vector k_;
	Fista::Vector b_;
	Fista::Vector lambda_;
};

int testVector(const char *tname)
{
	// Minimization driven by a vector of dependencies.

	Fista::Vector k(2);
	k[0] = 1.;
	k[1] = 1.;
	Fista::Vector b(2);
	b[0] = 10.;
	b[1] = 10.;
	Fista::Vector lambda(2);
	lambda[0] = 1.;
	lambda[1] = 1.;

	PosVecSquareMinimizer pl(k, b, lambda);
	
	Fista::Vector x(2);
	x[0] = -2.;
	x[1] = -1.;
	int n = 20;

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
	|| fabs(x_next[0] - 9.5) > 0.1) {
		printf("%s: Too far from the expected result!\n", tname);
		return 1;
	}

	// Do the whole iteration in one function.
	x[0] = 2.;
	x[1] = 1.;
	stop = Fista::repeatedSteps(pl, x, n);
	verbose && printf("after %2d steps: (%f, %f) stop=%d\n", n, x[0], x[1], stop);

	if (fabs(x[0] - 9.5) > 0.1
	|| fabs(x[0] - 9.5) > 0.1) {
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

	RUN_TEST(result, td, testVector);

	return result;
}
