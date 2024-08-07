// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
/*


 *	Gradient Descent for Unconstrained Convex Minimization with Nesterov's method

        Solves the problem \min_x \phi(x), where \phi is a convex (or continuous) function.
        Author: Rishabh Iyer
 *
        Input:  Continuous Function: c
                        Initial starting point x0
                        Initial step-size (alpha)
                        back-tracking parameter (gamma)
                        max number of function evaluations (maxEvals)
                        Tolerance (TOL)
                        resetAlpha (whether to reset alpha at every iteration or not)
                        verbosity

        Output: Output on convergence (x)
 */

#include <stdio.h>
#include <algorithm>
#include <iostream>
using namespace std;

#include "include/gdNesterov.h"
#include "include/group_strategy.h"
#include "include/utils.h"
#include <math.h>
namespace comlkit {

Vector gdNesterov(const ContinuousFunctions& c, const Vector& x0, double alpha, const double gamma,
                  const int maxEval, const double TOL, bool resetAlpha, bool useinputAlpha, int verbosity){
	Vector x(x0);
	Vector g;
	double f;
	c.eval(x, f, g); // 更新f和g
	Vector y = x;
	Vector xnew;
	double fnew;
	Vector gnew;
	double t, tnew;
	double gnorm = norm(g);
	int funcEval = 1;
	if (!useinputAlpha)
		alpha = 1/norm(g);
	while ((gnorm >= TOL) && (funcEval < maxEval) )
	{
		if (funcEval > 1) {
			tnew = (1 + sqrt(1 + 4*t*t))/2;
			y = xnew + ((t - 1)/tnew)*(xnew - x);
			x = xnew;
			t = tnew;
			c.eval(y, f, g);
			funcEval++;
		}

		xnew = y - alpha*g; // 更新x
		c.eval(xnew, fnew, gnew);
		funcEval++;

		double gg = g*g;
		// double fgoal = f - gamma*alpha*gg;
		// Backtracking line search
		while (fnew > f - gamma*alpha*gg) {
			alpha = alpha*alpha*gg/(2*(fnew + gg*alpha - f));
			xnew = y - alpha*g;
			c.eval(xnew, fnew, gnew);
			funcEval++;
		}
		if (resetAlpha)
			alpha = min(1, 2*(f - fnew)/gg);

		gnorm = norm(gnew);
//        if(myid == 0){
//            if (verbosity > 0)
//                printf("numIter: %d, alpha: %e, ObjVal: %e, OptCond: %e\n", funcEval, alpha, fnew, gnorm);
//        }
	}
	return xnew;
}
}
