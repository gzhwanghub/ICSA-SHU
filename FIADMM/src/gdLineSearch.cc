// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
/*


 *	Gradient Descent for Unconstrained Convex Minimization with backtracking line search
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

#include "include/gdLineSearch.h"
#include "include/group_strategy.h"
#include "include/utils.h"
namespace comlkit {

Vector gdLineSearch(const ContinuousFunctions& c, const Vector& x0, double alpha, const double gamma,
                    const int maxEval, const double TOL, bool resetAlpha, bool useinputAlpha, int verbosity){
	Vector x(x0);
	Vector g;
	double f;
	c.eval(x, f, g); // 更新f和g
	Vector xnew;
	double fnew;
	Vector gnew;
	double gnorm = norm(g);
	int funcEval = 1;
	if (!useinputAlpha)
		alpha = 1/norm(g);
	while ((gnorm >= TOL) && (funcEval < maxEval) )
	{
		multiplyAccumulate(xnew, x, alpha, g); // 更新x
		c.eval(xnew, fnew, gnew); // 更新new_f, new_g
		funcEval++;

		double gg = g*g; // 梯度向量相乘，有点二范数的意思
		// double fgoal = f - gamma*alpha*gg;
		// Backtracking line search
		while (fnew > f - gamma*alpha*gg) { //充分下降条件
			alpha = alpha*alpha*gg/(2*(fnew + gg*alpha - f));
			multiplyAccumulate(xnew, x, alpha, g); // 更新x
			c.eval(xnew, fnew, gnew); // 更新f和g
			funcEval++;
		}
		if (resetAlpha)
			alpha = min(1, 2*(f - fnew)/gg);

		x = xnew;
		f = fnew;
		g = gnew;
		gnorm = norm(g);
//		if (verbosity > 0)
//			printf("numIter: %d, alpha: %e, ObjVal: %e, OptCond: %e\n", funcEval, alpha, f, gnorm);
	}
	return x;
}
}
