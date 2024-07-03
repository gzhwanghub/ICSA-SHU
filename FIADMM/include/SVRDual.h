// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
/*
 *	Gradient Descent for Unconstrained Convex Minimization with constant step size
        Solves the problem \min_x \phi(x), where \phi is a convex (or continuous) function.
        Author: Rishabh Iyer
 *
        Input:  Continuous Function: c
                        Initial starting point x0
                        step-size parameter (alpha)
                        max number of iterations (maxiter)
                        Tolerance (TOL)
                        Verbosity

        Output: Output on convergence (x)
 */

#ifndef FIADMM_SVR_Dual
#define FIADMM_SVR_Dual

#include "include/Vector.h"
#include "include/VectorOperations.h"
#include "include/SparseFeature.h"


namespace comlkit {

Vector SVRDual(std::vector<SparseFeature>& features, Vector& y, int solver_type, double lambda, double p, double eps, int max_iter, const int verbosity = 1);

}
#endif
