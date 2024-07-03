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
                        back-tracking parameter (gamma)
                        max number of function evaluations (maxEvals)
                        Tolerance (TOL)
                        resetAlpha (whether to reset alpha at every iteration or not)
                        verbosity

        Output: Output on convergence (x)
 */

#ifndef FIADMM_TRON
#define FIADMM_TRON

#include "include/continuous_functions.h"
#include "include/Vector.h"
#include "include/VectorOperations.h"
#include "include/Matrix.h"
#include "include/MatrixOperations.h"

namespace comlkit {

    Vector tron(const ContinuousFunctions &c, int myid, const Vector &x0, const int maxEval = 50,
                const double TOL = 1e-3, int verbosity = 1);

}
#endif
