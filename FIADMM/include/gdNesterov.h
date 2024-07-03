// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
/*


 *	Gradient Descent for Unconstrained Convex Minimization with Nesterov
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

#ifndef FIADMM_GD_NESTEROV
#define FIADMM_GD_NESTEROV

#include "include/continuous_functions.h"
#include "include/Vector.h"
#include "include/VectorOperations.h"

namespace comlkit {

    Vector gdNesterov(const ContinuousFunctions &c, const Vector &x0, double alpha = 1, const double gamma = 1e-4,
                      const int maxEval = 1000, const double TOL = 1e-3, bool resetAlpha = true,
                      bool useinputAlpha = false, int verbosity = 1);

}
#endif
