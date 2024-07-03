/*************************************************************************
    > File Name: TRON.h
    > Description: Trust threshold Newton - method-truncated conjugate gradient method
    > Author: Guozheng Wang
    > Mail: gzh.wang@outlook.com
    > Created Time: 2020-10-14
 ************************************************************************/

#ifndef FIADMM__TRON_H
#define FIADMM__TRON_H

#include "include/continuous_functions.h"
#include "include/Vector.h"
#include "include/VectorOperations.h"
#include "include/Matrix.h"
#include "include/MatrixOperations.h"


namespace comlkit {
    Vector TRON(const ContinuousFunctions &c, std::vector<SparseFeature> &features, Vector &y, const Vector &x0,
                const int maxEval = 1000,
                const double TOL = 1e-3, const double cg_epsilon = 0.1, int verbosity = 1);
}
#endif //JFIADMM__TRON_H
