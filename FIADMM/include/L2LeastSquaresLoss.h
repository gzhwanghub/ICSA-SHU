// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
/*
        Jensen: A Convex Optimization And Machine Learning ToolKit
 *	Least Squares Loss with L2 regularization
        Author: Rishabh Iyer
 *
 */

#ifndef FIADMM_L2_LEAST_SQUARES_LOSS_H
#define FIADMM_L2_LEAST_SQUARES_LOSS_H

#include "include/Vector.h"
#include "include/Matrix.h"
#include "include/VectorOperations.h"
#include "include/continuous_functions.h"

namespace comlkit {

    template<class Feature>
    class L2LeastSquaresLoss : public ContinuousFunctions {
    protected:
        std::vector<Feature> &features;                 // size of features is number of trainins examples (n)
        Vector &y;                 // size of y is number of training examples (n)
        double lambda;                 // regularization parameter
        double rho_;
        int num_neighbor_;
        Vector sum_msg_;
    public:
        L2LeastSquaresLoss(int numFeatures, std::vector<Feature> &features, Vector &y, Vector &sum_msg,
                           int num_neighbor, double lambda, double rho);

        L2LeastSquaresLoss(const L2LeastSquaresLoss &c);         // copy constructor

        ~L2LeastSquaresLoss();

        double eval(const Vector &x) const;                 // functionEval
        Vector evalGradient(const Vector &x) const;                 // gradientEval
        void
        eval(const Vector &x, double &f, Vector &gradient) const;                 // combined function and gradient eval
        Vector evalStochasticGradient(const Vector &x,
                                      std::vector<int> &miniBatch) const;                 // stochastic gradient
        void evalStochastic(const Vector &x, double &f, Vector &g,
                            std::vector<int> &miniBatch) const;                 // stochastic evaluation
    };

}
#endif
