// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
/*
   Jensen: A Convex Optimization And Machine Learning ToolKit
 *	Smooth SVM Loss with L2 regularization
   Author: Rishabh Iyer
 *
 */

#ifndef FIADMM_L2_SMOOTHSVM_LOSS_H
#define FIADMM_L2_SMOOTHSVM_LOSS_H

#include "include/Vector.h"
#include "include/Matrix.h"
#include "include/VectorOperations.h"
#include "include/continuous_functions.h"

namespace comlkit {

    template<class Feature>
    class L2SmoothSVMLoss : public ContinuousFunctions {
    protected:
        std::vector<Feature> &features;     // size of features is number of trainins examples (n)
        Vector &y;     // size of y is number of training examples (n)
        double lambda;
        int *I;
        Vector sum_msg_;
        Vector new_y_, new_z_;
        int num_neighbor_;
        double rho_;
        int admm_;
    public:
        L2SmoothSVMLoss(int numFeatures, std::vector<Feature>& features, Vector& y, Vector& new_y, Vector& new_z, double lambda, double rho, int admm);

        L2SmoothSVMLoss(int numFeatures, std::vector<Feature> &features, Vector &y, Vector &sum_msg, int num_neighbor,
                        double lambda, double rho, int admm);

        L2SmoothSVMLoss(const L2SmoothSVMLoss &c);     // copy constructor

        ~L2SmoothSVMLoss();

        double eval(const Vector &x) const;     // functionEval
        Vector evalGradient(const Vector &x) const;     // gradientEval
        void eval(const Vector &x, double &f, Vector &gradient) const;     // combined function and gradient eval
        void evalHessianVectorProduct(const Vector &x, const Vector &v,
                                      Vector &Hxv) const;     // evaluate a product between a hessian and a vector
        Vector evalStochasticGradient(const Vector &x, std::vector<int> &miniBatch) const;     // stochastic gradient
        void evalStochastic(const Vector &x, double &f, Vector &g,
                            std::vector<int> &miniBatch) const;     // stochastic evaluation
        mutable int sizeI;
    };
}
#endif
