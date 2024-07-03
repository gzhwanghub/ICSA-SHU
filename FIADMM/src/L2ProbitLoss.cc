// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
/*
        Jensen: A Convex Optimization And Machine Learning ToolKit
 *	Probit Loss with L2 regularization
        Author: Rishabh Iyer
 *
 */

#include <iostream>
#include <math.h>

using namespace std;

#include "include/L2ProbitLoss.h"
#include "include/VectorOperations.h"
#include <assert.h>

#define EPSILON 1e-16
#define MAX 1e2
#define _USE_MATH_DEFINES

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace comlkit {
    template<class Feature>
    L2ProbitLoss<Feature>::L2ProbitLoss(int m, std::vector<Feature> &features, Vector &y, Vector &new_y, Vector &new_z,
                                        double lambda, double rho, int admm) :
            ContinuousFunctions(true, m, features.size()), features(features), y(y), new_y_(new_y), new_z_(new_z),
            lambda(lambda), rho_(rho), admm_(admm) {
        if (n > 0)
            assert(features[0].numFeatures == m);
        assert(features.size() == y.size());
    }

    template<class Feature>
    L2ProbitLoss<Feature>::L2ProbitLoss(int m, std::vector<Feature> &features, Vector &y, Vector &sum_msg,
                                        int num_neighbor, double lambda, double rho, int admm) :
            ContinuousFunctions(true, m, features.size()), features(features), y(y), sum_msg_(sum_msg),
            num_neighbor_(num_neighbor), lambda(lambda), rho_(rho), admm_(admm) {
        if (n > 0)
            assert(features[0].numFeatures == m);
        assert(features.size() == y.size());
    }

    template<class Feature>
    L2ProbitLoss<Feature>::L2ProbitLoss(const L2ProbitLoss &l) :
            ContinuousFunctions(true, l.m, l.n), features(l.features), y(l.y), lambda(l.lambda), sum_msg_(l.sum_msg_) {
    }

    template<class Feature>
    L2ProbitLoss<Feature>::~L2ProbitLoss() {
    }

    template<class Feature>
    double L2ProbitLoss<Feature>::eval(const Vector &x) const {
        assert(x.size() == m);
        double sum = 0.5 * lambda * (x * x);
        for (int i = 0; i < n; i++) {
            double val = y[i] * (x * features[i]) / sqrt(2);
            double probitval = (1 / 2) * (1 + erf(val)) + EPSILON;
            sum -= log(probitval);
        }
        return sum;
    }

    template<class Feature>
    Vector L2ProbitLoss<Feature>::evalGradient(const Vector &x) const {
        assert(x.size() == m);
        Vector g = lambda * x;
        for (int i = 0; i < n; i++) {
            double val = y[i] * (x * features[i]) / sqrt(2);
            double normval = (1 / sqrt(2 * M_PI)) * exp(-(val * val));
            double probitval = (1 / 2) * (1 + erf(val)) + EPSILON;
            g -= features[i] * (y[i] * normval / probitval);
        }
        return g;
    }

    template<class Feature>
    void L2ProbitLoss<Feature>::eval(const Vector &x, double &f, Vector &g) const {
	assert(x.size() == m);
//	g = lambda*x;
//	f = 0.5*lambda*(x*x);
        if (admm_ == 1) {
            // decentral admm
            f = 0;
            g = 2 * rho_ * num_neighbor_ * x + sum_msg_;
            for (int i = 0; i < n; i++) {
                double val = y[i] * (x * features[i]) / sqrt(2);
                double normval = (1 / sqrt(2 * M_PI)) * exp(-(val * val));
                double probitval = 0.5 * (1 + erf(val)) + EPSILON;
                g -= features[i] * (y[i] * normval / probitval);
                f -= log(probitval);
            }
            f += rho_ * num_neighbor_ * x * x + x * sum_msg_;
            return;
        } else if (admm_ == 2) {
            // global admm
            f = 0.5 * lambda * (new_z_ * new_z_);
            g = rho_ * (x - new_z_) + new_y_; // problem
            for (int i = 0; i < n; i++) {
                double val = y[i] * (x * features[i]) / sqrt(2);
                double normval = (1 / sqrt(2 * M_PI)) * exp(-(val * val));
                double probitval = 0.5 * (1 + erf(val)) + EPSILON;
                g -= features[i] * (y[i] * normval / probitval);
                f -= log(probitval);
            }
            f += (rho_ / 2) * (x - new_z_ + (1 / rho_) * new_y_) * (x - new_z_ + (1 / rho_) * new_y_);
            return;
        } else if (admm_ == 3) {
            // general admm
            f = lambda * norm(new_z_, 1);
            g = rho_ * (x - new_z_) + new_y_;
            for (int i = 0; i < n; i++) {
                double val = y[i] * (x * features[i]) / sqrt(2);
                double normval = (1 / sqrt(2 * M_PI)) * exp(-(val * val));
                double probitval = 0.5 * (1 + erf(val)) + EPSILON;
                g -= features[i] * (y[i] * normval / probitval);
                f -= log(probitval);
            }
            f += (rho_ / 2) * (x - new_z_ + (1 / rho_) * new_y_) * (x - new_z_ + (1 / rho_) * new_y_);
            return;
        }
    }

    template<class Feature>
    Vector L2ProbitLoss<Feature>::evalStochasticGradient(const Vector &x, std::vector<int> &miniBatch) const {
        assert(x.size() == m);
        Vector g = lambda * x;
        double val;
        for (vector<int>::iterator it = miniBatch.begin(); it != miniBatch.end(); it++) {
            double val = y[*it] * (x * features[*it]) / sqrt(2);
            double normval = (1 / sqrt(2 * M_PI)) * exp(-(val * val));
            double probitval = (1 / 2) * (1 + erf(val)) + EPSILON;
            g -= features[*it] * (y[*it] * normval / probitval);
        }
        return g;
    }

    template<class Feature>
    void
    L2ProbitLoss<Feature>::evalStochastic(const Vector &x, double &f, Vector &g, std::vector<int> &miniBatch) const {
        assert(x.size() == m);
        if (admm_ == 1) {
            // decentral admm
            f = 0;
            g = 2 * rho_ * num_neighbor_ * x + sum_msg_;
            for (vector<int>::iterator it = miniBatch.begin(); it != miniBatch.end(); it++) {
                double val = y[*it] * (x * features[*it]) / sqrt(2);
                double normval = (1 / sqrt(2 * M_PI)) * exp(-(val * val));
                double probitval = (1 / 2) * (1 + erf(val)) + EPSILON;
                g -= features[*it] * (y[*it] * normval / probitval);
                f -= log(probitval);
            }
            f += rho_ * num_neighbor_ * x * x + x * sum_msg_;
            return;
        } else if (admm_ == 2) {
            // global admm
            f = 0.5 * lambda * (new_z_ * new_z_);
            g = rho_ * (x - new_z_) + new_y_;
            for (vector<int>::iterator it = miniBatch.begin(); it != miniBatch.end(); it++) {
                double val = y[*it] * (x * features[*it]) / sqrt(2);
                double normval = (1 / sqrt(2 * M_PI)) * exp(-(val * val));
                double probitval = (1 / 2) * (1 + erf(val)) + EPSILON;
                g -= features[*it] * (y[*it] * normval / probitval);
                f -= log(probitval);
            }
            f += (rho_ / 2) * (x - new_z_ + (1 / rho_) * new_y_) * (x - new_z_ + (1 / rho_) * new_y_);
            return;
        } else if (admm_ == 3) {
            // general admm
            f = lambda * norm(new_z_, 1);
            g = rho_ * (x - new_z_) + new_y_;
            for (vector<int>::iterator it = miniBatch.begin(); it != miniBatch.end(); it++) {
                double val = y[*it] * (x * features[*it]) / sqrt(2);
                double normval = (1 / sqrt(2 * M_PI)) * exp(-(val * val));
                double probitval = (1 / 2) * (1 + erf(val)) + EPSILON;
                g -= features[*it] * (y[*it] * normval / probitval);
                f -= log(probitval);
            }
            f += (rho_ / 2) * (x - new_z_ + (1 / rho_) * new_y_) * (x - new_z_ + (1 / rho_) * new_y_);
            return;
        }
    }

    template
    class L2ProbitLoss<SparseFeature>;

    template
    class L2ProbitLoss<DenseFeature>;


}
