// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
/*
        Author: Rishabh Iyer
 *
 */

#include <iostream>
#include <math.h>

using namespace std;

#include "include/L2LogisticLoss.h"
#include "../include/VectorOperations.h"
#include <assert.h>

#define EPSILON 1e-6
#define MAX 1e2
namespace comlkit {

    void UpdateHessianVectorProd(vector <SparseFeature> &features, Vector &Hxv, Vector &w, int n) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < features[i].featureIndex.size(); j++) {
                Hxv[features[i].featureIndex[j]] += w[i] * features[i].featureVec[j];
            }
        }
    }

    void UpdateHessianVectorProd(vector <DenseFeature> &features, Vector &Hxv, Vector &w, int n) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < features[i].featureVec.size(); j++) {
                Hxv[j] += w[i] * features[i].featureVec[j];
            }
        }
    }

    template<class Feature>
    L2LogisticLoss<Feature>::L2LogisticLoss(int m, std::vector<Feature> &features, Vector &y, double lambda) :
            ContinuousFunctions(true, m, features.size()), features(features), y(y), lambda(lambda) {
        if (n > 0)
            assert(features[0].numFeatures == m);
        assert(features.size() == y.size());
    }

    template<class Feature>
    L2LogisticLoss<Feature>::L2LogisticLoss(int m, std::vector<Feature> &features, Vector &y, Vector &new_y,
                                            Vector &new_z, double lambda, double rho, int admm) :
            ContinuousFunctions(true, m, features.size()), features(features), y(y), new_y_(new_y), new_z_(new_z),
            lambda(lambda), rho_(rho), admm_(admm) {
        if (n > 0)
            assert(features[0].numFeatures == m);
        assert(features.size() == y.size());
    }

    template<class Feature>
    L2LogisticLoss<Feature>::L2LogisticLoss(int m, std::vector<Feature> &features, Vector &y, Vector &sum_msg,
                                            int num_neighbor, double lambda, double rho, int admm) :
            ContinuousFunctions(true, m, features.size()), features(features), y(y), sum_msg_(sum_msg),
            num_neighbor_(num_neighbor), lambda(lambda), rho_(rho), admm_(admm) {
        if (n > 0)
            assert(features[0].numFeatures == m);
        assert(features.size() == y.size());
    }

    template<class Feature>
    L2LogisticLoss<Feature>::L2LogisticLoss(const L2LogisticLoss &l) :
            ContinuousFunctions(true, l.m, l.n), features(l.features), y(l.y), lambda(l.lambda) {
    }

    template<class Feature>
    L2LogisticLoss<Feature>::~L2LogisticLoss() {
    }

    template<class Feature>
    double L2LogisticLoss<Feature>::eval(const Vector &x) const {
        assert(x.size() == m);
        double sum = 0.5 * lambda * (x * x);;
        for (int i = 0; i < n; i++) {
            double preval = y[i] * (x * features[i]);
            if (preval > MAX)
                continue;
            else if (preval < -1 * MAX)
                sum += preval;
            else
                sum += log(1 + exp(-preval));
        }
        return sum;
    }

    template<class Feature>
    Vector L2LogisticLoss<Feature>::evalGradient(const Vector &x) const {
        assert(x.size() == m);
        Vector g = lambda * x;
        for (int i = 0; i < n; i++) {
            double preval = y[i] * (x * features[i]);
            if (preval > MAX)
                g -= y[i] * features[i];
            else if (preval < -1 * MAX)
                continue;
            else
                g -= (y[i] / (1 + exp(-preval))) * features[i];
        }
        return g;
    }

    template<class Feature>
    void L2LogisticLoss<Feature>::eval(const Vector &x, double &f, Vector &g) const {
        assert(x.size() == m);
//	g = lambda*x;
//	f = 0.5*lambda*(x*x);
        if (admm_ == 1) {
            // decentral ADMM
            f = 0;
            g = 2 * rho_ * num_neighbor_ * x + sum_msg_;
            double val;
            for (int i = 0; i < n; i++) {
                double preval = y[i] * (x * features[i]);
                if (preval > MAX) {
                    g -= (y[i] / (1 + exp(preval))) * features[i];
                } else if (preval < -1 * MAX) {
                    g -= (y[i] / (1 + exp(preval))) * features[i];
                    f -= preval;
                } else {
                    f += log(1 + exp(-preval));
                    g -= (y[i] / (1 + exp(preval))) * features[i];
                }
            }
            f += rho_ * num_neighbor_ * x * x + x * sum_msg_;
            return;
        } else if (admm_ == 2) {
            //     global ADMM
            f = 0.5 * lambda * (new_z_ * new_z_);
            g = rho_ * (x - new_z_) + new_y_;
            double val;
            for (int i = 0; i < n; i++) {
                double preval = y[i] * (x * features[i]);
                if (preval > MAX) {
                    g -= (y[i] / (1 + exp(preval))) * features[i];
                } else if (preval < -1 * MAX) {
                    g -= (y[i] / (1 + exp(preval))) * features[i];
                    f -= preval;
                } else {
                    f += log(1 + exp(-preval));
                    g -= (y[i] / (1 + exp(preval))) * features[i];
                }
            }
            f += (rho_ / 2) * (x - new_z_ + (1 / rho_) * new_y_) * (x - new_z_ + (1 / rho_) * new_y_);
            return;
        } else if (admm_ == 3) {
            // general ADMM
            f = lambda * norm(new_z_, 1);
            g = rho_ * (x - new_z_) + new_y_;
            double val;
            for (int i = 0; i < n; i++) {
                double preval = y[i] * (x * features[i]);
                if (preval > MAX) {
                    g -= (y[i] / (1 + exp(preval))) * features[i];
                } else if (preval < -1 * MAX) {
                    g -= (y[i] / (1 + exp(preval))) * features[i];
                    f -= preval;
                } else {
                    f += log(1 + exp(-preval));
                    g -= (y[i] / (1 + exp(preval))) * features[i];
                }
            }
            f += (rho_ / 2) * (x - new_z_ + (1 / rho_) * new_y_) * (x - new_z_ + (1 / rho_) * new_y_);
            return;
        }


    }

    template<class Feature>
    void L2LogisticLoss<Feature>::evalHessianVectorProduct(const Vector &x, const Vector &v,
                                                           Vector &Hxv) const {  // evaluate a product between a hessian and a vector
        Vector w(n, 0);
        Hxv = Vector(m, 0);
        for (int i = 0; i < n; i++) {
            double val = y[i] * (x * features[i]);
            double z = 1 / (1 + exp(-val));
            double D = z * (1 - z);
            w[i] = D * (v * features[i]);
        }
        UpdateHessianVectorProd(features, Hxv, w, n);
        Hxv += v;
    }

    template<class Feature>
    Vector L2LogisticLoss<Feature>::evalStochasticGradient(const Vector &x, std::vector<int> &miniBatch) const {
        assert(x.size() == m);
        Vector g = lambda * x;
        for (vector<int>::iterator it = miniBatch.begin(); it != miniBatch.end(); it++) {
            double preval = y[*it] * (x * features[*it]);
            if (preval > MAX)
                g -= y[*it] * features[*it];
            else if (preval < -1 * MAX)
                continue;
            else
                g -= (y[*it] / (1 + exp(-preval))) * features[*it];
        }
        return g;
    }

    template<class Feature>
    void
    L2LogisticLoss<Feature>::evalStochastic(const Vector &x, double &f, Vector &g, std::vector<int> &miniBatch) const {
        assert(x.size() == m);
        if (admm_ == 1) {
            // decentral admm
            f = 0;
            g = 2 * rho_ * num_neighbor_ * x + sum_msg_;
            double val;
            for (vector<int>::iterator it = miniBatch.begin(); it != miniBatch.end(); it++) {
                double preval = y[*it] * (x * features[*it]);
                if (preval > MAX) {
                    g -= (y[*it] / (1 + exp(preval))) * features[*it];
                } else if (preval < -1 * MAX) {
                    g -= (y[*it] / (1 + exp(preval))) * features[*it];
                    f -= preval;
                } else {
                    f += log(1 + exp(-preval));
                    g -= (y[*it] / (1 + exp(preval))) * features[*it];
                }
            }
            f += rho_ * num_neighbor_ * x * x + x * sum_msg_;
            return;
        } else if (admm_ == 2) {
            // global admm
            f = 0.5 * lambda * (new_z_ * new_z_);
            g = rho_ * (x - new_z_) + new_y_;
            double val;
            for (vector<int>::iterator it = miniBatch.begin(); it != miniBatch.end(); it++) {
                double preval = y[*it] * (x * features[*it]);
                if (preval > MAX) {
                    g -= (y[*it] / (1 + exp(preval))) * features[*it];
                } else if (preval < -1 * MAX) {
                    g -= (y[*it] / (1 + exp(preval))) * features[*it];
                    f -= preval;
                } else {
                    f += log(1 + exp(-preval));
                    g -= (y[*it] / (1 + exp(preval))) * features[*it];
                }
            }
            f += (rho_ / 2) * (x - new_z_ + (1 / rho_) * new_y_) * (x - new_z_ + (1 / rho_) * new_y_);
            return;
        } else if (admm_ == 3) {
            // general admm
            f = lambda * norm(new_z_, 1);
            g = rho_ * (x - new_z_) + new_y_;
            double val;
            for (vector<int>::iterator it = miniBatch.begin(); it != miniBatch.end(); it++) {
                double preval = y[*it] * (x * features[*it]);
                if (preval > MAX) {
                    g -= (y[*it] / (1 + exp(preval))) * features[*it];
                } else if (preval < -1 * MAX) {
                    g -= (y[*it] / (1 + exp(preval))) * features[*it];
                    f -= preval;
                } else {
                    f += log(1 + exp(-preval));
                    g -= (y[*it] / (1 + exp(preval))) * features[*it];
                }
            }
            f += (rho_ / 2) * (x - new_z_ + (1 / rho_) * new_y_) * (x - new_z_ + (1 / rho_) * new_y_);
            return;
        }
    }

    template
    class L2LogisticLoss<SparseFeature>;

    template
    class L2LogisticLoss<DenseFeature>;
}
