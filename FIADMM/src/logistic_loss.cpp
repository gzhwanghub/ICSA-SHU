/*************************************************************************
    > File Name: logistic_loss.cpp
    > Description: Logistic function
    > Author: Guozheng Wang
    > Mail: gzh.wang@outlook.com
    > Created Time: 2023-08-27
 ************************************************************************/

#include "include/logistic_loss.h"
#include <iostream>
#include <math.h>
using namespace std;

#include "../include/VectorOperations.h"
#include <assert.h>
#define EPSILON 1e-6
#define MAX 1e2
namespace comlkit {
    template <class Feature>
    LogisticLoss<Feature>::LogisticLoss(int m, std::vector<Feature>& features, Vector& y, Vector& sum_msg, int num_neighbor, double lambda, double rho, int myid) :
            ContinuousFunctions(true, m, features.size()), features(features), y(y), sum_msg_(sum_msg), num_neighbor_(num_neighbor), lambda(lambda), rho_(rho), myid_(myid)
    {
        if (n > 0)
            assert(features[0].numFeatures == m);
        assert(features.size() == y.size());
        D_ = Vector(m,0);
    }

    template <class Feature>
    LogisticLoss<Feature>::LogisticLoss(const LogisticLoss& l) :
            ContinuousFunctions(true, l.m, l.n), features(l.features), y(l.y), sum_msg_(l.sum_msg_), num_neighbor_(l.num_neighbor_), lambda(l.lambda), rho_(l.rho_) {
    }

    template <class Feature>
    LogisticLoss<Feature>::~LogisticLoss(){
    }

    template <class Feature>
    double LogisticLoss<Feature>::eval(const Vector& x) const {
        assert(x.size() == m);
        double sum = 0;
//        double sum = lambda*norm(x, 1); // L1-norm
        for (int i = 0; i < n; i++) {
            double preval = y[i]*(x*features[i]);
            if (preval > MAX)
                continue;
            else if (preval < -1*MAX)
                sum += preval;
            else
                sum += log(1 + exp(-preval));
        }
        for(int i = 0; i < m; ++i){
            sum += rho_ * num_neighbor_ * x[i] * x[i] + x[i] * sum_msg_[i];
        }

        return sum;
    }

    template <class Feature>
    Vector LogisticLoss<Feature>::evalGradient(const Vector& x) const {
        assert(x.size() == m);
        Vector g = Vector(m, 0);
        for (int i = 0; i < m; ++i) {
            g[i] = 2 * rho_ * num_neighbor_ * x[i] + sum_msg_[i];
        }
        for (int i = 0; i < n; i++) {
            double preval = y[i]*(x*features[i]);
            if (preval > MAX)
                g -= y[i]*features[i];
            else if (preval < -1*MAX)
                continue;
            else
                g -= (y[i]/(1 + exp(-preval)))*features[i];
        }

//        l1-norm
//        for (int i = 0; i < m; i++)
//        {
//            if (x[i] != 0)
//            {
//                g[i] += lambda*sign(x[i]);
//            }
//            else
//            {
//                if (g[i] > lambda)
//                    g[i] -= lambda;
//                else if (g[i] < -lambda)
//                    g[i] += lambda;
//            }
//        }
        return g;
    }

    template <class Feature>
    void LogisticLoss<Feature>::eval(const Vector& x, double& f, Vector& g) const {
        assert(x.size() == m);
        g = Vector(m, 0);
//        f = lambda*norm(x, 1);
//        f = 0;
//        g = 2 * rho_ * num_neighbor_ * x + sum_msg_;
        double val;
        for (int i = 0; i < n; i++) {
            double preval = y[i]*(x*features[i]);
            if (preval > MAX) {
                g -= (y[i]/(1 + exp(preval)))*features[i];
//                D_ +=  features[i] * features[i] ;
            }
            else if (preval < -1*MAX) {
                g -= (y[i]/(1 + exp(preval)))*features[i];
                f-=preval;
            }
            else{
                f += log(1 + exp(-preval));
                g -= (y[i]/(1 + exp(preval)))*features[i];
            }
        }
//        f += rho_ * num_neighbor_ * x * x + x * sum_msg_;
//        for (int i = 0; i < m; i++)
//        {
//            if (x[i] != 0)
//            {
//                g[i] += lambda*sign(x[i]);
//            }
//            else
//            {
//                if (g[i] > lambda)
//                    g[i] -= lambda;
//                else if (g[i] < -lambda)
//                    g[i] += lambda;
//            }
//        }
        return;
    }

    template <class Feature>
    Vector LogisticLoss<Feature>::evalStochasticGradient(const Vector& x, std::vector<int>& miniBatch) const {
        assert(x.size() == m);
        Vector g = Vector(m, 0);
        for (vector<int>::iterator it = miniBatch.begin(); it != miniBatch.end(); it++) {
            double preval = y[*it]*(x*features[*it]);
            if (preval > MAX)
                g -= y[*it]*features[*it];
            else if (preval < -1*MAX)
                continue;
            else
                g -= (y[*it]/(1 + exp(-preval)))*features[*it];
        }
        for (int i = 0; i < m; i++)
        {
            if (x[i] != 0)
            {
                g[i] += lambda*sign(x[i]);
            }
            else
            {
                if (g[i] > lambda)
                    g[i] -= lambda;
                else if (g[i] < -lambda)
                    g[i] += lambda;
            }
        }
        return g;
    }

    template <class Feature>
    void LogisticLoss<Feature>::evalStochastic(const Vector& x, double& f, Vector& g, std::vector<int>& miniBatch) const {
        assert(x.size() == m);
        g = Vector(m, 0);
        f = lambda*norm(x, 1);
        double val;
        for (vector<int>::iterator it = miniBatch.begin(); it != miniBatch.end(); it++) {
            double preval = y[*it]*(x*features[*it]);
            if (preval > MAX) {
                g -= (y[*it]/(1 + exp(preval)))*features[*it];
            }
            else if (preval < -1*MAX) {
                g -= (y[*it]/(1 + exp(preval)))*features[*it];
                f-=preval;
            }
            else{
                f += log(1 + exp(-preval));
                g -= (y[*it]/(1 + exp(preval)))*features[*it];
            }
        }
        for (int i = 0; i < m; i++)
        {
            if (x[i] != 0)
            {
                g[i] += lambda*sign(x[i]);
            }
            else
            {
                if (g[i] > lambda)
                    g[i] -= lambda;
                else if (g[i] < -lambda)
                    g[i] += lambda;
            }
        }
        return;
    }
    template class LogisticLoss<SparseFeature>;
//    template class L1LogisticLoss<DenseFeature>;
}