/*************************************************************************
    > File Name: logistic_loss.h
    > Description: Logistic Regression
    > Author: Guozheng Wang
    > Mail: gzh.wang@outlook.com
    > Created Time: 2023-08-27
 ************************************************************************/

#ifndef FIADMM_LOGISTIC_LOSS_H
#define FIADMM_LOGISTIC_LOSS_H
#include "../include/Vector.h"
#include "../include/Matrix.h"
#include "../include/VectorOperations.h"
#include "../include/continuous_functions.h"
#include "../include/group_strategy.h"
#include "include/utils.h"

namespace comlkit {
    template <class Feature>
    class LogisticLoss : public ContinuousFunctions {
    protected:
        std::vector<Feature>& features;                 // size of features is number of trainins examples (n)
        Vector y;                 // size of y is number of training examples (n)
        Vector sum_msg_;
        Vector D_; // \nabla f(x)
        int num_neighbor_;
        double lambda;
        double rho_;
        int myid_;

    public:
        LogisticLoss(int numFeatures, std::vector<Feature>& features, Vector& y, Vector& sum_msg, int num_neighbor, double lambda, double rho, int myid);
        LogisticLoss(const LogisticLoss& c);         // copy constructor

        ~LogisticLoss();

        double eval(const Vector& x) const;                 // functionEval
        Vector evalGradient(const Vector& x) const;                 // gradientEval
        void eval(const Vector& x, double& f, Vector& gradien) const;                 // combined function and gradient eval
        Vector evalStochasticGradient(const Vector& x, std::vector<int>& miniBatch) const;                 // stochastic gradient
        void evalStochastic(const Vector& x, double& f, Vector& g, std::vector<int>& miniBatch) const;                 // stochastic evaluation
    };

}
#endif //FIADMM_LOGISTIC_LOSS_H
