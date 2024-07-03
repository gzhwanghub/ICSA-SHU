/*************************************************************************
    > File Name: continuous_funtions.h
    > Description: Parent clas of smooth continuous function
    > Author: Guozheng Wang
    > Mail: gzh.wang@outlook.com
    > Created Time: 2021-05-06
 ************************************************************************/
#ifndef FIADMM_CONTINUOUS_FUNCTIONS_H
#define FIADMM_CONTINUOUS_FUNCTIONS_H


#include "../include/Vector.h"
#include "../include/Matrix.h"
#include "../include/VectorOperations.h"
#include "../include/MatrixOperations.h"

namespace comlkit {

    class ContinuousFunctions {
    protected:
        int n;                  // The number of convex functions added together, i.e if g(X) = \sum_{i = 1}^n f_i(x)
        int m;                 // Dimension of vectors or features (i.e. size of x in f(x))
    public:
        bool isSmooth;

        ContinuousFunctions(bool isSmooth);

        ContinuousFunctions(bool isSmooth, int m, int n);

        ContinuousFunctions(const ContinuousFunctions &c);         // copy constructor

        virtual ~ContinuousFunctions();

        virtual double eval(const Vector &x) const;                 // functionEval
        virtual Vector evalGradient(const Vector &x) const;                 // gradientEval
        virtual void
        eval(const Vector &x, double &f, Vector &gradient) const;                 // combined function and gradient eval
        virtual Vector
        evalStochasticGradient(const Vector &x, std::vector<int> &batch) const;                 // stochastic gradient
        virtual void evalStochastic(const Vector &x, double &f, Vector &g,
                                    std::vector<int> &miniBatch) const;                 // stochastic combined evaluation
        virtual Matrix evalHessian(const Vector &x) const;                      // hessianEval
        virtual void evalHessianVectorProduct(const Vector &x, const Vector &v,
                                              Vector &Hxv) const;                 // evaluate a product between a hessian and a vector
        double operator()(const Vector &x) const;

        int size() const;                 // number of features or dimension size (m)
        int length() const;                 // number of convex functions adding up (n)
    };

}
#endif
