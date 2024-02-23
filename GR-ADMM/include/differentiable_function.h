//
// Created by cluster on 2021/5/6.
//

#ifndef GR_ADMM_DIFFERENTIABLE_FUNCTION_H
#define GR_ADMM_DIFFERENTIABLE_FUNCTION_H


class Function {
public:
    virtual double Evaluate(const double *x) = 0;
    virtual void Gradient(const double *x, double *g) = 0;
};

class DifferentiableFunction : public Function {

};


#endif //GR_ADMM_DIFFERENTIABLE_FUNCTION_H
