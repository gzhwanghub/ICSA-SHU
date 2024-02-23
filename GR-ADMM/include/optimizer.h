//
// Created by cluster on 2020/10/14.
//

#ifndef GR_ADMM_OPTIMIZER_H
#define GR_ADMM_OPTIMIZER_H


class Optimizer {
public:
    virtual void Optimize(double *x) = 0;
};

#endif //GR_ADMM_OPTIMIZER_H
