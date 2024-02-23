//
// Created by cluster on 2020/10/14.
//

#ifndef GR_ADMM_ADMM_LR_FUNCTION_H
#define GR_ADMM_ADMM_LR_FUNCTION_H


#include "differentiable_function.h"
#include "sparse_dataset.h"
#include "neighbors.h"

/* ADMM算法下LR模型的损失函数 */
class AdmmLRFunction : public DifferentiableFunction {
public:
    AdmmLRFunction(const double *y,double *msgX ,int dimension, double rho,
                   SparseDataset *dataset,neighbors *neighbors) : y_(y),sumX(msgX),rho_(rho), dimension_(dimension), dataset_(dataset),nears(neighbors){}

    double Evaluate(const double *x);

    void Gradient(const double *x, double *g);

    void SetRho(double rho) { rho_ = rho; }

    void SetDataset(SparseDataset *dataset) { dataset_ = dataset; }

    void SetY(const double *y) { y_ = y; }

    void SetSumX(const double *msgX) { sumX = msgX; }

//    void SetZ(const double *z) { z_ = z; }

    void SetDimension(int dimension) { dimension_ = dimension; }

private:
    int dimension_;
    double rho_;
    const double *y_;
//    const double *z_;
    const double *sumX;
    SparseDataset *dataset_;
    neighbors *nears;
};


#endif //GR_ADMM_ADMM_LR_FUNCTION_H
