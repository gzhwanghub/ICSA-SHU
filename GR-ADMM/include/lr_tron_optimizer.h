//
// Created by cluster on 2020/10/14.
//

#ifndef GR_ADMM_LR_TRON_OPTIMIZER_H
#define GR_ADMM_LR_TRON_OPTIMIZER_H

#include "optimizer.h"
#include "sparse_dataset.h"
#include "neighbors.h"

class LRTronOptimizer : public Optimizer {
public:
    LRTronOptimizer(const double *y, double *msgX, int dimension, double rho, int max_iterations,
                    double epsilon, double cg_epsilon, SparseDataset *dataset, neighbors *near);

    ~LRTronOptimizer();

    void Optimize(double *x);

    void SetDimension(int dimension) { dimension_ = dimension; }

    void SetMaxIterations(int max_iterations) { max_iterations_ = max_iterations; }

    void SetCgEpsilon(double cg_epsilon) { cg_epsilon_ = cg_epsilon; }

    void SetEpsilon(double epsilon) { epsilon_ = epsilon; }

    void SetY(const double *y) { y_ = y; }

    void SetSumX(double *msgX) { sumX = msgX; }

    void SetNears(neighbors *neighbors) { nears = neighbors; }

    void SetDataset(SparseDataset *dataset);

    void SetRho(double rho) { rho_ = rho; }

private:
    int dimension_;
    int max_iterations_;
    double epsilon_;
    double tron_epsilon_;
    double cg_epsilon_;
    double rho_;
    double *D_;
    double *sumX;
    const double *y_;
    neighbors *nears;
    SparseDataset *dataset_;
    int numsofneighbors;

    int trpcg(double delta, double *g, double *M, double *s, double *r, bool *reach_boundary);

    double function_value(const double *x);

    void gradient(const double *x, double *g);

    void get_diag_preconditioner(double *M);

    void Hv(double *s, double *Hs);
};


#endif //GR_ADMM_LR_TRON_OPTIMIZER_H
