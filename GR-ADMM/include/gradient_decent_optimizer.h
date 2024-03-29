//
// Created by cluster on 2021/5/6.
//

#ifndef GR_ADMM_GRADIENT_DECENT_OPTIMIZER_H
#define GR_ADMM_GRADIENT_DECENT_OPTIMIZER_H


#include "optimizer.h"
#include "differentiable_function.h"
#include "sparse_dataset.h"

class GradientDecentOptimizer : public Optimizer {
public:
    GradientDecentOptimizer(DifferentiableFunction *function, int dimension, double min_gradient_norm = 1e-5,
                            double factor = 1e-8, int max_iterations = 1000) : function_(function),
                                                                               dimension_(dimension),
                                                                               max_iterations_(max_iterations),
                                                                               min_gradient_norm_(min_gradient_norm),
                                                                               factor_(factor) {}

    void Optimize(double *x);

    void SetDataset(SparseDataset *dataset);

    void SetDimension(int dimension) { dimension_ = dimension; }

    void SetMaxIterations(int max_iterations) { max_iterations_ = max_iterations; }

    void SetMinGradientNorm(double min_gradient_norm) { min_gradient_norm_ = min_gradient_norm; }

    void SetFactor(double factor) { factor_ = factor; }

    void SetFunction(DifferentiableFunction *function) { function_ = function; }

private:
    int dimension_;
    int max_iterations_;
    double min_gradient_norm_;
    double factor_;
    DifferentiableFunction *function_;
};



#endif //GR_ADMM_GRADIENT_DECENT_OPTIMIZER_H
