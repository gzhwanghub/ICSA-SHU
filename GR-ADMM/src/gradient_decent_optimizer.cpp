//
// Created by cluster on 2021/5/6.
//

#include <iostream>
#include "../include/gradient_decent_optimizer.h"
#include "../include/simple_algebra.h"

using namespace std;
int BacktrackingLineSearch(Function *function, const double *x, const double *g, const double *d, double *new_x,
                           int dimension, double &step_size, double min_step = 1e-15, double max_step = 1e15,
                           double c = 1e-4, double r = 0.5, int max_iterations = 50) {
    double initial_dg = Dot(d, g, dimension);
    if (initial_dg > 0) {
//        LOG(WARNING) << "initial d dot g > 0";
        return 1;
    }
//    CHECK(step_size > 0);
    if (step_size > max_step) {
        step_size = max_step;
    }
    double function_value, next_function_value;
    function_value = function->Evaluate(x);
//    CHECK(!std::isnan(function_value));
    int k = 0;
    while (k < max_iterations) {
        if (step_size < min_step) {
            step_size = min_step;
        }
        for (int i = 0; i < dimension; ++i) {
            new_x[i] = x[i] + step_size * d[i];
        }
        next_function_value = function->Evaluate(new_x);
//        CHECK(!std::isnan(next_function_value));
        if (step_size == min_step || next_function_value <= function_value + c * step_size * initial_dg) {
            return 0;
        }
        step_size *= r;
        ++k;
    }
    return 2;
}


double learning_rate_decay(int step,double optimizer_max_iter_num=200,double end_learning_rate=1e-2, double base_learning_rate=1,double gamma=2)
{
    double ALPHA = end_learning_rate +
                   (base_learning_rate - end_learning_rate) * pow(1 - 1.0 * step / optimizer_max_iter_num, gamma);
    return ALPHA;
}

void GradientDecentOptimizer::Optimize(double *x) {
    double *d = new double[dimension_];
    double *g = new double[dimension_];
    double *new_g = new double[dimension_];
    double *new_x = new double[dimension_];
    double *s = new double[dimension_];
    double *y = new double[dimension_];

    int k = 0, result;
    double step_size, function_value, next_function_value;
    function_->Gradient(x, g);
    function_value = function_->Evaluate(x);
    while (k < max_iterations_) {
        for (int i = 0; i < dimension_; ++i) {
            d[i] = -g[i];
        }
        if (k == 0) {
            step_size = 1;
        } else {
            step_size = Dot(s, y, dimension_) / Dot(y, y, dimension_);
            step_size = step_size > 0 ? step_size : 1;
        }
        result = BacktrackingLineSearch(function_, x, g, d, new_x, dimension_, step_size);
        if (result != 0) {
            for (int i = 0; i < dimension_; ++i) {
                new_x[i] = x[i] + 1e-6 * d[i];
            }
        }
        function_->Gradient(new_x, new_g);
        next_function_value = function_->Evaluate(new_x);

        if (Norm(new_g, dimension_) <= min_gradient_norm_) {
            break;
        }
        double denom = std::max(std::max(std::abs(function_value), std::abs(next_function_value)), 1.0);
        if ((function_value - next_function_value) / denom <= factor_) {
            break;
        }

        for (int i = 0; i < dimension_; ++i) {
            s[i] = new_x[i] - x[i];
            y[i] = new_g[i] - g[i];
        }
        Assign(x, new_x, dimension_);
        Assign(g, new_g, dimension_);
        function_value = next_function_value;
        ++k;
    }
    //cout<<k<<endl;

    delete[] d;
    delete[] g;
    delete[] new_g;
    delete[] new_x;
    delete[] y;
    delete[] s;
}
