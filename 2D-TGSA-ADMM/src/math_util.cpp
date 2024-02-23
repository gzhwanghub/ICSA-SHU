#include <iostream>
#include "math_util.h"
#include <algorithm>
using namespace std;
/* x = x - by */
void XMinusBY(double *x, const double *y, double b, int dimension) {
    for (int i = 0; i < dimension; ++i) {
        x[i] -= b * y[i];
    }
}

/* x = x + by */
void XPlusBY(double *x, const double *y, double b, int dimension) {
    for (int i = 0; i < dimension; ++i) {
        x[i] += b * y[i];
    }
}

/* x = y */
void Assign(double *x, const double *y, int dimension) {
    for (int i = 0; i < dimension; ++i) {
        x[i] = y[i];
    }
}

double Dot(const double *x, const double *y, int dimension) {
    double sum = 0;
    for (int i = 0; i < dimension; ++i) {
        sum += x[i] * y[i];
    }
    return sum;
}
/* L2 Norm */
double Norm(const double *x, int dimension) {
    return sqrt(Dot(x, x, dimension));
}

double Sigmoid(const double x) {
    return 1.0 / (1 + exp(-x));
}

int BacktrackingLineSearch(Function *fun_o, double *x, double *y,double *z,double *g, double *d, double *new_x,
                           int dimension, double &step_size, double min_step, double max_step, double c, double r,
                           int max_iterations) {
    double initial_dg = Dot(d, g, dimension);
    if (initial_dg > 0) {
        cout << "initial d dot g > 0";
        return 1;
    }

    if (step_size > max_step) {
        step_size = max_step;
    }
    double function_value, next_function_value;
    function_value = fun_o->fun(x,y,z);

    int k = 0;
    while (k < max_iterations) {
        if (step_size < min_step) {
            step_size = min_step;
        }
        for (int i = 0; i < dimension; ++i) {
            new_x[i] = x[i] + step_size * d[i];
        }
        next_function_value = fun_o->fun(new_x,y,z);

        if (step_size == min_step || next_function_value <= function_value + c * step_size * initial_dg) {
            return 0;
        }
        step_size *= r;
        ++k;
    }
    return 2;
}
int cmp(const PAIR& x, const PAIR& y){
    return abs(x.second) > abs(y.second);
}
void TopK(std::vector<PAIR> raw_vector, double* topk_array, int N){
    sort(raw_vector.begin(), raw_vector.end(), cmp);
    for(auto it = raw_vector.begin() + N; it != raw_vector.end(); ++it){
//        store_difference[it->first] = topk_array[it->first];
        topk_array[it->first] = 0.0;
    }
}