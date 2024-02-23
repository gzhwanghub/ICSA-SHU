#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include <cmath>
#include "tron.h"
void XMinusBY(double *x, const double *y, double b, int dimension);
void XPlusBY(double *x, const double *y, double b, int dimension);
void Assign(double *x, const double *y, int dimension);
double Dot(const double *x, const double *y, int dimension);
double Norm(const double *x, int dimension);
double Sigmoid(double x);

int BacktrackingLineSearch(Function *fun_o, double *x, double *y,double *z,double *g, double *d, double *new_x,
                           int dimension, double &step_size,  double &statistial_time, double min_step = 1e-15, double max_step = 1e15,
                           double c = 1e-4, double r = 0.5, int max_iterations = 50);
#endif