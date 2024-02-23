
#ifndef GRDIENT_DECENT_H
#define GRDIENT_DECENT_H


#include<iostream>
#include<vector>
#include"tron.h"
#include"prob.h"
#include<cmath>

using namespace std;


class GD_OPT
{
public:
    GD_OPT(Function *obj,double min_gradient_norm = 1e-5, double factor = 1e-8,
           int max_iterations = 10):fun_obj(obj),max_iterations_(max_iterations),
                                    min_gradient_norm_(min_gradient_norm),factor_(factor) {}

    ~GD_OPT(){}
    void optimizer(double *xi,double *yi,double *zi);
    void sgd_optimizer(double *xi,double *yi,double *zi,int batch_size,double learn_rate);
    /*void XMinusBY(double *x, const double *y, double b, int dimension);
    void XPlusBY(double *x, const double *y, double b, int dimension);
    void Assign(double *x, const double *y, int dimension);
    double Dot(const double *x, const double *y, int dimension);
    double Norm(const double *x, int dimension);
    int BacktrackingLineSearch(function *fun_o, double *x, double *y,double *z,double *g, double *d, double *new_x,
                           int dimension, double &step_size, double min_step = 1e-15, double max_step = 1e15,
                           double c = 1e-4, double r = 0.5, int max_iterations = 50);*/
private:
    Function *fun_obj;
    //int dimension;   //维度
    int m_;     //？LBFGS相关参数，默认值可设置为2
    int max_iterations_;
    double min_gradient_norm_;
    double factor_;
};



#endif // LOGISTIC_H