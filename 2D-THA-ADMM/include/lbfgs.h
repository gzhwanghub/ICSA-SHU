
#ifndef LBFGS_H
#define LBFGS_H


#include<iostream>
#include<vector>
#include"tron.h"
#include"prob.h"
#include<cmath>

using namespace std;

 
class LBFGS_OPT
{
public:
	LBFGS_OPT(Function *obj,double min_gradient_norm = 1e-5, double factor = 1e-8,
	int m = 2, int max_iterations = 10):fun_obj(obj), m_(m),max_iterations_(max_iterations),
                                        min_gradient_norm_(min_gradient_norm),factor_(factor) {}
                                                                                 
	~LBFGS_OPT();
    void optimizer(double *xi,double *yi,double *zi, double *statistical_time);
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
