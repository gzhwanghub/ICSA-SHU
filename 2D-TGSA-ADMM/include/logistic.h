
#ifndef LOGISTIC_H
#define LOGISTIC_H


#include <iostream>
#include <vector>
#include "tron.h"
#include "prob.h"
#include <cmath>

using namespace std;

class base_math
{
public:
    static void swap(double& x, double& y) { double t=x; x=y; y=t; }
	static void swap(int& x, int& y) { int t=x; x=y; y=t; }
	static double min(double x,double y) { return (x<y)?x:y; }
	static double max(double x,double y) { return (x>y)?x:y; }
};
class sparse_operator
{
public:
	static double nrm2_sq(const feature_node *x)
	{
		double ret = 0;
		while(x->index != -1)
		{
			ret += x->value*x->value;
			x++;
		}
		return (ret);
	}

	static double dot(const double *s, const feature_node *x)
	{
		double ret = 0;
		while(x->index != -1)
		{
			ret += s[x->index-1]*x->value;
			x++;
		}
		return (ret);
	}

	static void axpy(const double a, const feature_node *x, double *y)
	{
		while(x->index != -1)
		{
			y[x->index-1] += a*x->value;
			x++;
		}
	}
};

class l2r_lr_fun: public Function
{
public:
	l2r_lr_fun(const problem *prob, double C);
	~l2r_lr_fun();

	double fun(double *w,double *y, double *z);
	void grad(double *w, double *g,double *y, double *z);
	void batch_grad(double *g,double *w,double *y,double *z,vector<int> &batch_val);
	void Hv(double *s, double *Hs);
    
	int get_nr_variable(void);
	int get_number();
	void get_diagH(double *M);

private:
	void Xv(double *v, double *Xv);
	void XTv(double *v, double *XTv);
    void one_grad(double *g,double *w,int data_index);
	
	double C;
	double *z;
	double *D;
	const problem *prob;
};



#endif // LOGISTIC_H
