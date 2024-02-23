//
// Created by wdx on 2020-10-17.
//

#include "lbfgs.h"
#include "math_util.h"
using namespace std;
/*LBFGS_OPT::LBFGS_OPT(const function *fun_obj)
{
    this->fun_obj=const_cast<function *>(fun_obj);
     
	 m_=2;
}*/

LBFGS_OPT::~LBFGS_OPT()
{
}

void LBFGS_OPT::optimizer(double *xi,double *yi,double *zi, double *statistical_time)
{
	int dimension_ = fun_obj->get_nr_variable();
    double *d = new double[dimension_];
    double *g = new double[dimension_];
    double *new_g = new double[dimension_];
    double *new_x = new double[dimension_];
    double *alpha = new double[m_];
    double **s = new double *[m_];
    double **y = new double *[m_];
    for (int i = 0; i < m_; ++i) {
        s[i] = new double[dimension_];
        y[i] = new double[dimension_];
    }
    bool failed = false;
    int k = 0, result;
    double step_size, beta, function_value, next_function_value;
    fun_obj->grad(xi, g,yi,zi);
    function_value = fun_obj->fun(xi,yi,zi);
     
    while (k < max_iterations_) {
        for (int i = 0; i < dimension_; ++i) {
            d[i] = -g[i];
        }
        int index;
        for (int i = k - 1; i >= 0 && i >= k - m_; --i) {
            index = i % m_;
            alpha[index] = Dot(s[index], d, dimension_) / Dot(s[index], y[index], dimension_);
            XMinusBY(d, y[index], alpha[index], dimension_);
        }
        for (int i = (k - m_ > 0 ? k - m_ : 0); i <= k - 1; ++i) {
            index = i % m_;
            beta = Dot(y[index], d, dimension_) / Dot(s[index], y[index], dimension_);
            XPlusBY(d, s[index], alpha[index] - beta, dimension_);
        }
        if (k == 0) {
            step_size = 1;
        } else {
            index = (k - 1) % m_;
            step_size = Dot(s[index], y[index], dimension_) / Dot(y[index], y[index], dimension_);
            step_size = step_size > 0 ? step_size : 1;
        }
        result = BacktrackingLineSearch(fun_obj, xi,yi,zi, g, d, new_x, dimension_, step_size, *statistical_time);
        if (result == 1) {
            failed = true;
            break;
        } else if (result == 2) {
            for (int i = 0; i < dimension_; ++i) {
                new_x[i] = xi[i] + 1e-6 * d[i];
            }
        }
		
        fun_obj->grad(new_x, new_g,yi,zi);
        next_function_value = fun_obj->fun(new_x,yi,zi);
         
        if (Norm(new_g, dimension_) <= min_gradient_norm_) {
            break;
        }
        double denom = max(max(abs(function_value), abs(next_function_value)), 1.0);
        if ((function_value - next_function_value) / denom <= factor_) {
            break;
        }
        index = k % m_;
        for (int i = 0; i < dimension_; ++i) {
            y[index][i] = new_g[i] - g[i];
            s[index][i] = new_x[i] - xi[i];
        }
        Assign(xi, new_x, dimension_);
        Assign(g, new_g, dimension_);
        function_value = next_function_value;
        ++k;
    }
    for (int i = 0; i < m_; ++i) {
        delete[] y[i];
        delete[] s[i];
    }
    delete[] d;
    delete[] g;
    delete[] new_g;
    delete[] new_x;
    delete[] y;
    delete[] s;
    delete[] alpha;
}
/*
// x = x - by  
void LBFGS_OPT::XMinusBY(double *x, const double *y, double b, int dimension) {
    for (int i = 0; i < dimension; ++i) {
        x[i] -= b * y[i];
    }
}

// x = x + by  
void LBFGS_OPT::XPlusBY(double *x, const double *y, double b, int dimension) {
    for (int i = 0; i < dimension; ++i) {
        x[i] += b * y[i];
    }
}

// x = y  
void LBFGS_OPT::Assign(double *x, const double *y, int dimension) {
    for (int i = 0; i < dimension; ++i) {
        x[i] = y[i];
    }
}


double LBFGS_OPT::Dot(const double *x, const double *y, int dimension) {
    double sum = 0;
    for (int i = 0; i < dimension; ++i) {
        sum += x[i] * y[i];
    }
    return sum;
}
// L2 Norm  
double LBFGS_OPT::Norm(const double *x, int dimension) {
    return sqrt(Dot(x, x, dimension));
}

int LBFGS_OPT::BacktrackingLineSearch(function *fun_o, double *x, double *y,double *z,double *g, double *d, double *new_x,
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
}*/