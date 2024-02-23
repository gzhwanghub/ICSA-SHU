//
// Created by wdx on 2020-10-17.
//
#include <algorithm>
#include "gd.h"
#include "math_util.h"
using namespace std;

void GD_OPT::optimizer(double *xi,double *yi,double *zi)
{
    int dimension_ = fun_obj->get_nr_variable();
    double *d = new double[dimension_];
    double *g = new double[dimension_];
    double *new_g = new double[dimension_];
    double *new_x = new double[dimension_];
    double *s = new double[dimension_];
    double *y = new double[dimension_];

    int k = 0, result;
    double step_size, function_value, next_function_value;
    fun_obj->grad(xi, g, yi, zi);
    function_value = fun_obj->fun(xi,yi,zi);

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
        result = BacktrackingLineSearch(fun_obj, xi,yi,zi, g, d, new_x, dimension_, step_size);
        if (result != 0) {
            for (int i = 0; i < dimension_; ++i) {
                new_x[i] = xi[i] + 1e-6 * d[i];
            }
        }
        fun_obj->grad(new_x, new_g, yi, zi);
        next_function_value = fun_obj->fun(new_x, yi, zi);

        if (Norm(new_g, dimension_) <= min_gradient_norm_) {
            break;
        }
        double denom = max(max(abs(function_value), abs(next_function_value)), 1.0);
        if ((function_value - next_function_value) / denom <= factor_) {
            break;
        }
        for (int i = 0; i < dimension_; ++i) {
            s[i] = new_x[i] - xi[i];
            y[i] = new_g[i] - g[i];
        }
        Assign(xi, new_x, dimension_);
        Assign(g, new_g, dimension_);
        function_value = next_function_value;
        ++k;
    }

    delete[] d;
    delete[] g;
    delete[] new_g;
    delete[] new_x;
    delete[] y;
    delete[] s;
}
void GD_OPT::sgd_optimizer(double *xi,double *yi,double *zi,int batch_size,double learn_rate)
{
    int dimension = fun_obj->get_nr_variable();
    int dataNum = fun_obj->get_number();
    int begin_index=0,end_index=batch_size;
    double *g = new double[dimension];
    vector<int> data_;
    for(int i=0;i<dataNum;i++)
    {
        data_.push_back(i);
    }
    int k=0;

    while(k<max_iterations_)
    {
        double function_value = fun_obj->fun(xi,yi,zi);
        random_shuffle(data_.begin(),data_.end());
        while(begin_index<dataNum){
            vector<int> current_datas(data_.begin()+begin_index,data_.begin()+min(end_index,dataNum));
            //得到梯度
            fun_obj->batch_grad(g,xi,yi,zi,current_datas);
            //更新局部变量
            for (int i = 0; i < dimension; i++) {
                xi[i]=xi[i]-learn_rate*g[i];
            }

            begin_index+=batch_size;
            end_index+=batch_size;
        }
        if (Norm(g, dimension) <= min_gradient_norm_) {
            break;
        }
        k++;
    }
    delete[] g;
}