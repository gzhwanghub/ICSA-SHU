//
// Created by wdx on 2020-10-17.
//
#include <algorithm>
#include "gd.h"
#include "math_util.h"
using namespace std;

void GD_OPT::optimizer(double *xi,double *yi,double *zi, double *statistical_time)
{
	int dimension_ = fun_obj->get_nr_variable();
	double *d = new double[dimension_];
    double *g = new double[dimension_];
    double *new_g = new double[dimension_];
    double *new_x = new double[dimension_];
    double *s = new double[dimension_];
    double *y = new double[dimension_];
    int k = 0, result;
    double object_function_begin_time(0), object_function_end_time(0),
            gradient_begin_time(0), gradient_end_time(0), GD_begin_time(0), GD_end_time(0);
    double step_size, function_value, next_function_value;
    object_function_begin_time = MPI_Wtime();
    function_value = fun_obj->fun(xi,yi,zi);
    object_function_end_time = MPI_Wtime();
    statistical_time[0] += object_function_end_time - object_function_begin_time;
    gradient_begin_time = MPI_Wtime();
    fun_obj->grad(xi, g, yi, zi);
    gradient_end_time = MPI_Wtime();
    statistical_time[1] += gradient_end_time - gradient_begin_time;
    GD_begin_time = MPI_Wtime();
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
		result = BacktrackingLineSearch(fun_obj, xi,yi,zi, g, d, new_x, dimension_, step_size, statistical_time[3]);
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
        /*stop criteria*/
        if ((function_value - next_function_value) / denom <= factor_) {
            break;
        }
        for (int i = 0; i < dimension_; ++i) {
            s[i] = new_x[i] - xi[i];//direction
            y[i] = new_g[i] - g[i];//gradient direction
        }
        Assign(xi, new_x, dimension_);
        Assign(g, new_g, dimension_);
        function_value = next_function_value;
        ++k;
    }
    GD_end_time = MPI_Wtime();
    statistical_time[2] += GD_end_time - GD_begin_time;
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
	
	while(k < max_iterations_)
	{
		double function_value = fun_obj->fun(xi,yi,zi);
		random_shuffle(data_.begin(),data_.end());
		while(begin_index<dataNum){
			vector<int> current_datas(data_.begin()+begin_index,data_.begin()+min(end_index,dataNum));
				//get gradient
			fun_obj->batch_grad(g,xi,yi,zi,current_datas);
				//更新局部变量
			for (int i = 0; i < dimension; i++) {
				xi[i]=xi[i]-learn_rate * g[i];
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