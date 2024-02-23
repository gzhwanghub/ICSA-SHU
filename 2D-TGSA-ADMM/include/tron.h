#ifndef _TRON_H
#define _TRON_H

#include <cmath>
#include <stdio.h>
#include <stdarg.h>
#include <cstring>
#include <vector>
using namespace std;

class Function
{
public:
    virtual double fun(double *w,double *y, double *z) = 0 ;
    virtual void grad(double *w, double *g,double *y, double *z) = 0 ;
    virtual void batch_grad(double *g,double *w,double *y,double *z,vector<int> &batch_val)=0;
    virtual void Hv(double *s, double *Hs) = 0 ;

    virtual int get_nr_variable(void) = 0 ;
    virtual int get_number(void) = 0 ;
    virtual void get_diagH(double *M) = 0 ;
    virtual ~Function(void){}
};

class TRON
{
public:
    TRON(const Function *fun_obj, double eps = 0.1, double eps_cg = 0.1, int max_iter = 1000);
    ~TRON();

    void tron(double *w, double *y, double*z);
    void set_print_string(void (*i_print) (const char *buf));

private:
    //int trcg(double delta, double *g, double *s, double *r, bool *reach_boundary);
    int trpcg(double delta, double *g, double *M, double *s, double *r, bool *reach_boundary);
    double norm_inf(int n, double *x);

    double eps;
    double eps_cg;
    int max_iter;
    Function *fun_obj;
    void info(const char *fmt,...);
    void (*tron_print_string)(const char *buf);
    double gnorm0;
    int n;
    double *s, *r, *g;

};
#endif