//
// Created by wdx on 2020/6/19.
//

#ifndef SUBPROB_ADMM_SVM_H
#define SUBPROB_ADMM_SVM_H

 
#include <cmath>
#include "prob.h"
#include "tron.h"
#include "function.h"
 

class l2r_l2_svc_fun: public function
{
public:
	l2r_l2_svc_fun(const Problem *prob, double rho);
	~l2r_l2_svc_fun();

	double fun(double *w,double *yi,double *zi);
	void grad(double *w, double *g,double *yi,double *zi);
	void Hv(double *s, double *Hs);
    void get_diagH(double *M);
	int get_nr_variable(void);
	//void get_diag_preconditioner(double *M);

protected:
	void Xv(double *v, double *Xv);
	void XTv(double *v, double *XTv);

    double rho_;
	double *z;
	//Reduce_Vectors *reduce_vectors; 

	int *I;
	int sizeI;
	const Problem *prob;
};


#endif //SUBPROB_ADMM_SVM_H