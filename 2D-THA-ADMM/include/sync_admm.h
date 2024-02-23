#ifndef ASYC_ADMM_H
#define ASYC_ADMM_H
#undef SEEK_SET
#undef SEEK_END
#undef SEEK_CUR

#include <mpi.h>
#include <iostream>
#include <cmath>
#include <fstream>
#include <time.h>
#include <stdio.h>
#include "prob.h"
#include "logistic.h"
#include "tron.h"
#include "gd.h"
#include "collective.h"
#include "properties.h"
//#include"conf_util.h"
#include "svm.h"
#include "lbfgs.h"
#include "gd.h"

using namespace std;

struct meta
{
    int key;
    double value;
};

class ADMM
{
public:
    ADMM(args_t *args, problem *prob, string test_file_path, Collective *collective);
   //  ADMM(args_t *args, problem *prob,int type);
    ~ADMM();
    void x_update();
    void y_update();
    void z_update();
    bool is_stop();
    void softThreshold(double t, double * z);
    void train();
    double predict(int last_iter);
    void draw();
    double GetObjectValue();
	double GetSVMObjectValue(int type);
    void subproblem_tron();
	void subproblem_lbfgs();
	void subproblem_gd();
    void rho_update();
    void rho_update_ac(int iter_time);
    void rho_update_dr(int iter_time);
    void CreateGroup();
	//void scd_l1_svm(double eps,double* opt_w,double Cp,double Cn);
private:
    int data_num_, dim_;
	ofstream of_;
    int myid_, procnum_;
    int barrier_size_, delta;
    double *x_, *y_, *z_, *z_pre_, *w_, *sum_w_;   //     *C,
    double statistical_time_[5];//ObjectFuntion[0],Gradient[1],CG[2],Diagonal[3],Hassian[4]
    meta **msgbuf_;
    double rho_;
    double lemada_;
    double ABSTOL;
    double RELTOL;
    double l2reg_, l1reg_;
    bool hasL1reg_;
    bool filter_flag;
    double primal_solver_tol_, eps_, eps_cg_;
    problem *prob_, *predprob_;
    TRON *tron_obj_;
    LBFGS_OPT *lbfgs_obj_;
    GD_OPT *gd_obj_;
    Function *fun_obj_, *pred_obj_;
    string outfile_;
    Collective *collective_;
    // int th_num;
    double *y_pre_;
    int rho_flag_;
    double costFunction_,preFunction_;
    double tau_sum_;
	double TAU;
    int power_n_;
    double* s_sendBuf_;
	int filter_type_;
	int max_iterations_;
    int tron_iteraton_;
    int worker_per_group_;
    int group_num_;
    int *cg_iter_;
    MPI_Comm  MAINGRP_COMM_, SUBGRP_COMM_, TORUS_COMM_;
    MPI_Group main_group_, world_group_;
    int *maingrp_rank_;
    int nbrs_[4];
};

#endif // ASYC_ADMM_H
