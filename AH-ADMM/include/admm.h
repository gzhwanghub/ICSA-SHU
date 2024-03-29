
//
// Created by zjh on 2019/11/25.
//

#ifndef ADMM_END_ADMM_H
#define ADMM_END_ADMM_H

#include <omp.h>
#include "admm_comm.h"
#include "prob.h"
#include "function.h"
#include "dcd.h"
#include "svm.h"
#include "tron.h"
#include "l2r_lr_fun.h"
#include "l2r_lr_fun_multicore_tron.h"


class Admm : public AdmmComm{

public:
    Admm(int proc_id,int proc_size,int master_id,std::vector<int> worker_id,Problem *prob,
             int partial_barrier,int bounded_delay,
             std::string solve_sub_problem="multicore_tron",std::string reg="l2",int thread_nums=1,
             double rho=1.0,double C=1.0,double abs_tol=1e-3,double rel_tol=1e-3,int admm_iter=100);

    ~Admm();

    bool MasterUpdate(int k);
    void WorkerUpdate(int k);
    void train();
    double t1,t2;

private:
    int data_num_,dim_;
    double *x_,*y_,*z_,*z_pre_;

    double rho_;
    double C_;
    double abs_tol_,rel_tol_;
    int admm_iter_;
    double *alpha;

    std::string solve_sub_problem_;
    std::string reg_;
    int thread_nums_;

    Problem *prob_;
    function *fun_obj_;

    void x_update();
    void y_update();
    void z_update();
    bool is_stop();
    double current_accuracy();
    double current_cost_function();

    void tron_svm();
    void tron_lr();
    void subproblem_gd();
    void subproblem_dcd(int f);

    double getObjectValue();
    double getSVMObjectValue();

    double total_run,total_compute;

};


#endif //ADMM_END_ADMM_H
