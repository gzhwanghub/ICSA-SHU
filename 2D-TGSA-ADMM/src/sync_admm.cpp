/*****************************************************************************************
    > File Name: sycn_admm.cpp
    > Author: Guozheng Wang
    > Date: 2021-7-16
    > Function:Primal ADMM，Using MPI_ALLREDUCE or 2D-Torus, Ring AllReduce, 2D-THA, Hierarchical AllReduce聚合w.
 ****************************************************************************************/
#include <iostream>
#include <time.h>
#include "sync_admm.h"
#include <unistd.h>
#include <math.h>
#include "math_util.h"

#define KEY_SIZE  sizeof(unsigned int)
#define VALUE_SIZE sizeof(double)
#define INF HUGE_VAL
typedef signed char schar;
using namespace std;
int ADMM::flag_ = 0;

ADMM::ADMM(args_t *args, problem *prob1, string test_file_path,
           Collective *collective) {
    prob_ = prob1;
    data_num_ = prob_->l;
    collective_ = collective;
    dim_ = prob_->n;
    myid_ = args->myid;
    procnum_ = args->procnum;
    barrier_size_ = args->min_barrier;
    delta = args->max_delay;
    max_iterations_ = args->max_iterations;
    filter_type_ = args->filter_type;
    rho_ = args->rho;
    ABSTOL = args->ABSTOL;
    RELTOL = args->RELTOL;
    l2reg_ = args->l2reg;
    l1reg_ = args->l1reg;
    worker_per_group_ = args->worker_per_group_;
    if (l1reg_ > 0)
        hasL1reg_ = true;
    else
        hasL1reg_ = false;
    lambda_ = 0.01;
    x_ = new double[dim_]();
    y_ = new double[dim_]();
//    y_bar_ = new double[dim_]();
    z_ = new double[dim_](); //C = new double[dim];
    z_pre_ = new double[dim_]();
    w_ = new double[dim_]();
    sum_w_ = new double[dim_]();
    sum_z_ = new double[dim_]();
    weight_ = new int[dim_]();
    grad_xminusz_ = new double[dim_]();
    int group_num = procnum_ / worker_per_group_;
    eps_ = 0.001;
    eps_cg_ = 0.1;
    int pos = 0;
    int neg = 0;
    for (int i = 0; i < prob_->l; i++)
        if (prob_->y[i] > 0)
            pos++;
    neg = prob_->l - pos;
    primal_solver_tol_ = eps_ * max(min(pos, neg), 1) / prob_->l;
    //MPI_Barrier(MPI_COMM_WORLD);
    if (myid_ == 0) {
        cout << "procnum = " << procnum_ << ",synchronize!!!" << endl;
        cout << "Max Iterations = " << max_iterations_ << endl;
        char filename[100];
        // 这里有个问题！！！！
//        strcpy(filename, test_file_path.c_str());
        sprintf(filename, test_file_path.c_str(), procnum_, myid_);
        string dataname(filename);
        string inputfile = dataname;
        predprob_ = new problem(inputfile.c_str());
    }
}

ADMM::~ADMM() {
    delete[] x_;
    delete[] y_;
    delete[] z_;
    delete[] z_pre_;
    delete[] w_;
    delete[] sum_w_;
    delete[] sum_z_;
    delete[] weight_;
}

double ADMM::GetObjectValue() {
    int instance_num = predprob_->l;
    vector<double> hypothesis(instance_num, 0);
    //计算每个数据中的sigmoid函数的值，计算costFunction
    double costFunction = 0.0;
    for (int i = 0; i < instance_num; i++) {
        int j = 0;
        while (true) {
            if (predprob_->x[i][j].index == -1)
                break;
            hypothesis[i] +=
                    predprob_->x[i][j].value * sum_z_[predprob_->x[i][j].index - 1];
            j++;
        }
        costFunction += std::log(
                1 + std::exp(-predprob_->y[i] * hypothesis[i]));
    }
    costFunction = costFunction / instance_num;
    return costFunction;
}

double ADMM::GetSVMObjectValue(int type) {
    int datanumber = prob_->l;
    vector<double> hypothesis(datanumber, 0);
    for (int i = 0; i < datanumber; i++) {
        int j = 0;
        while (true) {
            if (prob_->x[i][j].index == -1)
                break;
            hypothesis[i] +=
                    prob_->x[i][j].value * z_[prob_->x[i][j].index - 1];
            j++;
        }
    }
    //计算costFunction
    double costFunction = 0.0;
    if (type == 0)         // L2-SVM
    {
        for (int i = 0; i < datanumber; i++) {
            double d = 1 - prob_->y[i] * hypothesis[i];
            if (d > 0)
                costFunction += d * d;
        }
    } else {                                   //L1-SVM
        for (int i = 0; i < datanumber; i++) {
            double d = 1 - prob_->y[i] * hypothesis[i];
            if (d > 0)
                costFunction += d;
        }
    }
    costFunction = costFunction / datanumber;
    return costFunction;
}

double ADMM::predict(int last_iter) {
    //MPI_Allreduce(x,w,dim,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    int acnum = 0;
    int tp = 0;
    int fp = 0;
    int fn = 0;
    int tn = 0;
    int instance_num = predprob_->l;
    for (int i = 0; i < instance_num; i++) {
        feature_node **element = predprob_->x;
        double res = 1.0 / (1 + exp(-1 * sparse_operator::dot(sum_z_, element[i])));
        if (predprob_->y[i] == 1 && res >= 0.5) {
            ++acnum;
        }
        if (predprob_->y[i] == -1 && res < 0.5) {
            ++acnum;
        }
        // ROC
        /*if(last_iter == max_iterations_){
            if (res >= 0.5 && (predprob_->y[i] > 0)) tp++;
            if (res >= 0.5 && (predprob_->y[i] < 0)) fp++;
            if (res < 0.5 && (predprob_->y[i] < 0)) tn++;
            if (res < 0.5 && (predprob_->y[i] > 0)) fn++;
            ofstream file("../roc_auc.csv",ios::app);
            if(file){
                file << predprob_->y[i] << "," << res << "\n";
            }
        }*/
    }
    double p = (double) tp / (tp + fp);
    double r = (double) tp / (tp + fn);
    double ftr = (double) fp / (fp + tn);
    double acrate = (double) acnum / instance_num;
    return acrate;
}

void ADMM::x_update() {
//      omp_set_num_threads(th_num);
    subproblem_tron();
//    scd_l1_svm(0.01,x,1,1);
//    subproblem_lbfgs();
//    subproblem_gd();
}

void ADMM::subproblem_lbfgs() {
//    fun_obj_ = new l2r_l2_svc_fun(prob_,rho_);
    fun_obj_ = new l2r_lr_fun(prob_, rho_);
    lbfgs_obj_ = new LBFGS_OPT(fun_obj_);
    lbfgs_obj_->optimizer(x_, y_, z_);
    delete fun_obj_;
    delete lbfgs_obj_;
}

void ADMM::subproblem_gd() {
    fun_obj_ = new l2r_l2_svc_fun(prob_, rho_);
//    fun_obj_ = new l2r_lr_fun(prob_, rho_);
    gd_obj_ = new GD_OPT(fun_obj_);
    gd_obj_->optimizer(x_, y_, z_);
    delete fun_obj_;
    delete gd_obj_;

}

void ADMM::subproblem_tron() {
    fun_obj_ = new l2r_lr_fun(prob_, rho_);
//    fun_obj_ = new l2r_l2_svc_fun(prob_,rho_);
    tron_obj_ = new TRON(fun_obj_, primal_solver_tol_, eps_cg_, 1000);
    tron_obj_->tron(x_, y_, z_);
    delete fun_obj_;
    delete tron_obj_;
}

void ADMM::y_update() {
//    std::vector<PAIR> y_raw_vector;
    for (int i = 0; i < dim_; i++) {
        y_[i] += rho_ * (x_[i] - z_[i]);
//        y_raw_vector.push_back(PAIR(i, y_[i]));
    }
//    TopK(y_raw_vector, y_, 10000);

}

void ADMM::DualUpdate() {
    // gradient = x - z, larger gradient direction
    for (int i = 0; i < index_num_; i++) {
        int key = key_index_[i];
        y_[key] += rho_ * (x_[key] - z_[key]);
    }

//    std::vector<PAIR> y_raw_vector;
//    for (int i = 0; i < index_num_; i++) {
//        int key = key_index_[i];
//        y_raw_vector.push_back(PAIR(key, y_[key]));
//    }
//    TopK(y_raw_vector, y_, index_num_ * 0.5);
//    y_raw_vector.clear();
}
void ADMM::XMinusZ(){
    // gradient = x - z, larger gradient direction

    for (int i = 0; i < index_num_; i++) {
        int key = key_index_[i];
        grad_xminusz_[key] = (x_[key] - z_[key]);
    }
    std::vector<PAIR> y_raw_vector;
    for (int i = 0; i < index_num_; i++) {
        int key = key_index_[i];
        y_raw_vector.push_back(PAIR(key, grad_xminusz_[key]));
    }
    TopK(y_raw_vector, grad_xminusz_, index_num_ * 0.5);
    y_raw_vector.clear();
};

void ADMM::z_update() {
    double s = 1.0 / (rho_ * procnum_ + 2 * l2reg_);
    double t = s * l1reg_;
    for (int i = 0; i < dim_; i++) {
        z_pre_[i] = z_[i];
    }
    double tmp;
    for (int i = 0; i < dim_; i++) {
//        z_[i] = sum_w_[i] * s;  // primal allreduce
        z_[i] = w_[i] * s;
    }
    if (hasL1reg_) {
        SoftThreshold(t, z_);
    }
}

void ADMM::ConsensusUpdate() {
    double s = 1.0 / (rho_ * procnum_ + 2 * l2reg_);
    double t = s * l1reg_;
    for (int i = 0; i < dim_; i++) {
        z_pre_[i] = z_[i];
    }
    double tmp;
    for (int i = 0; i < index_num_; i++) {
        int key = key_index_[i];
        s = 1.0 / (rho_ * weight_[key] + 2 * l2reg_);
//        z_[key] = w_[key] * s;
        z_[key] = sum_w_[key] * s;  // HierarchicalAllreduce()
    }
    if (hasL1reg_)
        SoftThreshold(t, z_);
}

void ADMM::SoftThreshold(double t, double *z) {
    for (size_t i = 0; i < dim_; i++) {
        if (z[i] > t)
            z[i] -= t;
        else if (z[i] <= t && z[i] >= -t) {
            z[i] = 0.0;
        } else
            z[i] += t;
    }
}

void ADMM::adaptive_rho(){
    double *send = new double[3]();
    double *rcv = new double[3]();
    for (int i = 0; i < index_num_; i++) {
        int key = key_index_[i];
        send[0] += (x_[key] - z_[key]) * (x_[key] - z_[key]);
        send[1] += x_[key] * x_[key];
        send[2] += y_[key] * y_[key];
    }
    MPI_Allreduce(send, rcv, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double prires = sqrt(rcv[0] / procnum_);
    double zdiff = 0.0;
    double z_squrednorm = 0.0;
    for (int i = 0; i < index_num_; i++) {
        int key = key_index_[i];
        zdiff += (z_[key] - z_pre_[key]) * (z_[key] - z_pre_[key]);
        z_squrednorm += z_[key] * z_[key];
    }
    double dualres = rho_ * sqrt(zdiff);
    if (prires > 10 * dualres) {
        rho_ = 2 * rho_;
    } else if (dualres > 10 * prires) {
        rho_ = 0.5 * rho_;
    } else {
        rho_ = rho_;
    }
}

bool ADMM::is_stop() {
    double *send = new double[3];
    double *rcv = new double[3];
    for (int i = 0; i < 3; i++) {
        send[i] = 0;
        rcv[i] = 0;
    }
    for (int i = 0; i < dim_; i++) {
        send[0] += (x_[i] - z_[i]) * (x_[i] - z_[i]);
        send[1] += x_[i] * x_[i];
        send[2] += y_[i] * y_[i];
    }
    MPI_Allreduce(send, rcv, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double prires = sqrt(rcv[0] / procnum_);
    double nxstack = sqrt(rcv[1] / procnum_);
    double nystack = sqrt(rcv[2] / procnum_);
    double zdiff = 0.0;
    double z_squrednorm = 0.0;
    for (int i = 0; i < dim_; i++) {
        zdiff += (z_[i] - z_pre_[i]) * (z_[i] - z_pre_[i]);
        z_squrednorm += z_[i] * z_[i];
    }
    double z_norm = sqrt(z_squrednorm);
    double dualres = rho_ * sqrt(zdiff);
    if (prires > 10 * dualres) {
        rho_ = 2 * rho_;
    } else if (dualres > 10 * prires) {
        rho_ = 0.5 * rho_;
    } else {
        rho_ = rho_;
    }
    double eps_pri = sqrt(dim_) * ABSTOL + RELTOL * fmax(nxstack, z_norm);
    double eps_dual = sqrt(dim_) * ABSTOL + RELTOL * nystack;
    if (send != NULL) delete[] send;
    if (rcv != NULL) delete[] rcv;
    if (prires <= eps_pri && dualres <= eps_dual) {
        return true;
    }
    return false;
}

void ADMM::CreateGroup() {
    MPI_Comm_split(MPI_COMM_WORLD, myid_ / worker_per_group_, myid_, &SUBGRP_COMM_);
    MPI_Comm_split(MPI_COMM_WORLD, myid_ % worker_per_group_, myid_, &SUBGRP_COMM_Y_);
    int subgrp_rank, subgrp_size, subgrp_rank_y, subgrp_size_y;
    MPI_Comm_rank(SUBGRP_COMM_, &subgrp_rank);
    MPI_Comm_rank(SUBGRP_COMM_Y_, &subgrp_rank_y);
    MPI_Comm_size(SUBGRP_COMM_, &subgrp_size);
    MPI_Comm_size(SUBGRP_COMM_Y_, &subgrp_size_y);
    MPI_Group main_grp, world_grp;
    MPI_Comm_group(MPI_COMM_WORLD, &world_grp);
    int group_num = procnum_ / worker_per_group_;
    int *maingrp_ranks = (int *) malloc(sizeof(int) * group_num);
    for (int i = 0; i < group_num; ++i) {
        maingrp_ranks[i] = i * worker_per_group_;
    }
    MPI_Group_incl(world_grp, group_num, maingrp_ranks, &main_grp);
    MPI_Comm_create_group(MPI_COMM_WORLD, main_grp, 0, &MAINGRP_COMM_);
}

void ADMM::train() {
    //initialization
    double begin_time, end_time, begin_synchronization, end_synchronization, start_calculation;
    double iteration_time(0), update_time(0), wait_time(
            0), synchronization_time(0);
    double average_iteration_time(0), average_update_time(0), average_wait_time(
            0), average_synch_time(0);
    double sum_iteration_time(0), sum_update_time(0), sum_wait_time(
            0), sum_synch_time(0);
    double close_time(0);
    int k = 0;
    //initialization
    double accuracy = 0.0;
    double object_value = 0.0;
    double sparse_ratio = 0.0;
    float nonzero_count = 0.0;
    int flag = 0;
    // initial spectral stepsize reference Adaptive ADMM with Spectral Penalty Parameter Selection
/*    double* delta_ybar = new double[dim_]();
    double* ybar_zero = new double[dim_]();
    double* delta_x = new double[dim_]();
    double* delta_y = new double[dim_]();
    double* delta_z = new double[dim_]();
    double* x_zero = new double[dim_]();
    double* y_zero = new double[dim_]();
    double* z_zero = new double[dim_]();
    double alpha_sd, beta_sd;
    double alpha_mg, beta_mg;
    double alpha_bar, beta_bar;
    double alpha_cor, beta_cor;*/
    /*initial general form consensus model*/
//    fun_obj_ = new l2r_lr_fun(prob_, rho_);
//    tron_obj_ = new TRON(fun_obj_, primal_solver_tol_, eps_cg_, 1000);
//    tron_obj_->tron(x_, y_, z_);
//    delete fun_obj_;
//    delete tron_obj_;
//    MPI_Barrier(MPI_COMM_WORLD);
    x_update();
    index_num_ = 0;
    int *local_weight = new int[dim_]();
    for (int i = 0; i < dim_; i++) {
        if (x_[i] != 0) {
            index_num_++; // the number of non-zero value every dimension
            key_index_.push_back(i);
            local_weight[i] = 1;
        } else {
            local_weight[i] = 0;
        }
    }
    MPI_Allreduce(local_weight, weight_, dim_, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    delete[] local_weight;
    // output definition
    /*if (myid_ == 0) {       // test TRON method module time
        printf("%-3s %-4s %-11s %-10s %-20s %-9s %-11s %-10s %-10s %-10s %-10s %-10s \n", "#", "RANK",
               "TRONIter-k", "CG-time", "HvIter-k", "Hv-time", "Update-time",
               "ObjectValue","Accuracy","Gradient", "Diagonal", "Comm-time");
    }*/
    if (myid_ == 0)
        printf("%-3s %-4s %-11s %-10s %-12s %12s %11s %11s %-12s %-11s %-11s %-11s %-4s\n",
               "#", "RANK", "ObjectValue", "Accuracy",
               "SumIterTime", "SumUpdaTime", "SumWaitTime", "SumSynTime", "AvUpdaTime", "AvWaitTime", "AvSynTime",
               "SparseRatio", "Rho");
    MPI_Barrier(MPI_COMM_WORLD);
    CreateGroup();
//    collective_->CreateTorus(MPI_COMM_WORLD, TORUS_COMM_, sqrt(procnum_),
//                             nbrs_);
    begin_time = MPI_Wtime();
    while (k < max_iterations_) {
        accuracy = 0.0;
        nonzero_count = 0.0;
        start_calculation = MPI_Wtime();
        // global consensus ADMM
//        x_update();
//        for (int i = 0; i < dim_; i++) {
//            w_[i] = rho_ * x_[i] + y_[i];
//            if (w_[i] != 0.0) {
//                nonzero_count++;
//            }
//        }
        // general form consensus ADMM
        if(k > 0){
            x_update();
        }
        for(int i = 0; i < dim_; i++){
            w_[i] = 0;
        }
        for (int i = 0; i < index_num_; i++) {
            int key = key_index_[i];
//            x_[key] = 1.6 * x_[key] + (1 - 1.6) * z_[key]; // over-relaxation condition
            w_[key] = rho_ * x_[key] + y_[key];
        }
        for (int i = 0; i < dim_; i++) {
            if(w_[i] != 0.0){
                nonzero_count++;
            }
        }
        sparse_ratio = float(nonzero_count) / dim_;
        MPI_Barrier(MPI_COMM_WORLD);
        begin_synchronization = MPI_Wtime();
        /*sum_w_*/
//        MPI_Allreduce(w_, sum_w_, dim_, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        collective_->HierarchicalSparseAllreduce(w_, sum_w_, MAINGRP_COMM_, SUBGRP_COMM_);
        /*w_*/
//        collective_->WRHT(w_, sum_w_, MAINGRP_COMM_, SUBGRP_COMM_);
        collective_->MoshpitAllreduce(w_, sum_w_, SUBGRP_COMM_, SUBGRP_COMM_Y_);
        collective_->SparseRingAllreduce(w_, dim_, MPI_COMM_WORLD);
//        collective_->TorusAllreduce(w_, sqrt(procnum_), TORUS_COMM_, nbrs_);
//        collective_->SparseTorusAllreduce(w_, sqrt(procnum_), TORUS_COMM_, nbrs_);
//        collective_->RingAllreduce(w_, dim_, MPI_COMM_WORLD);
//        collective_->HierarchicalRingAllreduce(w_, MAINGRP_COMM_, SUBGRP_COMM_);
//        collective_->HierarchicalTorus(w_, MAINGRP_COMM_, SUBGRP_COMM_, nbrs_);
//        collective_->SparseHierarchicalTorus(w_, MAINGRP_COMM_, SUBGRP_COMM_,nbrs_);
        end_synchronization = MPI_Wtime();
        // global consensus ADMM
//        z_update();
//        if (is_stop()) {
//                    flag = 1;
//                }
//        y_update();
        // general form consensus ADMM(adaptive rho, TopK sparse gradient)
        ConsensusUpdate();
        XMinusZ();
        adaptive_rho();
        DualUpdate();
        end_time = MPI_Wtime();
        // Adaptive ADMM with Spectral Penalty Parameter Selection
/*        if(k % 2 == 1){
            for (int i = 0; i < dim_; i++) {
                y_bar_[i] = y_[i] + rho_ * (x_[i] - z_pre_[i]);
            }
            for (int i = 0; i < dim_; i++) {
                delta_ybar[i] = y_bar_[i] - ybar_zero[i];
                delta_y[i] = y_[i] - y_zero[i];
                delta_x[i] = x_[i] - x_zero[i];
                delta_z[i] = z_[i] - z_zero[i];
            }

            alpha_sd = Dot(delta_ybar, delta_ybar, dim_) / Dot(delta_x, delta_ybar, dim_);
            alpha_mg = Dot(delta_x, delta_ybar, dim_) / Dot(delta_x, delta_x,dim_);
            beta_sd = Dot(delta_y, delta_y, dim_) / Dot(delta_z, delta_y, dim_);
            beta_mg = Dot(delta_z, delta_y, dim_) / Dot(delta_z, delta_z, dim_);
            if(2 * alpha_mg > alpha_sd){
                alpha_bar = alpha_mg;
            }else{
                alpha_bar = alpha_sd - 0.5 * alpha_mg;
            }
            if( 2 * beta_mg > beta_sd){
                beta_bar = beta_mg;
            }else{
                beta_bar = beta_sd - 0.5 * beta_mg;
            }
            alpha_cor = Dot(delta_x, delta_ybar, dim_) / (sqrt(Norm(delta_x, dim_)) *
                                                           sqrt(Norm(delta_ybar, dim_)));
            beta_cor = Dot(delta_z, delta_y, dim_) /(sqrt(Norm(delta_z, dim_)) *
                                                      sqrt(Norm(delta_y, dim_)));
            if(alpha_cor > 0.2 && beta_cor > 0.2){
                rho_ = sqrt(alpha_bar * beta_bar);
            }else if(alpha_cor > 0.2 && beta_cor <= 0.2){
                rho_ = alpha_bar;
            }else if(alpha_cor <= 0.2 && beta_cor > 0.2){
                rho_ = beta_bar;
            }else{
                rho_ = rho_;
            }
            for (int i = 0; i < dim_; i++) {
                ybar_zero[i] = y_bar_[i];
                y_zero[i] = y_[i];
                x_zero[i] = x_[i];
                z_zero[i] = z_[i];
            }
        } else{
            rho_ = rho_;
        }*/
        // general form consensus ADMM
        MPI_Allreduce(z_, sum_z_, dim_, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        for (int i = 0; i < dim_; i++) {
            if (weight_[i] != 0) {
                sum_z_[i] = sum_z_[i] / weight_[i];
            }
        }
        k++;
        if (myid_ ==
            0) { // if node0 is Stragglers, it will increase training time.
            accuracy = predict(k);
            object_value = GetObjectValue();
        }
        /*calculate update_time/communication_time/synchronization_time*/
        iteration_time = end_time - start_calculation;
        synchronization_time = end_synchronization - begin_synchronization;
        update_time = begin_synchronization - start_calculation;
        wait_time = iteration_time - update_time;
        MPI_Allreduce(&iteration_time, &average_iteration_time, 1, MPI_DOUBLE,
                      MPI_SUM,
                      MPI_COMM_WORLD);
        MPI_Allreduce(&synchronization_time, &average_synch_time, 1, MPI_DOUBLE,
                      MPI_SUM,
                      MPI_COMM_WORLD);
        MPI_Allreduce(&update_time, &average_update_time, 1, MPI_DOUBLE,
                      MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&wait_time, &average_wait_time, 1, MPI_DOUBLE,
                      MPI_SUM, MPI_COMM_WORLD);
        average_iteration_time /= procnum_;
        average_update_time /= procnum_;
        average_wait_time /= procnum_;
        average_synch_time /= procnum_;
        sum_iteration_time += average_iteration_time;
        sum_update_time += average_update_time;
        sum_wait_time += average_wait_time;
        sum_synch_time += average_synch_time;
        if (myid_ == 0) {
            printf("%-3d %-4d %-11f %-10f %-12f %12f %11f %11f %-12f %-11f %-11f %-11f %-4f\n",
                   k, myid_, object_value, accuracy, sum_iteration_time,
                   sum_update_time, sum_wait_time, sum_synch_time,
            average_update_time, average_wait_time, average_synch_time,
                    sparse_ratio, rho_);
        }
    }
    close_time = MPI_Wtime();
    double training_time = close_time - begin_time;
    double average_training_time(0);
    MPI_Allreduce(&training_time, &average_training_time, 1, MPI_DOUBLE,
                  MPI_SUM, MPI_COMM_WORLD);
    average_training_time /= procnum_;
    //include calculation communicaiton and wait time.
    if (myid_ == 0) {
        cout << "total average training time is: " << average_training_time
             << "\n" << endl;
        cout << "total average update time is: " << sum_update_time
             << "\n" << endl;
        cout << "total average wait time is: " << sum_wait_time
             << "\n" << endl;
        cout << "total average synchronization time is: "
             << sum_synch_time << "\n" << endl;
    }
}

void ADMM::draw() {
    if (myid_ == 0) {
        int acnum = 0;
        int tp = 0;
        int fp = 0;
        int fn = 0;
        int tn = 0;
        int datalen = predprob_->l;
        // int datalen = prob->l;
        double *gradValue = new double[51];
        for (int j = 0; j < 51; j++) {
            gradValue[j] = 0.02 * j;
        }
        for (int j = 0; j < 51; j++) {
            for (int i = 0; i < datalen; i++) {
                feature_node **data = predprob_->x;
                double res =
                        1.0 / (1 + exp(-1 * sparse_operator::dot(z_, data[i])));
                if (res > gradValue[j] && (predprob_->y[i] > 0)) tp++;
                if (res > gradValue[j] && (predprob_->y[i] < 0)) fp++;
                if (res <= gradValue[j] && (predprob_->y[i] <= 0)) tn++;
                if (res <= gradValue[j] && (predprob_->y[i] > 0)) fn++;
            }
            double p = (double) tp / (tp + fp);
            double r = (double) tp / (tp + fn);
            double ftr = (double) fp / (fp + tn);
            cout << p << "\t" << r << "\t" << ftr << "\n";
        }
        delete[]gradValue;
    }
}
