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
#define KEY_SIZE  sizeof(unsigned int)
#define VALUE_SIZE sizeof(double)
#define INF HUGE_VAL
typedef signed char schar;
using namespace std;

ADMM::ADMM(args_t *args, problem *prob1, string test_file_path, Collective *collective) {
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
    if (l1reg_ > 0) hasL1reg_ = true;
    else hasL1reg_ = false;
    lemada_ = 0.01;
    x_ = new double[dim_];
    y_ = new double[dim_];
    z_ = new double[dim_]; //C = new double[dim];
    z_pre_ = new double[dim_];
    w_ = new double[dim_];
    sum_w_ = new double[dim_];
    cg_iter_ = new int[10]();
    group_num_ = procnum_ / worker_per_group_;
    maingrp_rank_ = (int *) malloc(sizeof(int) * group_num_);
    for (int i = 0; i < dim_; i++) {
        x_[i] = 0;
        y_[i] = 0;
        z_[i] = 0; //C[i] = rho;
        w_[i] = 0;
        sum_w_[i] = 0;
    }
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
        cout << "procnum=" << procnum_ << ",synchronize!!!" << endl;
        char *outfile = new char[100];
        sprintf(outfile, "../out/results_%d_%d_%d.dat", procnum_, barrier_size_,
                filter_type_);
        of_.open(outfile);
        of_ << "procnum=" << procnum_ << ",dim=" << dim_ << ",synchronize!!!"
           << endl;
        char filename[100];
        sprintf(filename, test_file_path.c_str(),procnum_, myid_);
        string dataname(filename);
        string inputfile = dataname;
        predprob_ = new problem(inputfile.c_str());
    }
}
ADMM::~ADMM() {
    delete[]w_;
    delete[]x_;
    delete[]y_;
    delete[]z_;
    // delete []C;
    delete[]z_pre_;
}
double ADMM::GetObjectValue() {
    int instance_num = predprob_->l;
    vector<double> hypothesis(instance_num, 0);
    //计算每个数据中的sigmoid函数的值
    //计算costFunction
    double costFunction = 0.0;
    for (int i = 0; i < instance_num; i++) {
        int j = 0;
        while (true) {
            if (predprob_->x[i][j].index == -1)
                break;
            hypothesis[i] +=
                    predprob_->x[i][j].value * z_[predprob_->x[i][j].index - 1];
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
            hypothesis[i] += prob_->x[i][j].value * z_[prob_->x[i][j].index - 1];
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
void ADMM::x_update() {
    //  omp_set_num_threads(th_num);
    subproblem_tron();
//    scd_l1_svm(0.01,x,1,1);
//    subproblem_lbfgs();
//    subproblem_gd();
}

void ADMM::subproblem_lbfgs() {
//    fun_obj_ = new l2r_l2_svc_fun(prob_,rho_);
    fun_obj_ = new l2r_lr_fun(prob_, rho_);
    lbfgs_obj_ = new LBFGS_OPT(fun_obj_);
    lbfgs_obj_->optimizer(x_, y_, z_, statistical_time_);
    delete fun_obj_;
    delete lbfgs_obj_;
}

void ADMM::subproblem_gd() {
    fun_obj_ = new l2r_l2_svc_fun(prob_,rho_);
//    fun_obj_ = new l2r_lr_fun(prob_, rho_);
    gd_obj_ = new GD_OPT(fun_obj_);
    gd_obj_->optimizer(x_, y_, z_, statistical_time_);
    delete fun_obj_;
    delete gd_obj_;

}
void ADMM::subproblem_tron() {
    fun_obj_ = new l2r_lr_fun(prob_, rho_);
//    fun_obj_ = new l2r_l2_svc_fun(prob_,rho_);
    tron_obj_ = new TRON(fun_obj_, primal_solver_tol_, eps_cg_, 1000);
    tron_obj_->tron(x_, y_, z_, statistical_time_, tron_iteraton_, cg_iter_);
    //free(fun_obj); free(tron_obj);
    delete fun_obj_;
    delete tron_obj_;
}
void ADMM::y_update() {
    int i;
    for (i = 0; i < dim_; i++) {
        y_[i] += rho_ * (x_[i] - z_[i]);
    }
}
void ADMM::softThreshold(double t, double *z) {
    for (size_t i = 0; i < dim_; i++) {
        if (z[i] > t)
            z[i] -= t;
        else if (z[i] <= t && z[i] >= -t) {
            z[i] = 0.0;
        } else
            z[i] += t;
    }
}
void ADMM::z_update() {
    double s = 1.0 / (rho_ * procnum_ + 2 * l2reg_);
    double t = s * l1reg_;
    for (int i = 0; i < dim_; i++) {
        z_pre_[i] = z_[i];
    }
    double tmp;
    for (int i = 0; i < dim_; i++) {
        z_[i] = sum_w_[i] * s;
//        z_[i] = w_[i] * s;
    }
    if (hasL1reg_)
        softThreshold(t, z_);
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
    double eps_pri = sqrt(dim_) * ABSTOL + RELTOL * fmax(nxstack, z_norm);
    double eps_dual = sqrt(dim_) * ABSTOL + RELTOL * nystack;
    //if(myid==0)printf("%10.4f %10.4f %10.4f %10.4f\t", prires, eps_pri, dualres, eps_dual);
    //of<<prires<<"\t"<<eps_pri<<"\t"<<dualres<<"\t"<<eps_dual<<"\t";
    if (send != NULL) delete[] send;
    if (rcv != NULL) delete[] rcv;
    if (prires <= eps_pri && dualres <= eps_dual) {
        return true;
    }
    return false;
}
void ADMM::CreateGroup() {
    int color = myid_ / worker_per_group_;
    MPI_Comm_split(MPI_COMM_WORLD, color, myid_, &SUBGRP_COMM_);
    int subgrp_rank, subgrp_size;
    MPI_Comm_rank(SUBGRP_COMM_, &subgrp_rank);
    MPI_Comm_size(SUBGRP_COMM_, &subgrp_size);
    MPI_Group main_grp, world_grp;
    MPI_Comm_group(MPI_COMM_WORLD, &world_grp);// 返回通信域
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
    double communication_time = 0;
    double total_synchronization_time(0);
    double synchronization_time(0);
    double max_synchronization_time(0);
    double update_time = 0;
    double max_update_time(0);
    double min_update_time(0);
    double total_average_update_time(0);
    double max_wait_time(0);
    double total_average_wait_time(0);
    double close_time = 0;
    double average_updata_time(0);
    int k = 0;
    double total_time(0), single_iteration_time(0);
    double average_wait_time = 0;
    int flag = 0;
    if(myid_ == 0)
        printf("%-3s %-4s %-11s %-10s %-10s %-10s %-10s \n", "#", "RANK", "ObjectValue", "Accuracy", "MUpdateTime", "MWaitTime", "SynchTime");
    MPI_Barrier(MPI_COMM_WORLD);
    CreateGroup();
    collective_->CreateTorus(MPI_COMM_WORLD, TORUS_COMM_, worker_per_group_, procnum_, nbrs_);
    begin_time = MPI_Wtime();
    while (k < max_iterations_) {
        //initialization
        /*for(int i = 0; i < 10; i++){
            cg_iter_[i] = 0;
        }*/
        double accuracy = 0.0;
        double object_value = 0.0;
        start_calculation = MPI_Wtime();
        x_update();
        for (int i = 0; i < dim_; i++) {
            w_[i] = rho_ * x_[i] + y_[i];
        }
        MPI_Barrier(MPI_COMM_WORLD);
        begin_synchronization = MPI_Wtime();
        MPI_Allreduce(w_, sum_w_, dim_, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        collective_->TorusAllreduce(w_, worker_per_group_, group_num_, TORUS_COMM_, nbrs_);
        collective_->RingAllreduce(w_, dim_, MPI_COMM_WORLD);
//        collective_->HierarchicalAllreduce(w_, MAINGRP_COMM_, SUBGRP_COMM_);
//        collective_->HierarchicalTorus(w_, MAINGRP_COMM_, SUBGRP_COMM_, nbrs_);
        end_synchronization = MPI_Wtime();
        z_update();
        /*if (is_stop()){
            flag = 1;
        }*/
        y_update();
        end_time = MPI_Wtime();
        k++;
        if(myid_ == 0){
            accuracy = predict(k);
            object_value = GetObjectValue();
        }
        single_iteration_time = end_time - start_calculation;
        /*calculate update_time/communication_time/synchronization_time*/
        synchronization_time = end_synchronization - begin_synchronization;
        update_time = begin_synchronization - start_calculation;
        MPI_Allreduce(&synchronization_time, &max_synchronization_time, 1, MPI_DOUBLE, MPI_MAX,
                      MPI_COMM_WORLD);
        /*MPI_Allreduce(&update_time, &min_update_time, 1, MPI_DOUBLE, MPI_MIN,
                      MPI_COMM_WORLD);
        MPI_Allreduce(&update_time, &max_update_time, 1, MPI_DOUBLE, MPI_MAX,
                      MPI_COMM_WORLD);*/
        MPI_Allreduce(&update_time, &average_updata_time, 1, MPI_DOUBLE,
                      MPI_SUM, MPI_COMM_WORLD);
        average_updata_time /= procnum_;
        total_synchronization_time += max_synchronization_time;
        average_wait_time = single_iteration_time - average_updata_time;
        total_average_wait_time += average_wait_time;
        total_average_update_time += average_updata_time;
        if(myid_ == 0){
            printf("%-3d %-4d %-11f %-10f %-10f %-10f %-10f \n",
                   k, myid_, object_value, accuracy,  update_time, average_wait_time, max_synchronization_time);
        }
    }
    close_time = MPI_Wtime();
    int training_time = close_time - begin_time;

    //include calculation communicaiton and wait time.
    if (myid_ == 0) {
        cout << "total training time is: " << training_time << "\n" << endl;
        cout << "total average update time is: " << total_average_update_time << "\n" << endl;
        cout << "total average wait time is: " << total_average_wait_time << "\n" << endl;
        cout << "total max synchronization time is: " << total_synchronization_time << "\n" << endl;
    }
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
        double res = 1.0 / (1 + exp (-1 * sparse_operator::dot(z_, element[i])));
        if(predprob_->y[i] == 1 && res >= 0.5){
            ++acnum;
        }
        if(predprob_->y[i] == -1 && res < 0.5){
            ++acnum;
        }
        /*
        if(last_iter == max_iterations_){
            if (res >= 0.5 && (predprob_->y[i] > 0)) tp++;
            if (res >= 0.5 && (predprob_->y[i] < 0)) fp++;
            if (res < 0.5 && (predprob_->y[i] < 0)) tn++;
            if (res < 0.5 && (predprob_->y[i] > 0)) fn++;
            ofstream file("../roc_auc_webspam_128_30.csv",ios::app);
            if(file){
                file << predprob_->y[i] << "," << res << "\n";
            }
        }
         */
    }

    double p = (double) tp / (tp + fp);
    double r = (double) tp / (tp + fn);
    double ftr = (double) fp / (fp + tn);
    double acrate = (double) acnum / instance_num;
    return acrate;
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
    }
}
