/*************************************************************************
    > File Name: group_admm.cpp
    > Description: ADMM algorithm
    > Author: Guozheng Wang
    > Mail: gzh.wang@outlook.com
    > Created Time: 2023-7-21
 ************************************************************************/
#include <iostream>
#include "include/group_admm.h"
#include "allreduce/ringallreduce.h"
#include "include/collective.h"
#include "include/SparseFeature.h"
#include <vector>
#include <unistd.h>
#include "include/logistic_loss.h"
#include "include/L2LeastSquaresLoss.h"
#include "include/L2SmoothSVMLoss.h"
#include "include/L2LogisticLoss.h"
#include "include/L2SmoothSVRLoss.h"
#include "include/L2ProbitLoss.h"
#include "include/tron.h"
#include "include/TRON.h"
#include "include/gd.h"
#include "include/gdLineSearch.h"
#include "include/gdBarzilaiBorwein.h"
#include "include/cg.h"
#include "include/sgdAdagrad.h"
#include "include/sgdDecayingLearningRate.h"
#include "include/gdNesterov.h"
#include "include/lbfgsMin.h"
#include "include/SVRDual.h"
#include <mpi.h>
#include "../Eigen/Dense"
#include "include/L1LogisticLoss.h"


#define  random(x)(rand()%x)
#define LR 1
#define SVM 2
#define SVR 3
#define EPSILON 1e-16;
using namespace spar;
using namespace comlkit;

ADMM::ADMM(args_t *args, vector<struct SparseFeature> train_features, comlkit::Vector ytrain,
           vector<struct SparseFeature> test_features, comlkit::Vector ytest, int dimension, int optimizer,
           double beta) { // jensen lib
    // MPI
//        CreateGroup();
    myid = args->myid;
    procnum = args->procnum;
    // Dataset settings.
    train_features_ = train_features;
    ytrain_ = ytrain;
    mtrian_ = train_features_[0].numFeatures;
    ntrian_ = train_features_.size();
    if (myid == 0) {
        test_features_ = test_features;
        ytest_ = ytest;
        mtest_ = test_features_[0].numFeatures;
        ntest_ = test_features_.size();
    }
    dim_ = dimension;
    data_number_ = ntrian_;
    // ADMM parameter setting and initialization.
    rho = args->rho;
    beta_ = 0.2; // high precision && faster convergence
    l2reg_ = 0;
    l1reg_ = 1;
    if (l1reg_ > 0)
        hasL1reg_ = true;
    else
        hasL1reg_ = false;
    optimizer_ = optimizer;
    comlkit::Vector new_x(dim_, 0);
    comlkit::Vector sum_msg(dim_, 0);
    comlkit::Vector new_x_old(dim_, 0);
    comlkit::Vector new_alpha(dim_, 0);
    comlkit::Vector new_alpha_old(dim_, 0);
    comlkit::Vector new_y(dim_, 0);
    comlkit::Vector new_z(dim_, 0);
    weight_ = new int[dim_](); // Sparse ADMM
    y_ = new double[dim_]();
    sum_w_ = new double[dim_]();
    z_ = new double[dim_]();
    w_ = new double[dim_]();
    z_pre_ = new double[dim_]();
    sum_z_ = new double[dim_]();
    new_x_ = new_x;
    new_y_ = new_y;
    new_z_ = new_z;
    sum_msg_ = sum_msg;
    new_x_old_ = new_x_old;
    new_alpha_ = new_alpha;
    new_alpha_old_ = new_alpha;
    maxIteration = args->maxIteration;
    nodesOfGroup = args->nodesOfGroup;// nodesOfGroup = 8
    worker_ranks_ = new int[nodesOfGroup - 1];
    nears.neighborsNums = nodesOfGroup; // Number of nodes in the group, including this node.
    nears.neighs = new int[nears.neighborsNums];
}

void ADMM::CreateGroup() {
    int color_odd, color_even;
    color_odd = myid / nears.neighborsNums;
    MPI_Comm_split(MPI_COMM_WORLD, color_odd, myid, &SUBGRP_COMM_ODD_);
    int subgrp_rank_odd, subgrp_size_odd;
    MPI_Comm_rank(SUBGRP_COMM_ODD_, &subgrp_rank_odd);
    MPI_Comm_size(SUBGRP_COMM_ODD_, &subgrp_size_odd);
    color_even = myid % nears.neighborsNums;
    MPI_Comm_split(MPI_COMM_WORLD, color_even, myid, &SUBGRP_COMM_EVEN_);
    int subgrp_rank_even, subgrp_size_even;
    MPI_Comm_rank(SUBGRP_COMM_EVEN_, &subgrp_rank_even);
    MPI_Comm_size(SUBGRP_COMM_EVEN_, &subgrp_size_even);
}

ADMM::~ADMM() {
    delete[] worker_ranks_;
    delete[] weight_;
    delete[] y_;
    delete[] sum_w_;
    delete[] z_;
    delete[] w_;
    delete[] z_pre_;
    delete[] sum_z_;
}

void ADMM::alpha_update(const comlkit::Vector &new_x, const comlkit::Vector &old, const comlkit::Vector &sumx) {
    int numsofneighbors = nears.neighborsNums - 1;
//    new_alpha_ += rho * (numsofneighbors * new_x - sumx) - beta_ * (new_alpha_ - old); // bad
    new_alpha_ += rho * (numsofneighbors * new_x - sumx) -
                  beta_ * (new_x - old); // primal variable history message is better than dual variable
//    new_alpha_ += rho * (numsofneighbors * new_x - sumx) + beta_ * (new_x - old); // svr regression
//    new_alpha_ +=
//            rho * (numsofneighbors * new_x - sumx); // primal variable history message is better than dual variable
//    new_alpha_ += rho * (numsofneighbors * new_x - sumx);
}

void ADMM::SoftThreshold(double t, Vector &z) {
    for (size_t i = 0; i < dim_; i++) {
        if (z[i] > t)
            z[i] -= t;
        else if (z[i] <= t && z[i] >= -t) {
            z[i] = 0.0;
        } else
            z[i] += t;
    }
}

double ADMM::predict_comlkit(int method, Vector parameter) {
    double p = 0.1;
    double error = 0;
    if (ytest_.empty()) {
        return 0;
    } else {
        int counter = 0;
        int notuse = 0;
        int sample_num = ytest_.size();
        for (int i = 0; i < sample_num; ++i) {
            if (method == 1) {
                // logistic regression && probit
                double val = 1.0 / (1 + exp(-1 * parameter * test_features_[i]));
                if (val >= 0.5 && ytest_[i] == 1) {
                    counter++;
                } else if (val < 0.5 && ytest_[i] == -1) {
                    counter++;
                }
            } else if (method == 2) {
                // svm
                double val = parameter * test_features_[i];
                if (val >= 0 && ytest_[i] == 1) {
                    counter++;
                } else if (val < 0 && ytest_[i] == -1) {
                    counter++;
                }
            } else if (method == 3) {
                // svr
                error += pow(((parameter * test_features_[i]) - ytest_[i]), 2);
            } else if (method == 4) {
                counter++;
            }
        }
        if (method == 3) {
            return pow(error / sample_num, 0.5); // RMSE of svr
        } else {
            return counter * 100.0 / sample_num; // svm & lr prediction accuracy.
        }
    }
}

double ADMM::loss_value_comlkit(int method, Vector parameter) {
    double p = 0.1;
    if (ytest_.empty()) {
        return 0;
    } else {
        double sum = 0;
        int sample_num = test_features_.size();
        if (method == 1) {
            // LR loss
            for (int i = 0; i < sample_num; ++i) {
                sum += std::log(1 + std::exp(-ytest_[i] * parameter * test_features_[i])); // logistic regression loss
            }
        } else if (method == 2) {
            // SVM loss
            for (int i = 0; i < sample_num; i++) {
                double preval = ytest_[i] * (parameter * test_features_[i]);
                if (1 - preval >= 0) {
                    sum += (1 - preval) * (1 - preval);
                }
            }
        } else if (method == 3) {
            // SVR loss
            for (int i = 0; i < sample_num; ++i) {
                double preval = (parameter * test_features_[i]) - ytest_[i];
                if (preval < -p) {
                    sum += (preval + p) * (preval + p);
                } else if (preval > p) {
                    sum += (preval - p) * (preval - p);
                }
            }
        } else if (method == 4) {
            // Probit loss
            for (int i = 0; i < sample_num; i++) {
                double val = ytest_[i] * (parameter * test_features_[i]) / sqrt(2);

                double probitval = 0.5 * (1 + erf(val)) + EPSILON;
                sum -= log(probitval);
            }
        } else if (method == 5) {
            // LS loss
            for (int i = 0; i < sample_num; i++) {
                double val = ytest_[i] - (parameter * test_features_[i]);
                sum += val * val;
            }
        }

        return sum / sample_num;
    }
}

void ADMM::group_train(clock_t start_time) {
    // Instantiate the communication time and calculate the time variable.
    double b_time, e_time;
    double comm_btime, comm_etime, cal_btime, cal_etime;
    double comm_time, avarage_comm_time, cal_time, one_iteration_time, sum_iteration_time = 0.0, avarage_cal_time, fore_time = 0.0, sum_cal_time = 0.0;
    MPI_Status status;
    int32_t k = 1;
    vector<int> nodes;
    double sparseCount = 0;
    if (myid == 0)
        printf("%3s %12s %12s %12s %12s %12s %12s\n", "#", "loss", "accuracy or RMES", "single_iter_time", "comm_time",
               "cal_time",
               "sum_time");
    b_time = start_time;
    // Create Torus group.
    //    CreateGroup();
    // Experimental results are saved to a csv file.
    FILE *fp = fopen("probit_rcv1_16_gadmm2_sgddecay_50.csv", "w+");
    if (fp == NULL) {
        fprintf(stderr, "fopen() failed.\n");
        exit(EXIT_FAILURE);
    }
    while (k <= maxIteration) {
        // Request group generation from the group generator Send the current iteration where it is located.
        MPI_Send(&k, 1, MPI_INT, procnum - 1, 1, MPI_COMM_WORLD);
        // Get the generated group.
        MPI_Probe(procnum - 1, 2, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, MPI_INT, &nears.neighborsNums); // Get node's neighbor states.
        MPI_Recv(nears.neighs, nears.neighborsNums, MPI_INT, procnum - 1, 2, MPI_COMM_WORLD,
                 &status); // Receive a vector containing INT.
        // Grouping method, can be divided into groups.
        MPI_Group world_group;
        MPI_Comm_group(MPI_COMM_WORLD, &world_group);
        MPI_Group worker_group;
        MPI_Group_incl(world_group, nears.neighborsNums, nears.neighs, &worker_group);
        MPI_Comm worker_comm;
        MPI_Comm_create_group(MPI_COMM_WORLD, worker_group, 0, &worker_comm);
        comm_time = 0;
        // Torus分组
        if (k != 1) {
            for (int i = 0; i < dim_; i++) {
                sum_msg_[i] = new_alpha_[i] -
                              rho *
                              ((nears.neighborsNums - 1) * new_x_[i] +
                               sum_msg_[i]); // 本worker与邻居worker聚合的x_j{j\in neighs}
            }
        }
        new_x_old_ = new_x_;
        new_alpha_old_ = new_alpha_;
        cal_btime = MPI_Wtime();
        // Instantiating the correspondence problem form.
//        LogisticLoss<SparseFeature> ll(mtrian_, train_features_, ytrain_, sum_msg_, nears.neighborsNums-1, 1, rho, myid);
//        L2LeastSquaresLoss<SparseFeature> ls(mtrian_, features_, label_, sum_msg_, nears.neighborsNums, 1, rho);
//        L2LogisticLoss<SparseFeature> lr(mtrian_, train_features_, ytrain_, sum_msg_, nears.neighborsNums-1, 1, rho);
//        L2SmoothSVRLoss<SparseFeature> svr(mtrian_, train_features_, ytrain_, sum_msg_, nears.neighborsNums, 1, rho, 1);
//        L2SmoothSVMLoss<SparseFeature> svm(mtrian_, train_features_, ytrain_, sum_msg_, nears.neighborsNums, 1, rho, 1);
        L2ProbitLoss<SparseFeature> probit(mtrian_, train_features_, ytrain_, sum_msg_, nears.neighborsNums, 1, rho, 1);
        // Subproblem Solving Optimizer
//        if (k <= 2) {zhi
//            new_x_ = gdNesterov(svr, new_x_, 1, 1e-4, 50);
//        } else {
//            new_x_ = gdNesterov(svr, new_x_, 1, 1e-4, 10);
//        }
//        new_x_ = gd(probit, new_x_, 0.1, 50);
//        new_x_ = gdLineSearch(svm, new_x_, 1, 1e-4, 50);
//        new_x_ = gdNesterov(svm, new_x_, 1, 1e-4, 50);
//        new_x_ = gdBarzilaiBorwein(svm, new_x_, 1, 1e-4, 50);
//        new_x_ = sgdAdagrad(svr, new_x_, ntrian_, 1e-2, 200, 1e-4, 50);
        new_x_ = sgdDecayingLearningRate(probit, new_x_, ntrian_, 0.5 * 1e-1, 200, 1e-4, 50);
//        new_x_ = cg(svm, new_x_, 1, 1e-4, 50);
//        new_x_ = lbfgsMin(svm, new_x_, 1, 1e-4, 50);
//        new_x_ = tron(svm, myid, new_x_, 50);
//        new_x_ = gdNesterov(svm, new_x_, 1, 1e-4, 50);
//        new_x_ = SVRDual(train_features_, ytrain_, 2, 1, 0.1, 1e-3, 50);
        cal_etime = MPI_Wtime();
        // Model parameter synchronization
        double *new_x_temp = new double[dim_];
        double *sum_msg_temp = new double[dim_];
        for (int i = 0; i < dim_; ++i) {
            new_x_temp[i] = new_x_[i];
            sum_msg_temp[i] = sum_msg_[i];
        }
        comm_btime = MPI_Wtime();
        MPI_Allreduce(new_x_temp, sum_msg_temp, dim_, MPI_DOUBLE, MPI_SUM, worker_comm);
        // Torus synchronizaiton method.
//        if (k % 2 == 0) {
//            MPI_Allreduce(new_x_temp, sum_msg_temp, dim_, MPI_DOUBLE, MPI_SUM, SUBGRP_COMM_EVEN_);
//        } else {
//            MPI_Allreduce(new_x_temp, sum_msg_temp, dim_, MPI_DOUBLE, MPI_SUM, SUBGRP_COMM_ODD_);
//        }
        comm_etime = MPI_Wtime();
        for (int i = 0; i < dim_; ++i) {
            sum_msg_temp[i] -= new_x_temp[i];
            new_x_[i] = new_x_temp[i];
            sum_msg_[i] = sum_msg_temp[i];
        }
        alpha_update(new_x_, new_x_old_, sum_msg_);
        delete[] new_x_temp;
        delete[] sum_msg_temp;
        e_time = MPI_Wtime();
        one_iteration_time = (double) (e_time - cal_btime);

        cal_time = (double) (cal_etime - cal_btime);
        comm_time = (double) (comm_etime - comm_btime);
//        MPI_Barrier(MPI_COMM_WORLD);
        if (myid == 0) {
            sparseCount = 0;
            sum_iteration_time += one_iteration_time;
            sum_cal_time += cal_time;
            // Model sparsity calculation
//                for (int i = 0; i < dim; i++) {
//                    if (new_x_[i] < 1e-5) {
//                        sparseCount++;
//                    }
//                }
            // Output of calculation results to the console
            double loss = loss_value_comlkit(4, new_x_);
            double predict = predict_comlkit(1, new_x_);

            printf("%3d %12f %12f %12f %12f %12f %12f\n", k, loss, predict,
                   one_iteration_time,
                   comm_time, sum_cal_time, sum_iteration_time);
            // Calculation results are stored to a file
            fprintf(fp, "%d %f %f %f %f %f %f\n", k, loss, predict,
                    one_iteration_time,
                    comm_time, sum_cal_time, sum_iteration_time);
            fore_time = one_iteration_time;
            sum_comm_ += comm_time;
        }
        k++;
    }
    fclose(fp);
}
