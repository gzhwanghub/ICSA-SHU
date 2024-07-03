/*************************************************************************
    > File Name: group_admm.h
    > Description: ADMM
    > Author: Guozheng Wang
    > Mail: gzh.wang@outlook.com
    > Created Time: 2023-7-21
 ************************************************************************/

#ifndef FIADMM_GROUP_ADMM_H
#define FIADMM_GROUP_ADMM_H
#undef SEEK_SET
#undef SEEK_END
#undef SEEK_CUR

#include <mpi.h>
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <fstream>
#include <time.h>
#include <stdio.h>
#include "conf_util.h"
#include "neighbors.h"
#include "coefficient_matrix.h"
#include "Eigen/Dense"
#include "Vector.h"
#include "FileIO.h"

using namespace std;
using namespace comlkit;


class ADMM {
public:

    ADMM(args_t *args, vector<struct SparseFeature> train_features, comlkit::Vector ytrain,
         vector<struct SparseFeature> test_features, comlkit::Vector ytest, int dimension, int optimizer, double beta);

    ~ADMM();

    void alpha_update(const comlkit::Vector &new_x, const comlkit::Vector &x_old, const comlkit::Vector &sumx);

    void SoftThreshold(double t, Vector& z);

    void group_train(clock_t start_time);

    double predict_comlkit(int method, Vector parameter);

    double loss_value_comlkit(int method, Vector parameter);

    void CreateGroup();

    ofstream of;
    neighbors nears;
    double sum_cal_, sum_comm_;
    int quantify_part_, dynamic_group_, update_method_, sparse_comm_;
private:
    // ADMM algorithm
    int myid, procnum, dim_, data_number_;
    comlkit::Vector new_x_, sum_msg_, new_y_, new_z_, new_x_old_, new_alpha_, new_alpha_old_;
    double rho; // Penalty term parameter
    double beta_; // Inertial parameters
    int optimizer_;
    // Problem
    vector<struct SparseFeature> features_;
    Vector label_;
    Vector solution_;
    vector<struct SparseFeature> train_features_, test_features_;
    comlkit::Vector ytrain_, ytest_;
    int ntrian_, mtrian_, ntest_, mtest_;
    int maxIteration;
    int nodesOfGroup;
    // Group Strategy
    MPI_Comm SUBGRP_COMM_ODD_;
    MPI_Comm new_comm2_;
    MPI_Comm SUBGRP_COMM_EVEN_;
    int *worker_ranks_;
    //Sparse ADMM
    int index_num_, l1reg_, l2reg_;
    bool hasL1reg_;
    vector<int> key_index_;
    double *w_, *y_, *sum_w_, *z_, *z_pre_, *sum_z_;
    int *weight_;
};


#endif //GR_ADMM_GROUP_ADMM_H
