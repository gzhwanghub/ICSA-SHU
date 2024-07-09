/*************************************************************************
    > File Name: train.cpp
    > Description: Main entrance. Target: Nodes are divided into groups
      before each iteration and sets communicate within groups.
    > Author: Guozheng Wang
    > Mail: gzh.wang@outlook.com
    > Created Time: 2020-10-14
 ************************************************************************/

#include <mpi.h>
#include <iostream>
#include <stdio.h>
#include <string>
#include "include/conf_util.h"
#include "include/group_admm.h"
#include "include/properties.h"
#include "include/group_strategy.h"
#include "allreduce/reduceoperator.h"
#include "allreduce/simpleallreduce.h"
#include "include/FileIO.h"
using namespace std;
using namespace comlkit;

int main(int argc, char **argv) {
    int myid, procnum;
    double start_time, end_time;
    char filename[100];
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &procnum);
    // Two ways to get information from the configuration file and store it in the argument
    args_t *args = new args_t(myid, procnum); // Way 1
    Properties properties(argc, argv); // Way 2
    double rho = properties.GetDouble("rho");
    int maxIteration = properties.GetInt("maxIteration");
    int nodesOfGroup = properties.GetInt("nodesOfGroup");
    int DynamicGroup = properties.GetInt("DynamicGroup");
    int QuantifyPart = properties.GetInt("QuantifyPart");
    int Comm_method = properties.GetInt("Comm_method");
    int Update_method = properties.GetInt("Update_method");
    int Repeat_iter = properties.GetInt("repeat_iter");
    int Sparse_comm = properties.GetInt("sparse_comm");
//    int optimizer = properties.GetInt("optimizer");
//    double beta = properties.GetInt("beta");
    int optimizer;
    double beta;
    std::string train_data_path = properties.GetString("train_data_path");
    std::string test_data_path = properties.GetString("test_data_path");
    // The last node acts as a group generator. No file reads are required.
    if (myid == procnum - 1) {
        GroupStrategy group_trategy(Repeat_iter);
        group_trategy.MasterNodes(procnum, nodesOfGroup, DynamicGroup, maxIteration);
        MPI_Finalize();
    } else {
        // Read training set and test set files.
//        sprintf(filename, "/mirror/dataset/log1p/%d/data%03d", procnum - 1, myid);
//        char const *testdata_file = "/mirror/dataset/log1p/test";
//        sprintf(filename, "/mirror/dataset/tfidf/%d/data%03d", procnum - 1, myid);
//        char const *testdata_file = "/mirror/dataset/tfidf/test";
//        sprintf(filename, "/mirror/dataset/real/%d/data%03d", procnum - 1, myid);
//        char const *testdata_file = "/mirror/dataset/real/test";
//        sprintf(filename, "/mirror/dataset/gisette/%d/data%03d", procnum - 1, myid);
//        char const *testdata_file = "/mirror/dataset/gisette/test";
        sprintf(filename, "/mirror/dataset/news20old/%d/data%03d", procnum - 1, myid);
        char const *testdata_file = "/mirror/dataset/news20old/test";
//        sprintf(filename, "/mirror/dataset/rcv1/%d/data%03d", procnum - 1, myid);
//        char const *testdata_file = "/mirror/dataset/rcv1/test";
//        sprintf(filename, "/mirror/dataset/url/%d/data%03d", procnum, myid);
//        char const *testdata_file = "/mirror/dataset/url/test";
//        sprintf(filename, "/mirror/dataset/kddbr/%d/data%03d", procnum, myid);
//        char const *testdata_file = "/mirror/dataset/kddbr/test";
//        sprintf(filename, "/mirror/dataset/avazu/%d/data%03d", procnum, myid);
//        char const *testdata_file = "/mirror/dataset/avazu/test";
        vector<struct SparseFeature> train_features, test_features;
        comlkit::Vector ytrain, ytest;
        int ntrian, mtrian, ntest, mtest;
        readFeatureLabelsLibSVM(filename, train_features, ytrain, ntrian, mtrian);
        if (myid == 0) {
            readFeatureLabelsLibSVM(testdata_file, test_features, ytest, ntest, mtest);
        }
        // Ensure that the Consistency of model parameter dimensions between processes.
        int temp = train_features[0].numFeatures;
        if(myid == 0){
            temp = max(temp, test_features[0].numFeatures); // select the max dim between train feature and test feature.
        }
        vector<int> nodelist;
        for (int i = 0; i < procnum - 1; i++) // warning!!!
            nodelist.push_back(i);
        spar::SimpleAllreduce<spar::MaxOperator, int>(&temp, 1, myid, nodelist, MPI_COMM_WORLD);
        for (int i = 0; i < train_features.size(); ++i) {
            train_features[i].numFeatures = temp;
        }
        // Parameters read in the Properties class are assigned to the args_t class, redundant operation.
        args->maxIteration = maxIteration;
        args->nodesOfGroup = nodesOfGroup;
        args->rho = rho;
        args->Comm_method = Comm_method;
        args->Update_method = Update_method;
        args->Repeat_iter = Repeat_iter;
        // Instantiate admm and assign a value.
        ADMM admm(args, train_features, ytrain, test_features, ytest, temp, optimizer, beta);
        admm.sum_cal_ = 0.0;
        admm.sum_comm_ = 0.0;
        admm.quantify_part_ = QuantifyPart;
        admm.dynamic_group_ = DynamicGroup;
        admm.update_method_ = Update_method;
        admm.sparse_comm_ = Sparse_comm;
        // Exporting relevant configuration information.
        if (myid == 0)
            args->print_args();
        start_time = MPI_Wtime();
        admm.group_train(start_time);
        end_time = MPI_Wtime();
        if (myid == 0) {
            double temp = (double) (end_time - start_time);
            cout << "run time: " << temp << "  "
                 << "comm time:" << admm.sum_comm_ << "  "
                 << "cal  time:" << temp - admm.sum_comm_ << endl;
        }
        MPI_Finalize();
    }
}
