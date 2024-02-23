#ifndef RADADMM_ADMM_H
#define RADADMM_ADMM_H

#include <string>
#include <mutex>

#include "sparse_allreduce.h"
#include "data/sparse_dataset.h"
#include "optimizer/lr_tron_optimizer.h"
#include "other/neighbors.h"

using Communicator = spar::Communicator<spar::SumOperator, double>;
using spar::Coordinator;

class ADMM {
public:
    ADMM(int dimension,  int max_iterations, double rho, double l2reg,
         double ABSTOL, double RELTOL, std::string train_data_path, std::string test_data_path,
         int repeat_Iter,int Dynamic_Group,int nodes_Group);

    ~ADMM();

    int GetID() { return id_; }

    void Run();

private:
    bool stopped_;
    bool triggered_;
    int k_;
    int id_;
    int worker_number_;
    int dimension_;
    int max_iterations_;
    int repeatIter;
    int DynamicGroup;
    int nodesOfGroup;

    double rho_;
    double l2reg_;
    uint64_t calculate_time_;
    double ABSTOL_, RELTOL_;
    double *x_, *y_, *z_, *w_;
    double *old_x_, *old_y_, *old_z_;
    double *temp_z_;
    /*double end_time_;
    double start_time_;*/
    timeval start_time_, end_time_;
    std::mutex m_;
    SparseDataset *train_data_;
    SparseDataset *test_data_;
    LRTronOptimizer *optimizer_;
    Coordinator *coordinator_;
    Communicator *communicator_;

    static void PostAsynAllreduce(double *buffer, int dimension, void *parameter, int id);

    static void SupplyAllreduceData(double *buffer, int dimension, void *parameter);

    static void PreTerminate(void *parameter);
};


#endif //RADADMM_ADMM_H
