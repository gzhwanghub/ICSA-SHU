#ifndef _ADPREDICT_ADMM_H_
#define _ADPREDICT_ADMM_H_

#include <iostream>
#include <fstream>

#include "dmlc/data.h"
#include "dmlc/io.h"
#include "learner.h"
//#include "communication.h"
#include "./internal/reduce_operator.h"
#include "./internal/ring_allreduce.h"
#include "metric.h"
#include <math.h>
#include <time.h>
using namespace spar;

namespace adPredictAlgo {

const double RELTOL = 1e-4;
const double ABSTOL = 1e-5;

class ADMM {
  public:
   ADMM(dmlc::RowBlockIter<unsigned> *_dtrain) : dtrain(_dtrain)
   {
     learner = "NULL";
     num_fea = 0;
     num_data = 0;
     num_procs = 0;
     rank = -1;
    
     l1_reg = 0.0f;
     l2_reg = 0.0f;
     rho = 0.0f;

     admm_max_iter = 5;
     train_data = "NULL";
     pred_out = "pred.txt";
     model_in = "NULL";
     model_out = "lr_model.dat";

     primal = nullptr;
     dual = nullptr;
     cons = nullptr;
     w = nullptr;
     cons_pre = nullptr;
   }

   virtual ~ADMM() {
     if(primal != nullptr)
       delete [] primal;
     if(dual != nullptr)
       delete [] dual;
     if(cons != nullptr)
       delete [] cons;
     if(cons_pre != nullptr)
       delete [] cons_pre;
     if(w != nullptr)
       delete [] w;

     if(optimizer != nullptr)
       delete optimizer;
   }

   //init
   inline void Init() {
     CHECK(num_fea != 0 || learner != "NULL" || train_data != "NULL") 
           << "num_fea and name_leaner must be set!";
      //init paramter
     primal = new float[num_fea];
     dual = new float[num_fea];
     cons = new float[num_fea];
     cons_pre = new float[num_fea];
     w = new float[num_fea];
     traffic = new int[2*(num_procs-1)*num_procs];
     total_traffic = new int[2*(num_procs-1)*num_procs];
     average_traffic = new float[2*(num_procs-1)*num_procs];

     memset(primal,0.0,sizeof(float) * num_fea);
     memset(dual,0.0,sizeof(float) * num_fea);
     memset(cons,0.0,sizeof(float) * num_fea);
   }
 
   void Configure(
      std::vector<std::pair<std::string,std::string> > cfg)
   {
      for(const auto &kv : cfg) {
        cfg_[kv.first] = kv.second;
      }

      if(cfg_.count("rank"))
        rank = static_cast<int>(atoi(cfg_["rank"].c_str()));
      if(cfg_.count("num_fea"))
        num_fea = static_cast<size_t>(atoi(cfg_["num_fea"].c_str()));
      if(cfg_.count("admm_max_iter"))
        admm_max_iter = static_cast<int>(atoi(cfg_["admm_max_iter"].c_str()));
      if(cfg_.count("l1_reg"))
        l1_reg = static_cast<float>(atof(cfg_["l1_reg"].c_str()));
      if(cfg_.count("l2_reg"))
        l2_reg = static_cast<float>(atof(cfg_["l2_reg"].c_str()));
      if(cfg_.count("rho"))
        rho = static_cast<float>(atof(cfg_["rho"].c_str()));
      if(cfg_.count("num_procs"))
        num_procs = static_cast<int>(atoi(cfg_["num_procs"].c_str()));
      if(cfg_.count("num_data"))
        num_data = static_cast<uint32_t>(atoi(cfg_["num_data"].c_str()));
      if(cfg_.count("learner"))
        learner = cfg_["learner"];
      if(cfg_.count("train_data"))
        train_data = cfg_["train_data"];
      if(cfg_.count("pred_data"))
	pred_data = cfg_["pred_data"];
      if(cfg_.count("pred_out"))
        pred_out = cfg_["pred_out"];
      if(cfg_.count("model_out"))
        model_out = cfg_["model_out"];
      if(cfg_.count("model_in"))
        model_in = cfg_["model_in"];

      if (rank == 0)
        LOG(INFO) << "num_fea=" << num_fea << ",num_data=" << num_data
                  << ",l1_reg=" << l1_reg << ",l2_reg=" << l2_reg 
                  << ",learner=" << learner << ",admm_max_iter=" << admm_max_iter
                  << ",rho=" << rho << ",train_data=" << train_data;

      //optimizer
      optimizer = Learner::Create(learner.c_str());
      if(optimizer == nullptr)
        LOG(FATAL) << "learner inital error!";
      optimizer->Configure(cfg);
   }

   void UpdatePrimal()
   {
      if(learner == "lbfgs")
      {
	  optimizer->Train(primal, dual, cons, rho, dtrain);
      } else if(learner == "sparse_lbfgs")
      {
	  optimizer->Train(primal, dual, cons, rho, dtrain);
	  related_num = optimizer->related_num_;
	  need_send_idx = new int[num_fea];
	  for(uint32_t i = 0; i < num_fea; i ++)
	      need_send_idx[i] = optimizer->need_send_idx[i];
	  max_related_num = 0;
	  for(uint32_t i = 0; i < num_fea; i ++)
	      if(related_num[i] != 0)
		  max_related_num ++;

	  need_update_idx = optimizer->need_update_idx; 
      }
   }

   //update dual parameter y
   void UpdateDual() {
     for(uint32_t i = 0;i < num_fea;++i) {
       dual[i] += rho * (primal[i] - cons[i]);
     }
   }

   void UpdateConsensus_no_sparse(){
    float s = 1. / (rho * num_procs + 2 * l2_reg);
    float kappa = s * l1_reg;
    for(uint32_t i = 0; i < num_fea; i++)
    {
	w[i] = s * (rho * primal[i] + dual[i]);
	cons_pre[i] = cons[i];
    }
    double start_comm, end_comm;
    start_comm = MPI_Wtime();
    //sparse_split_scannter(w, num_procs, rank, num_fea);
    MPI_Allreduce(w, cons,  num_fea, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    end_comm = MPI_Wtime();
    //for(uint32_t i = 0; i < num_fea; i++)LOG(INFO) << "myid" << myid << "111";
//	cons[i] = w[i];
    comm_time +=(end_comm - start_comm);
    if(l1_reg != 0.0f)
	SoftThreshold(kappa, cons);
   }

   //update z
   void UpdateConsensus(){
     memset(w,0.0,sizeof(float) * num_fea);
     uint32_t nnz = 0;
     sparse_in_gather = 0;
     sparse_in_reduce = 0;
     if(learner == "lbfgs")
     {
	float s = 1. / (rho * num_procs + 2 * l2_reg);
	float kappa = s * l1_reg;
	for(uint32_t i = 0;i < num_fea;i++)
	{
	    w[i] =  (rho * primal[i] + dual[i]);
	    if(w[i] != 0)
		nnz++;
	    cons_pre[i] = cons[i];
	}
	sparseRatio = nnz;
	double start_comm,end_comm;
	MPI_Barrier(MPI_COMM_WORLD);
	start_comm = MPI_Wtime();
	//SparseRingAllreduce<spar::SumOperator, float>(w, num_fea, rank, num_procs,  MPI_COMM_WORLD);
	RingAllreduce<spar::SumOperator, float>(w, num_fea, rank, num_procs, MPI_COMM_WORLD);
	//MPI_Allreduce(w, cons, num_fea, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
	end_comm = MPI_Wtime();
	for(uint32_t i=0; i < num_fea; i ++)
	    cons[i] = s * w[i];
	//comm_time +=(end_comm - start_comm);
	if(l1_reg != 0.0f)
	    SoftThreshold(kappa,cons);
    } else if(learner == "sparse_lbfgs")
     {
	 float kappa= l1_reg / (rho * num_procs + 2 * l2_reg);
	 for(uint32_t i = 0;i < num_fea; i ++)
	 {
	     cons_pre[i] = cons[i];
	 }
	 for(auto idx : need_update_idx)
	 {
	     w[idx] = (rho * primal[idx] + dual[idx]);
	     if(w[idx] != 0)
		 nnz ++;
	 }
	 double start_comm, end_comm;
	 MPI_Barrier(MPI_COMM_WORLD);
	 start_comm = MPI_Wtime();
	 memset(traffic, 0,sizeof(int) * 2 * (num_procs-1)*num_procs);
	 memset(total_traffic, 0, sizeof(int) * 2 * (num_procs-1)*num_procs);
	 SparseRingAllreduce<spar::SumOperator, float>(w, num_fea, rank, num_procs, need_send_idx,traffic,10, 0, sparse_in_reduce, sparse_in_gather, MPI_COMM_WORLD);
	 MPI_Allreduce(traffic, total_traffic, 2*(num_procs-1)*num_procs, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	 for(int i = 0; i < num_procs;i ++)
	     for(int j = 0; j < 2*(num_procs-1); j++)
		 average_traffic[j]+= total_traffic[i*2*(num_procs-1)+j];
	 //for(int i = 0; i < 2*(num_procs-1); i++)
	//	average_traffic[i] += total_traffic[i];
	 //RingAllreduce<spar::SumOperator, float>(w, num_fea, rank, num_procs, MPI_COMM_WORLD);
	 //MPI_Allreduce(w, cons, num_fea, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
	 //RingAllreduce(w, num_procs, rank, num_fea);
	 end_comm = MPI_Wtime();
	 for(uint32_t i = 0; i < num_fea; i ++)
		cons[i] = w[i]/(rho * num_procs + 2 * l2_reg);
	 comm_time += (end_comm - start_comm);
	 if(l1_reg != 0.0f)
	     SoftThreshold(kappa, cons);
     }
   }
   //soft threshold
   void SoftThreshold(float t_,float *z) {
     for(uint32_t i = 0;i < num_fea;i++){
       if(z[i] > t_){
         z[i] -= t_;
       }else if(z[i] <= t_ && z[i] >= -t_){
         z[i] = 0.0;
       }else{
         z[i] += t_;
       }
     }
   }

   //train task
   void TaskTrain() {
     int iter = 0;
     double start_time, end_time, start_cal, end_cal, local_cal_time;
     double max_cal,min_cal,single_cal,average_cal,average_wait;
     cal_time = 0;
     comm_time = 0;
     memset(average_traffic,0.0,sizeof(float)*2*(num_procs-1)*num_procs);
     start_time = MPI_Wtime();
     if(rank == 0)
      printf("%3s %10s %10s %11s %11s %7s %13s %10s %12s %10s %10s %10s\n", 
        "#", "pri_res", "esp_pri", "dual_res", "esp_dual", "loss", "total_time", "cal_time", "comm_time","sparseRatio","sparse_in_reduce","sparse_in_gather");
     while(iter < admm_max_iter) {
       start_cal = MPI_Wtime();
       this->UpdatePrimal();
       end_cal = MPI_Wtime();
       this->UpdateConsensus();
       //this->UpdateConsensus_no_sparse();
       this->UpdateDual();
       end_time =  MPI_Wtime();
       //cal_time += (end_cal -start_cal);
       single_cal = end_cal - start_cal;
       MPI_Allreduce(&single_cal, &max_cal, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
       MPI_Allreduce(&single_cal, &average_cal, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
       average_cal /= num_procs;
       average_wait = max_cal - average_cal;
       cal_time += average_cal;
       comm_time += average_wait;
       //MPI_Allreduce(&comm_time,&average_comm_time, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
       //average_comm_time/=num_procs;
       total_time = cal_time + comm_time;
       if(rank == 0)
        printf("%3d", iter);
       IsStop(iter);
         //break;
       iter++;
     }
     if(rank == 0)
        TaskPred();
        //SaveModel();
     if(rank == 0)
     {
	 printf("comm_traffic:\n");
	 for(int i = 0; i < 2*(num_procs-1); i ++)
	     printf("%f\n", average_traffic[i]/num_procs);
	printf("max_related_num=%d", max_related_num);
	
     }
   }

   float Eva()
   {
     float sum = 0;
     dtrain->BeforeFirst();
     while(dtrain->Next())
     {
       const dmlc::RowBlock<unsigned> &batch = dtrain->Value();
       for(size_t i = 0;i < batch.size;++i)
        {
          dmlc::Row<unsigned> v = batch[i];
          if(v.get_label() == 0.0)
            sum += std::log(1 - optimizer->PredIns(v, cons));
          else
            sum += std::log(optimizer->PredIns(v, cons));
        }
      }
    return -sum;
   }

   //predict task
   void TaskPred() {
     //pred_data = "./data/kddb/shitu_test_0";
     dmlc::RowBlockIter<unsigned> *dtest
        = dmlc::RowBlockIter<unsigned>::Create
        (pred_data.c_str(),
          0,
          1,
          "libsvm");
     pair_vec.clear();
      dtest->BeforeFirst();
      while(dtest->Next()) {
        const dmlc::RowBlock<unsigned> &batch = dtest->Value();
        for(size_t i = 0;i < batch.size;i++) {
          dmlc::Row<unsigned> v = batch[i];
          float score = optimizer->PredIns(v,cons);
          Metric::pair_t p(score,v.get_label());
          pair_vec.push_back(p);
        }
      }
      LOG(INFO) << "Test AUC=" << Metric::CalAUC(pair_vec) 
                << ",COPC=" << Metric::CalCOPC(pair_vec)
                << ",LogLoss=" << Metric::CalLogLoss(pair_vec)
                << ",RMSE=" << Metric::CalMSE(pair_vec)
                << ",MAE=" << Metric::CalMAE(pair_vec)
                << ",ACC=" <<Metric::CalAcc(pair_vec);
      printf("Test AUC=%.5f ,COPC=%.5f ,LogLoss=%.5f ,RMSE=%.5f ,MAE=%.5f, ACC=%.5f", 
                                  Metric::CalAUC(pair_vec)
                                  ,Metric::CalCOPC(pair_vec)
                                  ,Metric::CalLogLoss(pair_vec)
                                  ,Metric::CalMSE(pair_vec)
                                  ,Metric::CalMAE(pair_vec)
                                  ,Metric::CalAcc(pair_vec));
 
      std::ofstream os(pred_out.c_str());
      for(size_t i = 0;i < pair_vec.size();i++)
        os << pair_vec[i].t_label << " " << pair_vec[i].score << std::endl;
      os.close();
   }

   //save model
   void SaveModel() {
     std::ofstream os(model_out.c_str());
     for(uint32_t i = 0;i < num_fea;i++)
       os << cons[i] << std::endl;
     os.close();
   }
   
   void LoadModel() {
    std::ifstream is(model_in.c_str());
    for(uint32_t i = 0;i < num_fea;i++)
      is >> cons[i];
    is.close();
   }

   bool IsStop(int iter) {
     float send[4] = {0};
     float recv[4] = {0};
    // if(learner == "lbfgs")
     //{
	for(uint32_t i = 0;i < num_fea;i++)
	{
	    send[0] += (primal[i] - cons[i]) * (primal[i] - cons[i]);
	    send[1] += (primal[i]) * (primal[i]);
	    send[2] += (dual[i]) * (dual[i]);
	}
     //} else if(learner == "sparse_lbfgs")
     //{
	//for(auto idx : need_update_idx)
	//{
	  //  send[0] += (primal[idx] - cons[idx]) * (primal[idx] - cons[idx]);
	   // send[1] += (primal[idx]) * (primal[idx]);
	    //send[2] += (dual[idx]) * (dual[idx]);
	//}
     //}
     send[3] = Eva();

     MPI_Allreduce(send, recv, 4, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

     float prires  = sqrt(recv[0]);  /* sqrt(sum ||r_i||_2^2) */
     float nxstack = sqrt(recv[1]);  /* sqrt(sum ||x_i||_2^2) */
     float nystack = sqrt(recv[2]);  /* sqrt(sum ||y_i||_2^2) */
     
     float zdiff = 0.0;
     float z_squrednorm = 0.0; 

     for(uint32_t i = 0;i < num_fea;i++){
       zdiff += (cons[i] - cons_pre[i]) * (cons[i] - cons_pre[i]);
       z_squrednorm += cons[i] * cons[i];
     }

     float z_norm = sqrt(num_procs) * sqrt(z_squrednorm);
     float dualres = sqrt(num_procs) * rho * sqrt(zdiff); /* ||s^k||_2^2 = N rho^2 ||z - zprev||_2^2 */
     //double vmax = nxstack > z_norm?nxstack:z_norm;

     float eps_pri  = sqrt(num_procs * num_data)*ABSTOL + RELTOL * fmax(nxstack,z_norm);
     float eps_dual = sqrt(num_procs * num_data)*ABSTOL + RELTOL * nystack;
    
     if(rank == 0){
       char buf[500];
       snprintf(buf,500,"iter=%2d,prires=%.5f,eps_pri=%.5f,dualres=%.5f,eps_dual=%.5f,loss=%.5f, total_time=%.5f",iter,prires,eps_pri,dualres,eps_dual,recv[3]/(num_procs*num_data), total_time);
       LOG(INFO) << buf;
       printf("%10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10d %10d\n", prires, eps_pri, dualres, eps_dual, recv[3]/(num_procs*num_data), total_time, cal_time, comm_time,sparseRatio, sparse_in_reduce, sparse_in_gather);
     }
     if(prires <= eps_pri && dualres <= eps_dual) {
       return true;
     }

     return false;
   }

  private:
   //training data
   std::string train_data;
   std::string pred_out;
   std::string pred_data;
   dmlc::RowBlockIter<unsigned> *dtrain;
   dmlc::RowBlockIter<unsigned> *dtest;
  

   //admm parameter
   float *primal,*dual,*cons;
   float *w,*cons_pre;

   size_t num_fea;
   uint32_t num_data;
   int admm_max_iter;

   float rho;
   float l1_reg;
   float l2_reg;
   double total_time;
   double cal_time;
   double comm_time;
   double average_comm_time;
   double sparseRatio = 0.0;
   float error;
   int *traffic;
   int *total_traffic;
   float *average_traffic;

   int rank; 
   int num_procs;
   int max_related_num=0;
   int sparse_in_gather;
   int sparse_in_reduce;

   //optimizer 
   std::string learner;
   Learner *optimizer;
   std::vector<Metric::pair_t> pair_vec;
   //configure
   std::map<std::string,std::string> cfg_;

   std::string model_in;
   std::string model_out;
   std::vector<int> related_num;
   std::vector<int> worker_list;
   std::vector<unsigned> need_update_idx;
   int *need_send_idx;
};// class ADMM

}// namespace adPredictAlgo

#endif
