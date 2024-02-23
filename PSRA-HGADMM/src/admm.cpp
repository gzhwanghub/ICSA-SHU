#ifdef _WIN32
#include <time.h>
#else

#include <sys/time.h>

#endif


#include "../include/admm.h"
#include "../include/math/simple_algebra.h"

#include <vector>
#include <chrono>

using namespace std;

double Predict(const double *z, SparseDataset *test_dataset) {
    int counter = 0;
    int sample_num = test_dataset->GetSampleNumber();
    for (int i = 0; i < sample_num; ++i) {
        double temp = 1.0 / (1 + exp(-1 * Dot(z, test_dataset->GetSample(i))));
        if (test_dataset->GetLabel(i) == 1 && temp >= 0.5) {
            ++counter;
        }
        if (test_dataset->GetLabel(i) == -1 && temp < 0.5) {
            ++counter;
        }
    }
    return counter * 100.0 / sample_num;
}

double ObjectiveValue(const double *z, SparseDataset *test_dataset) {
    double sum = 0;
    int sample_num = test_dataset->GetSampleNumber();
    for (int i = 0; i < sample_num; ++i) {
        sum += std::log(1 + std::exp(-test_dataset->GetLabel(i) * Dot(z, test_dataset->GetSample(i))));
    }
    sum = sum / sample_num;
    return sum;
}


ADMM::ADMM(int dimension,  int max_iterations,  double rho,
           double l2reg, double ABSTOL, double RELTOL, std::string train_data_path,
           std::string test_data_path,int repeat_Iter,int Dynamic_Group,int nodes_Group) : dimension_(dimension), max_iterations_(max_iterations), rho_(rho),
                                         l2reg_(l2reg), ABSTOL_(ABSTOL), RELTOL_(RELTOL),
                                         triggered_(false), stopped_(false), k_(0), id_(-1), worker_number_(-1),
                                         calculate_time_(0), optimizer_(NULL), train_data_(NULL), test_data_(NULL),
                                         repeatIter(repeat_Iter),DynamicGroup(Dynamic_Group),nodesOfGroup(nodes_Group),
                                         x_(NULL), y_(NULL), z_(NULL), w_(NULL),
                                         old_x_(NULL), old_y_(NULL), old_z_(NULL), temp_z_(NULL) {
    /*检查当前MPI版本是否支持多线程*/
    spar::Init(NULL, NULL);
    /*如果是辅助节点则创建辅助进程并设置相关参数*/
    if (!spar::IsWorker()) {
        coordinator_ = Coordinator::GetInstance();
        coordinator_->Set_max_iterations(max_iterations);
        coordinator_->Set_repeatIter(repeatIter);
        coordinator_->Set_DynamicGroup(DynamicGroup);
        coordinator_->Set_nodesOfGroup(nodesOfGroup);
    } else {
        /*工作节点需要设置通信进程*/
        communicator_ = Communicator::GetInstance();

        int flag;
        MPI_Initialized(&flag);
        CHECK_EQ(flag, 1) << "请在调用spar::Init()之后启动Communicator";
        MPI_Comm_rank(MPI_COMM_WORLD, &id_);
        MPI_Comm_size(MPI_COMM_WORLD, &worker_number_);
        // Worker的数量为总进程数减一
        --worker_number_;
        CHECK_NE(id_, worker_number_) << "不要在Coordinator节点上启动Communicator";


        x_ = new double[dimension_];
        y_ = new double[dimension_];
        z_ = new double[dimension_];
        w_ = new double[dimension_];
        old_x_ = new double[dimension_];
        old_y_ = new double[dimension_];
        old_z_ = new double[dimension_];
        temp_z_ = new double[dimension_];
        char real_path[50];
        sprintf(real_path, train_data_path.c_str(), id_);
        train_data_ = new SparseDataset(real_path);
        if (id_ == 0) {
            test_data_ = new SparseDataset(test_data_path);
        }

        optimizer_ = new LRTronOptimizer(y_, z_, dimension_, rho_, 5, 1e-4, 0.1, train_data_);
    }
}

ADMM::~ADMM() {
    delete[] x_;
    delete[] y_;
    delete[] z_;
    delete[] w_;
    delete[] old_x_;
    delete[] old_y_;
    delete[] old_z_;
    delete[] temp_z_;
    delete optimizer_;
    delete train_data_;
    delete test_data_;
}

void ADMM::Run() {

    /*ADMM初始化时协助节点的ID初始化为-1*/
    if (id_ == -1) {
        coordinator_->Run();
        MPI_Finalize();
    } else {
        // cout<<"id "<<id_<<endl;
        FillZero(x_, dimension_);
        FillZero(y_, dimension_);
        FillZero(z_, dimension_);
        FillZero(w_, dimension_);
        FillZero(old_x_, dimension_);
        FillZero(old_y_, dimension_);
        FillZero(old_z_, dimension_);
        FillZero(temp_z_, dimension_);
        communicator_->Run();

        /*double cal_start_time, cal_end_time;*/
        timeval cal_start_time, cal_end_time;
        gettimeofday(&start_time_, NULL);

        //start_time_ = MPI_Wtime();
        /*  auto cal_start_time, cal_end_time;*/
        if (id_ == 0) {
            printf("%3s %10s %10s %10s %10s %10s %10s %10s\n", "#", "r_orm", "esp_pri", "s_norm", "esp_dual",
                   "obj_val", "accuracy", "time");
        }

        int leader_id = communicator_->Get_leaderID();
        int my_id = communicator_->Get_MyID();
	int coordinator_id_ = worker_number_;
        while (true) {
            /*gettimeofday(&cal_start_time, NULL);*/
            if (my_id == leader_id) {
                communicator_->Creat_inter_Group(k_+1);
            }

            gettimeofday(&cal_start_time, NULL);

            optimizer_->Optimize(x_);

            for (int i = 0; i < dimension_; ++i) {
                w_[i] = rho_ * x_[i] + y_[i];
            }
            gettimeofday(&cal_end_time, NULL);
            m_.lock();
            calculate_time_ += ((cal_end_time.tv_sec - cal_start_time.tv_sec) +
                                (cal_end_time.tv_usec - cal_start_time.tv_usec) / 1000000.0);
	    //cout<<"calculate_time:"<<calculate_time_<<endl;
            m_.unlock();


            /*对w进行*/
            communicator_->SyncAllreduce<spar::SumOperator>(w_, dimension_);


            PostAsynAllreduce(w_, dimension_, this, my_id);
            //cout<<"communicate_w: "<<id_<<endl;
            for (int i = 0; i < dimension_; ++i) {
                y_[i] += rho_ * (x_[i] - z_[i]);
            }
            //cout<<stopped_<<endl;
            if (stopped_) {
		if(id_==0)
		    MPI_Send(NULL,0,MPI_INT,coordinator_id_,spar::MessageType::kTerminateCommand,MPI_COMM_WORLD);
                break;
            }

        }
       /*  communicator_->WaitUntilCommunicatorStop();*/
        PreTerminate(this);
    }

    spar::Finalize();
}

void ADMM::SupplyAllreduceData(double *buffer, int dimension, void *parameter) {
    ADMM *admm = static_cast<ADMM *>(parameter);
    admm->triggered_ = true;
    for (int i = 0; i < dimension; ++i) {
        buffer[i] = admm->rho_ * admm->old_x_[i] + admm->old_y_[i];
    }
}

void ADMM::PostAsynAllreduce(double *buffer, int dimension, void *parameter, int id) {

    ADMM *admm = static_cast<ADMM *>(parameter);
    ++(admm->k_);
    double *z;

    if (!admm->triggered_) {
        Assign(admm->old_x_, admm->x_, dimension);
        Assign(admm->old_y_, admm->y_, dimension);
        z = admm->z_;
    } else {
        z = admm->temp_z_;
    }
    int worker_number = admm->worker_number_;
    double rho = admm->rho_;
    double l2reg = admm->l2reg_;
    double ABSTOL = admm->ABSTOL_;
    double RELTOL = admm->RELTOL_;
    double temp = 2 * l2reg + worker_number * rho;
    for (int i = 0; i < dimension; ++i) {
        z[i] = buffer[i] / temp;

    }
    double temp_buffer[3] = {0, 0, 0};
    for (int i = 0; i < dimension; ++i) {
        temp_buffer[0] += admm->old_x_[i] * admm->old_x_[i];
        temp_buffer[1] += admm->old_y_[i] * admm->old_y_[i];
        double temp = admm->old_x_[i] - z[i];
        temp_buffer[2] += temp * temp;
    }
    //cout<<"PostAsyn"<<endl;
    admm->communicator_->SyncAllreduce<spar::SumOperator>(temp_buffer, 3);

    double nxstack = sqrt(temp_buffer[0]); /* sqrt(sum ||x_i||_2^2) */
    double nystack = sqrt(temp_buffer[1]); /* sqrt(sum ||y_i||_2^2) */
    double prires = sqrt(temp_buffer[2]); /* sqrt(sum ||r_i||_2^2) */
    double z_diff = 0; /* 存放||z_new - z_old||_2^2 */
    double z_norm = 0; /* 存放||z_new||_2^2 */
    for (int i = 0; i < dimension; ++i) {
        double temp = admm->old_z_[i] - z[i];
        z_diff += temp * temp;
        z_norm += z[i] * z[i];
    }
    double dualres = rho * sqrt(worker_number * z_diff);
    double eps_pri = sqrt(dimension * worker_number) * ABSTOL + RELTOL * fmax(nxstack, sqrt(worker_number * z_norm));
    double eps_dual = sqrt(dimension * worker_number) * ABSTOL + RELTOL * nystack;
    //double temp_ov = ObjectiveValue(z,admm->test_data_);
    if (admm->id_ == 0) {
        /*gettimeofday(&admm->end_time_, NULL);*/
        gettimeofday(&admm->end_time_, NULL);
        double wait_time = (admm->end_time_.tv_sec - admm->start_time_.tv_sec) +
                           (admm->end_time_.tv_usec - admm->start_time_.tv_usec) / 1000000.0;
        double temp_ov = ObjectiveValue(z, admm->test_data_);

        double temp_ac = Predict(z, admm->test_data_);
        printf("%3d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f \n", admm->k_, prires, eps_pri, dualres,
               eps_dual, temp_ov, temp_ac, wait_time);
    }
      if (admm->k_ >= admm->max_iterations_) {

    //if (admm->k_ >= admm->max_iterations_ || (prires <= eps_pri && dualres <= eps_dual)) {
        admm->stopped_ = true;
        //cout<<"true"<<endl;
    }
    Assign(admm->old_z_, z, dimension);
    admm->triggered_ = false;
}

void ADMM::PreTerminate(void *parameter) {
    ADMM *admm = static_cast<ADMM *>(parameter);
    gettimeofday(&admm->end_time_, NULL);
    double total_time = (admm->end_time_.tv_sec - admm->start_time_.tv_sec) +
                        (admm->end_time_.tv_usec - admm->start_time_.tv_usec) / 1000000.0;
    //admm->end_time_ = MPI_Wtime();
    //uint64_t total_time = admm->end_time_ - admm->start_time_;
    double temp[] = {static_cast<double>(total_time), 0};
    admm->m_.lock();
    temp[1] = admm->calculate_time_;
    admm->m_.unlock();
    admm->communicator_->AllReduce<spar::SumOperator>(temp,2);

    if (admm->id_ == 0) {

        std::cout << "总运行时间：" <<temp[0] / admm->worker_number_ << std::endl;
        std::cout << "平均计算时间：" <<temp[1] / admm->worker_number_ << std::endl;
        std::cout << "平均等待时间：" << temp[0] / admm->worker_number_ - temp[1] / admm->worker_number_ << std::endl;
    }
}
