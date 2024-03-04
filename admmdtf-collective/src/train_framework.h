#include <map>
#include <mpi.h>
#include <iostream>
#include <stdlib.h>
#include <ctype.h>
#include <errno.h>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#define Malloc(type, n) (type *)malloc((n)*sizeof(type))
#define LEFT 0
#define RIGHT 1
#define UP 2
#define DOWN 3
using namespace std;

struct feature_node {
    int index;
    double value;
};

//class base_math
//{
//public:
//    static void swap(double& x, double& y) { double t=x; x=y; y=t; }
//	static void swap(int& x, int& y) { int t=x; x=y; y=t; }
//	static double min(double x,double y) { return (x<y)?x:y; }
//	static double max(double x,double y) { return (x>y)?x:y; }
//};

class Function
{
public:
    virtual double fun(double *w,double *y, double *z) = 0 ;
    virtual void grad(double *w, double *g,double *y, double *z) = 0 ;
    virtual void batch_grad(double *g,double *w,double *y,double *z,vector<int> &batch_val)=0;
    virtual void Hv(double *s, double *Hs) = 0 ;
    virtual int get_nr_variable(void) = 0 ;
    virtual int get_number(void) = 0 ;
    virtual void get_diagH(double *M) = 0 ;
    virtual ~Function(void){}
};

class sparse_operator
{
public:
//	static double nrm2_sq(const feature_node *x)
//	{
//		double ret = 0;
//		while(x->index != -1)
//		{
//			ret += x->value*x->value;
//			x++;
//		}
//		return (ret);
//	}

	static double dot(const double *s, const feature_node *x)
	{
		double ret = 0;
		while(x->index != -1)
		{
			ret += s[x->index-1]*x->value;
			x++;
		}
		return (ret);
	}

	static void axpy(const double a, const feature_node *x, double *y)
	{
		while(x->index != -1)
		{
			y[x->index-1] += a*x->value;
			x++;
		}
	}
};

std::string &LeftTrim(std::string &s);

std::string &RightTrim(std::string &s);

std::string &Trim(std::string &s);

template<typename Target, typename Source, bool Same>
class Converter {
public:
    static Target Convert(const Source &arg) {
        Target ret;
        std::stringstream ss;
        if (!(ss << arg && ss >> ret && ss.eof())) {
            printf("类型转换失败");
        }
        return ret;
    }
};

template<typename Target, typename Source>
class Converter<Target, Source, true> {
public:
    static Target Convert(const Source &arg) {
        return arg;
    }
};

template<typename Source>
class Converter<std::string, Source, false> {
public:
    static std::string Convert(const Source &arg) {
        std::ostringstream ss;
        ss << arg;
        return ss.str();
    }
};

template<typename Target>
class Converter<Target, std::string, false> {
public:
    static Target Convert(const std::string &arg) {
        Target ret;
        std::istringstream ss(arg);
        if (!(ss >> ret && ss.eof())) {
            printf("类型转换失败");
        }
        return ret;
    }
};

template<typename T1, typename T2>
struct IsSame {
    static const bool value = false;
};

template<typename T>
struct IsSame<T, T> {
    static const bool value = true;
};

template<typename Target, typename Source>
Target Convert(const Source &arg) {
    return Converter<Target, Source, IsSame<Target, Source>::value>::Convert(arg);
}

class Properties {
public:
    Properties(int &argc, char **&argv);

    Properties(const std::string &path);

    std::string GetString(const std::string &property_name);

    int GetInt(const std::string &property_name);

    double GetDouble(const std::string &property_name);

    bool GetBool(const std::string &property_name);

    bool HasProperty(const std::string &property_name);

    void CheckProperty(const std::string &property_name);

    void Print();

private:
    std::map<std::string, std::string> properties_;

    void ParseFromFile(const std::string &path);
};

class args_t {
public:
    args_t(int rank, int size, Properties properties);

    //process param
    int myid;
    int procnum;
    int worker_per_group_;
    int sqrt_procnum_, sqrt_leader_, leader_num_;
    //async-model-param
    int max_delay;
    int min_barrier;
    //admm
    double rho;
    int max_iterations;
    double l1reg;
    double l2reg;
    double ABSTOL;
    double RELTOL;
    //group admm
    int group_count;
    int group_type;
    //hybrid admm
    int thread_num;
    //sparse admm
    int filter_type;
};


struct meta
{
    int key;
    double value;
};

class problem {
public:
    problem() = default;

    problem(const char *filename);

    void read_problem(const char *filename);

    char *readline(FILE *input);

    void exit_input_error(int line_num);

    int l, n;
    double *y, *sol;
    struct feature_node **x;
    struct feature_node *x_space;
    double bias;            /* < 0 if no bias term */
    char *line;
    int max_line_len;
};

class TRON
{
public:
	TRON(const Function *fun_obj, double eps = 0.1, double eps_cg = 0.1, int max_iter = 1000);
	~TRON();
	void tron(double *w, double *y, double*z, double *statistical_time, int &tron_iteration, int *cg_iter_array);
	void set_print_string(void (*i_print) (const char *buf));

private:
	//int trcg(double delta, double *g, double *s, double *r, bool *reach_boundary);
	int trpcg(double delta, double *g, double *M, double *s, double *r, bool *reach_boundary, double &hassian_time);
	double norm_inf(int n, double *x);

	double eps;
	double eps_cg;
	int max_iter;
	Function *fun_obj;
	void info(const char *fmt,...);
	void (*tron_print_string)(const char *buf);
	double gnorm0;
	int n;
	double *s, *r, *g;


};

class l2r_lr_fun: public Function
{
public:
    l2r_lr_fun(const problem *prob, double C);
    ~l2r_lr_fun();

    double fun(double *w,double *y, double *z);
    void grad(double *w, double *g,double *y, double *z);
    void batch_grad(double *g,double *w,double *y,double *z,vector<int> &batch_val);
    void Hv(double *s, double *Hs);

    int get_nr_variable(void);
    int get_number();
    void get_diagH(double *M);

private:
    void Xv(double *v, double *Xv);
    void XTv(double *v, double *XTv);
    void one_grad(double *g,double *w,int data_index);

    double C;
    double *z;
    double *D;
    const problem *prob;
};

class Collective {
public:
    Collective(args_t *args, problem *problem);

//    ~Collective();

    void RingAllreduce(double *data, int count, MPI_Comm communicator);

    void TorusAllreduce(double *data, int worker_per_group, int group_count, MPI_Comm communicator, int *nbrs);

    void HierarchicalTorus(double *data, MPI_Comm &MAINGRP_COMM, MPI_Comm &SUBGRP_COMM, int *nbrs);

    void HierarchicalAllreduce(double *data, MPI_Comm &MAINGRP_COMM, MPI_Comm &SUBGRP_COMM);

    void CreateTorus(MPI_Comm OLD_COMM, MPI_Comm &TORUS_COMM, int worker_per_group, int main_size, int *nbrs);

private:
    int comm_rank_, comm_size_, dim_;
    MPI_Datatype datatype_;
    int sqrt_procnum_, sqrt_leader_;
    int worker_per_group_;
//    MPI_Comm SUBGRP_COMM_;
//    MPI_Comm MAINGRP_COMM_;
};

class ADMM {
public:
    ADMM(args_t *args, problem *prob, string test_file_path, Collective *collective);

    //  ADMM(args_t *args, problem *prob,int type);
    ~ADMM();

    void x_update();

    void y_update();

    void z_update();

    bool is_stop();

    void softThreshold(double t, double *z);

    void train();

    double predict(int last_iter);

    double GetObjectValue();

    void subproblem_tron();

    void CreateGroup();

    double GetSVMObjectValue(int type);

//	void subproblem_lbfgs();
//	void subproblem_gd();
    void draw();
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
//    LBFGS_OPT *lbfgs_obj_;
//    GD_OPT *gd_obj_;
    Function *fun_obj_, *pred_obj_;
    string outfile_;
    Collective *collective_;
    // int th_num;
    double *y_pre_;
    int rho_flag_;
    double costFunction_, preFunction_;
    double tau_sum_;
    double TAU;
    int power_n_;
    double *s_sendBuf_;
    int filter_type_;
    int max_iterations_;
    int tron_iteraton_;
    int worker_per_group_;
    int group_num_;
    int *cg_iter_;
    MPI_Comm MAINGRP_COMM_, SUBGRP_COMM_, TORUS_COMM_;
    MPI_Group main_group_, world_group_;
    int *maingrp_rank_;
    int nbrs_[4];
};

void test_main(MPI_Comm comm);

void test_main3(MPI_Comm comm);




