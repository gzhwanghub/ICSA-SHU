#ifndef UTILS_PROPERTIES_H
#define UTILS_PROPERTIES_H

#include <map>
#include <string>

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
class args_t
{
public:
    args_t(int rank, int size,Properties properties);
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
#endif
