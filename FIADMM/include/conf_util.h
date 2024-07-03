/*************************************************************************
    > File Name: conf_util.h
    > Description: Parameter configuration file parsing
    > Author: Guozheng Wang
    > Mail: gzh.wang@outlook.com
    > Created Time: 2020-10-14
 ************************************************************************/

#ifndef FIADMM_CONF_UTIL_H
#define FIADMM_CONF_UTIL_H


#include <iostream>
#include <map>
#include <string>
#include <string.h>

using namespace std;

class conf_util
{
public:
    conf_util();
    void parse(const string & conf_file);
    template<class T> T getItem(const string &item_name);
private:
    map<string, string> conf_items;
};

class args_t
{
public:
    args_t(int rank, int size);
    int myid;
    int procnum;
    string train_data_path;
    string test_data_path;
    string data_direction_;
    int Comm_method;
    int maxIteration;
    int nodesOfGroup;
    int Update_method;
    int Repeat_iter;
    // admm
    double rho;
    void get_args();
    void print_args();

};


#endif //FIADMM_CONF_UTIL_H
