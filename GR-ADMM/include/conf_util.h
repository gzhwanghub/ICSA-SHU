//
// Created by cluster on 2020/10/14.
//

#ifndef GR_ADMM_CONF_UTIL_H
#define GR_ADMM_CONF_UTIL_H


#include<iostream>
#include<map>
#include<string>
#include<string.h>

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
    int Comm_method;

    int maxIteration;
    int nodesOfGroup;
    int Update_method;
    int Repeat_iter;

    //admm
    double rho;
    void get_args();
    void print_args();

};


#endif //GR_ADMM_CONF_UTIL_H
