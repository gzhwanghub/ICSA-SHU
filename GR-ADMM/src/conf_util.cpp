//
// Created by cluster on 2020/10/14.
//

#include "../include/conf_util.h"
#include<iostream>
#include<fstream>
#include<sstream>
#include<vector>
#include <cstring>

using namespace std;

//构造函数 读取配置文件
conf_util::conf_util() {
    string conf_file = "/mirror/wgz/hx/GR_ADMM/group_admm.conf";
    parse(conf_file);
}

/// 读取配置文件
void conf_util::parse(const string &conf_file) {
    ifstream confIn(conf_file.c_str());
    string line;
    vector<string> vitems;
    while (getline(confIn, line)) {
        vitems.clear();
        if (line.empty() || line[0] == '#')
            continue;
        const int len = line.length();
        char s[len + 1];
        strcpy(s, line.c_str());
        char *pos = strtok(s, " =");
        int32_t k = 0;
        while (pos != NULL) {
            vitems.push_back(pos);
            pos = strtok(NULL, "=");
            k++;
        }
        if (k != 2) {
            cout << "args conf error!" << endl;
            exit(0);
        }
        conf_items.insert({vitems[0], vitems[1]});
    }
}

/// 返回参数
/// \tparam T
/// \param item_name
/// \return
template<class T>
T conf_util::getItem(const std::string &item_name) {
    stringstream sitem;
    T result;
    sitem << conf_items[item_name];
    sitem >> result;
    return result;
}

args_t::args_t(int rank, int size) {
    myid = rank;
    procnum = size;
    rho = 0.0;
}

void args_t::get_args() {
    conf_util admm_conf;
    rho = admm_conf.getItem<double>("rho");
}

void args_t::print_args() {
    cout << "#************************configuration***********************";
    cout << endl;
    cout << "#Number of processors: " << procnum << endl;
    cout << "#Train data: " << train_data_path << endl;
    cout << "#Test data: " << test_data_path << endl;
    cout << "#Max iteration: " << maxIteration << endl;
    cout << "#Update_method: " << Update_method << endl;
    cout << "#Comm_method: " << Comm_method << endl;
    cout << "#Node of per Group: " << nodesOfGroup << endl;
    cout << "#rho: " << rho << endl;
    cout << "#************************************************************";
    cout << endl;
}