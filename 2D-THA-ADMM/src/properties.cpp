/*************************************************************************
    > File Name: properties.cpp
    > Author: wdx   
    > Created Time: 2020-12-12
    > Function: 解析配置文件中参数信息
 ************************************************************************/
#include <sstream>
#include <fstream>
#include <iostream>
#include <math.h>
#include "properties.h"
#include "type_convert.h"
#include "string_util.h"


Properties::Properties(int &argc, char **&argv) {
    // 第一个参数是程序名，因此从i=1开始循环
    // 如果一个命令行参数以-开头，那么我们认为这是一个有效的参数
    // 因此有效参数的格式为 -key1 value1 -key2 value2 other
    // 当遇到不以-开头的命令行参数，停止解析
    int i = 1;
    for (; i < argc && argv[i][0] == '-';) {
        std::string key(argv[i] + 1);
        // 如果命令行参数中有 -file path，那么将会从path读取配置文件
        // 因此file为保留的参数名，用户无法使用
        if (key == "file") {
            ParseFromFile(argv[i + 1]);
        } else {
            properties_[key] = argv[i + 1];
        }
        i += 2;
    }
    // 将剩余的命令行参数挪动位置，并修改argc
    int j = 1, k = i;
    for (; k > 1 && k < argc;) {
        argv[j++] = argv[k++];
    }
    argc -= (i - 1);
}
Properties::Properties(const std::string &path)
{
	ParseFromFile(path);
}
void Properties::ParseFromFile(const std::string &path) {
    std::ifstream reader(path);
    if (reader.fail()) {
        std::cout << "无法打开配置文件，文件名：" << path<<std::endl;
    }

    // 新建一个map临时存放属性值
    std::map<std::string, std::string> temp;
    std::string line;
    while (std::getline(reader, line)) {
        // 每一行中#号后面的内容为注释，因此删去这些内容
        std::size_t pos = line.find_first_of('#');
        if (pos != std::string::npos) {
            line.erase(pos);
        }
        // 除去每一行内容的前后空格
        Trim(line);
        if (line.empty()) {
            continue;
        }
        // 每一行内容的格式为key:value，冒号两边可以有空格
        pos = line.find_first_of(':');
        if (pos == std::string::npos || pos == 0 || pos == line.length() - 1) {
            std::cout << "格式错误，应该为key:value格式，" << line<<std::endl;
        }
        std::string key = line.substr(0, pos);
        std::string value = line.substr(pos + 1);
        Trim(key);
        Trim(value);
        temp[key] = value;
    }
    reader.close();

    //命令行参数的优先程度大于配置文件的参数，因此只把properties中没有的参数复制过去
    for (auto it = temp.begin(); it != temp.end(); ++it) {
        if (properties_.count(it->first) == 0) {
            properties_[it->first] = it->second;
        }
    }
}

std::string Properties::GetString(const std::string &property_name) {
    return properties_.at(property_name);
}

int Properties::GetInt(const std::string &property_name) {
    return Convert<int, std::string>(properties_.at(property_name));
}

double Properties::GetDouble(const std::string &property_name) {
    return Convert<double, std::string>(properties_.at(property_name));
}

bool Properties::GetBool(const std::string &property_name) {
    if (properties_.at(property_name) == "true") {
        return true;
    } else if (properties_.at(property_name) == "false") {
        return false;
    }
    std::cout << property_name << "的值必须为true或者false" << std::endl;
    return false;
}

bool Properties::HasProperty(const std::string &property_name) {
    return properties_.count(property_name) != 0;
}

void Properties::CheckProperty(const std::string &property_name) {
    if (!HasProperty(property_name)) {
        std::cout<<"缺少参数\n";
    }
}

void Properties::Print() {
    std::cout<<"**************************************"<<std::endl;
    for (auto it = properties_.begin(); it != properties_.end(); ++it) {
       std::cout << it->first << ":" << it->second<<std::endl;
    }
    std::cout<<"**************************************"<<std::endl;
}
args_t::args_t(int rank, int size,Properties properties)
{
    myid = rank;
    procnum = size;
    min_barrier = properties.GetInt("min_barrier");
	max_delay = properties.GetInt("max_delay");
	max_iterations = properties.GetInt("max_iterations");
	filter_type= properties.GetInt("filter_type");
	rho = properties.GetDouble("rho");
	l1reg = properties.GetDouble("l1reg");
	l2reg = properties.GetDouble("l2reg");
	ABSTOL = properties.GetDouble("ABSTOL");
	RELTOL = properties.GetDouble("RELTOL");
    worker_per_group_ = properties.GetInt("worker_per_group");
    group_type= properties.GetInt("group_type");
    leader_num_ = procnum / worker_per_group_;
//    if(procnum == (int)sqrt(procnum) * (int)sqrt(procnum)){
//        sqrt_procnum_ = sqrt(procnum);
//    }else{
//        fprintf(stderr,"Unable to perform torus allreduce algorithm！\n");
//    }
    if (leader_num_ == (int) sqrt(leader_num_) * (int) sqrt(leader_num_)) {
        sqrt_leader_ = sqrt(leader_num_);
    }else{
        fprintf(stderr,"Unable to perform hierarchical torus algorithm！\n");
    }
	//thread_num = properties.GetInt("thread_num");
}


