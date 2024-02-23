//
// Created by cluster on 2020/10/14.
//

#ifndef GR_ADMM_PROB_H
#define GR_ADMM_PROB_H


#include<iostream>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include<fstream>
#include<sstream>
#include<string>
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

using namespace std;
struct feature_node
{
    int index;
    double value;
};

class problem
{
public:
    problem(const char *filename);
    void read_problem(const char *filename);
    void read_solution(const char *solution);
    char* readline(FILE *input);
    void exit_input_error(int line_num);

    int l, n; //l为数据的行数  n为最大的列数
    double *y;   //y存标签
    struct feature_node **x; //*x 存的是xspace
    struct feature_node *x_space;//存储的是数据  index 下标记 value 值   可以看作是每次存储用的暂存每行数据的temp
    double bias;            /* < 0 if no bias term */
    char *line;   //读取每行数据用的 temp
    int max_line_len;  //默认使用1024 初始估计一行有多少个数据，因为可能一行比一行大，所以放这里
};


#endif //GR_ADMM_PROB_H
