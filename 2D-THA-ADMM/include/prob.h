#ifndef PROBLEM_H
#define PROBLEM_H

#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <fstream>
#include <sstream>
#include <string>
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
    problem()=default;
    problem(const char *filename);
    void read_problem(const char *filename);
    void read_solution(const char *solution);
    char* readline(FILE *input);
    void exit_input_error(int line_num);
    int l, n;
	double *y, *sol;
	struct feature_node **x;
	struct feature_node *x_space;
	double bias;            /* < 0 if no bias term */
	char *line;
	int max_line_len;
};
#endif // PROBLEM_H
