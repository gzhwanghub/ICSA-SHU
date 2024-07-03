/*************************************************************************
    > File Name: CoefficientMatrix.h
    > Description: Read matrix
    > Author: Guozheng Wang
    > Mail: gzh.wang@outlook.com
    > Created Time: 2023-08-12
 ************************************************************************/

#ifndef FIADMM_COEFFICIENTMATRIX_H
#define FIADMM_COEFFICIENTMATRIX_H

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include "include/Vector.h"
#include "include/Matrix.h"
using namespace std;
using namespace comlkit;

class CoefficientMatrix{
public:
    CoefficientMatrix(const char *filename, const char *filenameb, int myid);
    ~CoefficientMatrix();
    void Init();
    void LoadData();
    int GetDataNum();
    int GetDimention();
    Matrix data_mat_;
    Vector label_;
    Vector solution_;
private:
    int datanum_;
    int dim_;
    int myid_;
    const char *filename_a_, *filename_b_;
};


#endif //FIADMM_COEFFICIENTMATRIX_H
