/*************************************************************************
    > File Name: CoefficientMatrix.cpp
    > Author: gzhwang
    > Mail: gzh.wang@outlook.com
    > Function: Read coefficient matrix
    > Created Time: 2023-08-12
 ************************************************************************/

#include "include/coefficient_matrix.h"
#include <iostream>

using namespace std;

CoefficientMatrix::CoefficientMatrix(const char *filename, const char *filenameb, int myid) {
    this->filename_a_ = filename;
    this->filename_b_ = filenameb;
    Init();
    Matrix datamat(datanum_, dim_, 0);
    data_mat_ = datamat;
    label_.resize(datanum_, 0);
    solution_.resize(dim_, 0);
    myid_ = myid;
    LoadData();
}

void CoefficientMatrix::Init() {
    stringstream ss;
    double val;
    ifstream in(filename_a_);
    if (in) {
        string line;
        int m = 0, n = 0;
        while (getline(in, line)) {
            if (m == 0) {
                ss.clear();
                ss << line;
                while (ss >> val)
                    n++;
            }
            m++;
        }
        datanum_ = m;
        dim_ = n;
    } else
        cerr << filename_a_ << " doesn't exist!";
}

void CoefficientMatrix::LoadData() {
    stringstream ss;
    double val;
    ifstream matIn(filename_a_);
    ifstream bIn(filename_b_);
    ifstream solIn("/mirror/wgz/hx/GR_ADMM_v1/data/solution.dat");
    int m = 0, n;
    if (matIn && bIn && solIn) {
        string line;
        while (getline(matIn, line)) {
            n = 0;
            ss.clear();
            ss << line;
            while (ss >> val) {
                data_mat_(m,n) = val;
                if (m == 0)
                    solIn >> solution_[n];
                n++;
            }
            bIn >> label_[m];
            m++;
        }
    } else
        cerr << "file:lineardataA.dat, lineardatab.dat, solution.dat doesn't exist!";
}

int32_t CoefficientMatrix::GetDataNum() {
    return datanum_;
}

int32_t CoefficientMatrix::GetDimention() {
    return dim_;
}