// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
/*
        Jensen: A Convex Optimization And Machine Learning ToolKit
 *	Matrix class
        Author: Rishabh Iyer
 *
 */

#ifndef FIADMM_MATRIX_H
#define FIADMM_MATRIX_H

#include "Vector.h"

namespace comlkit {

class Matrix {
protected:
std::vector<std::vector<double> > matrix;
int m;
int n;
public:
Matrix();
Matrix(int m, int n);
Matrix(int m, int n, int val);
Matrix(int m, int n, bool);
Matrix(const Matrix& M);
double& operator()(const int i, const int j);         // point access
const double& operator()(const int i, const int j) const;         // const point access
Vector& operator[](const int i);         // row access
const Vector& operator[](const int i) const;         // const row access
Vector operator()(const int i) const;         // const column access (read only)
void push_back(const Vector& v);         // add a row at the end
void remove(int i);         // delete a row at position i (i starts from 0)
int numRows() const;
int numColumns() const;
int size() const;
};

}
#endif
