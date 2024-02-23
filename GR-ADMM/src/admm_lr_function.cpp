//
// Created by cluster on 2020/10/14.
//

#include "../include/admm_lr_function.h"
#include "../include/simple_algebra.h"
//求f(x)
double AdmmLRFunction::Evaluate(const double *x) {
    int numsofneighbors=nears->neighborsNums-1;
    double sum = 0.0;
    for (int i = 0; i < dimension_; ++i) {
        sum+=rho_*numsofneighbors*x[i]*x[i]+x[i]*sumX[i];
    }
    int sample_num = dataset_->GetSampleNumber();
    for (int i = 0; i < sample_num; ++i) {
        sum += std::log(1 + exp(-dataset_->GetLabel(i) * Dot(x, dataset_->GetSample(i))));
    }
    return sum;
}

//求g（x）
void AdmmLRFunction::Gradient(const double *x, double *g) {
    int numsofneighbors=nears->neighborsNums-1;
    for (int i = 0; i < dimension_; ++i) {
        g[i]=2*rho_*numsofneighbors*x[i]+sumX[i];
    }
    int sample_num = dataset_->GetSampleNumber();
    for (int i = 0; i < sample_num; ++i) {
        double temp = Sigmoid(dataset_->GetLabel(i) * Dot(x, dataset_->GetSample(i)));
        temp = (temp - 1) * dataset_->GetLabel(i);
        const Feature *sample = dataset_->GetSample(i);
        while (sample->index != -1) {
            g[sample->index] += (sample->value * temp);
            ++sample;
        }
    }
}