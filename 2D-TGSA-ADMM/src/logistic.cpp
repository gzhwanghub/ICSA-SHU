#include"logistic.h"
#include "math_util.h"

l2r_lr_fun::l2r_lr_fun(const problem *prob, double C) {
    int l = prob->l;

    this->prob = prob;

    z = new double[l];
    D = new double[l];
    this->C = C;
}

l2r_lr_fun::~l2r_lr_fun() {
    delete[] z;
    delete[] D;
}


double l2r_lr_fun::fun(double *w, double *yi, double *zi) {
    int i;
    double f = 0;
    double *y = prob->y;
    int l = prob->l;
    int w_size = get_nr_variable();

    Xv(w, z);

    for (i = 0; i < w_size; i++) {
        double temp = w[i] - zi[i] + yi[i] / C;
        f += temp * temp;
    }
    f *= (C / 2.0);
    for (i = 0; i < l; i++) {
        f += std::log(
                1 + std::exp(-prob->y[i] * z[i]));
    }
    return (f);
}

void l2r_lr_fun::grad(double *w, double *g, double *yi, double *zi) {

    int i;
    double *y = prob->y;
    int l = prob->l;
    int w_size = get_nr_variable();
    for (i = 0; i < l; i++) {
        z[i] = 1 / (1 + exp(-y[i] * z[i]));
        D[i] = z[i] * (1 - z[i]);
        z[i] = (z[i] - 1) * y[i];
    }
    XTv(z, g);

    for (i = 0; i < w_size; i++) {
        g[i] = g[i] + C * (w[i] - zi[i]) + yi[i];
    }
}

int l2r_lr_fun::get_nr_variable(void) {
    return prob->n;
}

int l2r_lr_fun::get_number() {
    return prob->l;
}

void l2r_lr_fun::get_diagH(double *M) {
    int i;
    int l = prob->l;
    int w_size = get_nr_variable();
    feature_node **x = prob->x;

    for (i = 0; i < w_size; i++)
        M[i] = 1;

    for (i = 0; i < l; i++) {
        feature_node *s = x[i];
        while (s->index != -1) {
            M[s->index - 1] += s->value * s->value * D[i];
            s++;
        }
    }
}

//迭代法计算hassian矩阵与向量s乘积，Hs:hassian矩阵
void l2r_lr_fun::Hv(double *s, double *Hs) {
    //\nablaf(w)s Hs:\nablaf(w)

    int i;
    int l = prob->l;
    int w_size = get_nr_variable();
    feature_node **x = prob->x;

    for (i = 0; i < w_size; i++)
        Hs[i] = 0;
    for (i = 0; i < l; i++) {
        feature_node *const xi = x[i];
        double xTs = sparse_operator::dot(s, xi);

        xTs = D[i] * xTs;

        sparse_operator::axpy(xTs, xi, Hs);//X^TDX
    }
    for (i = 0; i < w_size; i++)
        //s + CX^T(D(Xs))
        Hs[i] = C * s[i] + Hs[i];
}

void l2r_lr_fun::Xv(double *v, double *Xv) {
    int i;
    int l = prob->l;
    feature_node **x = prob->x;

    for (i = 0; i < l; i++)
        Xv[i] = sparse_operator::dot(v, x[i]);
}

void l2r_lr_fun::XTv(double *v, double *XTv) {
    int i;
    int l = prob->l;
    int w_size = get_nr_variable();
    feature_node **x = prob->x;

    for (i = 0; i < w_size; i++)
        XTv[i] = 0;
    for (i = 0; i < l; i++)
        sparse_operator::axpy(v[i], x[i], XTv);
}

void l2r_lr_fun::one_grad(double *g, double *w, int data_index) {
    //1.求h(xi)
    feature_node **x = prob->x;
    double *y = prob->y;
    double zi = sparse_operator::dot(w, x[data_index]);//w向量与x[data_index]点乘
    zi = 1 / (1 + exp(-y[data_index] * zi));
    zi = (zi - 1) * y[data_index];
    sparse_operator::axpy(zi, x[data_index], g);//zi标量与xx[data_index]向量相乘
}

void l2r_lr_fun::batch_grad(double *g, double *w, double *y, double *z, vector<int> &batch_vals) {
    for (int j = 0; j < get_nr_variable(); ++j) {
        g[j] = 0.0;
    }
    int i;
    vector<int>::iterator batch_begin = batch_vals.begin(), batch_end;
    feature_node **x = prob->x;
    double *label = prob->y;
    for (; batch_begin != batch_vals.end(); batch_begin++) {
        //one_grad(g,w,*batch_begin);
        int index = *batch_begin;
        //double zi=sparse_operator::dot(w,x[index]);
        //zi=1/(1+exp(-label[index]*zi));
        //zi=(zi-1)*label[index];
        z[index] = 1 / (1 + exp(-label[index] * z[index]));
        z[index] = (z[index] - 1) * label[index];
        sparse_operator::axpy(z[index], x[index], g);
    }
    for (i = 0; i < get_nr_variable(); ++i) {
        g[i] += (C * (w[i] - z[i]) + y[i]);
    }
}
