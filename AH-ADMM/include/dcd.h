#ifndef SSP_HIERARCHICAL_PARALLEL_SUBPROBLEM_ADMM_DCD_H
#define SSP_HIERARCHICAL_PARALLEL_SUBPROBLEM_ADMM_DCD_H

#include <omp.h>
#include <cmath>
#include "prob.h"
#include "sparse_operator.h"

typedef signed char schar;
template <class T> static inline void swap(T& x, T& y) { T t=x; x=y; y=t; }
#ifndef min
template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
#endif
#ifndef max
template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif

#undef GETI
#define GETI(i) (y[i]+1)

class DCD{
public:
    DCD(const Problem *prob, double *w, double *u, double *alpha, double *z, double eps, double Cp, double Cn,double rho,int f);

    void solve_proximity_l1l2_svc(const Problem *prob, double *w, double *u, double *alpha, double *z, double eps, double Cp, double Cn, double rho);
    void parallel_solve_proximity_l1l2_svc(const Problem *prob, double *w, double *u, double *alpha, double *z, double eps, double Cp, double Cn, double rho);
    void batch_solve_proximity_l1l2_svc(const Problem *prob, double *w, double *u, double *alpha, double *z, double eps, double Cp, double Cn, double rho);

};

static inline int rand_int(const int max)
{
    static int seed = omp_get_thread_num();
#ifdef CV_OMP
#pragma omp threadprivate(seed)
#endif
    seed = ((seed * 1103515245) + 12345) & 0x7fffffff;
    return seed%max;
}

#endif //SSP_HIERARCHICAL_PARALLEL_SUBPROBLEM_ADMM_DCD_H

