// Copyright (c) 2007-2015 The LIBLINEAR Project.
// Modified for use in Jensen by Rishabh Iyer
/*


 *	Trust Region Newton Method, using Conjugate Gradient at every iteration
        Solves the problem \min_x L(x) + |x|_1, i.e L1 regularized optimization problems.
        This algorithm directly encourages sparsity.
        Author: Rishabh Iyer
 *
        Input:  Continuous Function: c
                        Initial starting point x0
                        Initial step-size (alpha)
                        back-tracking parameter (gamma)
                        max number of function evaluations (maxEvals)
                        Tolerance (TOL)
                        resetAlpha (whether to reset alpha at every iteration or not)
                        verbosity

        Output: Output on convergence (x)
 */

#include <stdio.h>
#include <algorithm>
#include <iostream>
using namespace std;

#include "include/TRON.h"
#include "include/group_strategy.h"
#include <math.h>
namespace comlkit {
    int daxpy_(int *n, double *sa, Vector& sx, int *incx, Vector& sy, int *incy) {
        long int i, m, ix, iy, nn, iincx, iincy;
        register double ssa;

        /* constant times a vector plus a vector.
           uses unrolled loop for increments equal to one.
           jack dongarra, linpack, 3/11/78.
           modified 12/3/93, array(1) declarations changed to array(*) */

        /* Dereference inputs */
        nn = *n;
        ssa = *sa;
        iincx = *incx;
        iincy = *incy;

        if (nn > 0 && ssa != 0.0) {
            if (iincx == 1 && iincy == 1) /* code for both increments equal to 1 */
            {
                m = nn - 3;
                for (i = 0; i < m; i += 4) {
                    sy[i] += ssa * sx[i];
                    sy[i + 1] += ssa * sx[i + 1];
                    sy[i + 2] += ssa * sx[i + 2];
                    sy[i + 3] += ssa * sx[i + 3];
                }
                for (; i < nn; ++i) /* clean-up loop */
                    sy[i] += ssa * sx[i];
            } else /* code for unequal increments or equal increments not equal to 1 */
            {
                ix = iincx >= 0 ? 0 : (1 - nn) * iincx;
                iy = iincy >= 0 ? 0 : (1 - nn) * iincy;
                for (i = 0; i < nn; i++) {
                    sy[iy] += ssa * sx[ix];
                    ix += iincx;
                    iy += iincy;
                }
            }
        }

        return 0;
    }

    double ddot_(int *n, Vector& sx, int *incx, Vector& sy, int *incy) {
        long int i, m, nn, iincx, iincy;
        double stemp;
        long int ix, iy;

        /* forms the dot product of two vectors.
           uses unrolled loops for increments equal to one.
           jack dongarra, linpack, 3/11/78.
           modified 12/3/93, array(1) declarations changed to array(*) */

        /* Dereference inputs */
        nn = *n;
        iincx = *incx;
        iincy = *incy;

        stemp = 0.0;
        if (nn > 0) {
            if (iincx == 1 && iincy == 1) /* code for both increments equal to 1 */
            {
                m = nn - 4;
                for (i = 0; i < m; i += 5)
                    stemp += sx[i] * sy[i] + sx[i + 1] * sy[i + 1] + sx[i + 2] * sy[i + 2] +
                             sx[i + 3] * sy[i + 3] + sx[i + 4] * sy[i + 4];

                for (; i < nn; i++)        /* clean-up loop */
                    stemp += sx[i] * sy[i];
            } else /* code for unequal increments or equal increments not equal to 1 */
            {
                ix = 0;
                iy = 0;
                if (iincx < 0)
                    ix = (1 - nn) * iincx;
                if (iincy < 0)
                    iy = (1 - nn) * iincy;
                for (i = 0; i < nn; i++) {
                    stemp += sx[ix] * sy[iy];
                    ix += iincx;
                    iy += iincy;
                }
            }
        }
        return stemp;
    } /* ddot_ */

    double dnrm2_(int *n, double *x, int *incx) { // 求二范数方法
        long int ix, nn, iincx;
        double norm, scale, absxi, ssq, temp;

/*  DNRM2 returns the euclidean norm of a vector via the function
    name, so that
       DNRM2 := sqrt( x'*x )
    -- This version written on 25-October-1982.
       Modified on 14-October-1993 to inline the call to SLASSQ.
       Sven Hammarling, Nag Ltd.   */
        /* Dereference inputs */

        nn = *n;
        iincx = *incx;
        if (nn > 0 && iincx > 0) {
            if (nn == 1) {
                norm = fabs(x[0]);
            } else {
                scale = 0.0;
                ssq = 1.0;

                /* The following loop is equivalent to this call to the LAPACK
                   auxiliary routine:   CALL SLASSQ( N, X, INCX, SCALE, SSQ ) */

                for (ix = (nn - 1) * iincx; ix >= 0; ix -= iincx) {
                    if (x[ix] != 0.0) {
                        absxi = fabs(x[ix]); // Absolute value function.
                        if (scale < absxi) {
                            temp = scale / absxi;
                            ssq = ssq * (temp * temp) + 1.0;
                            scale = absxi;
                        } else {
                            temp = absxi / scale;
                            ssq += temp * temp;
                        }
                    }
                }
                norm = scale * sqrt(ssq);
            }
        } else
            norm = 0.0;
        return norm;
    } /* dnrm2_ */

    int dscal_(int *n, double *sa, Vector& sx, int *incx) {
        long int i, m, nincx, nn, iincx;
        double ssa;
        /* scales a vector by a constant.
           uses unrolled loops for increment equal to 1.
           jack dongarra, linpack, 3/11/78.
           modified 3/93 to return if incx .le. 0.
           modified 12/3/93, array(1) declarations changed to array(*) */
        /* Dereference inputs */
        nn = *n;
        iincx = *incx;
        ssa = *sa;
        if (nn > 0 && iincx > 0) {
            if (iincx == 1) /* code for increment equal to 1 */
            {
                m = nn - 4;
                for (i = 0; i < m; i += 5) {
                    sx[i] = ssa * sx[i];
                    sx[i + 1] = ssa * sx[i + 1];
                    sx[i + 2] = ssa * sx[i + 2];
                    sx[i + 3] = ssa * sx[i + 3];
                    sx[i + 4] = ssa * sx[i + 4];
                }
                for (; i < nn; ++i) /* clean-up loop */
                    sx[i] = ssa * sx[i];
            } else /* code for increment not equal to 1 */
            {
                nincx = nn * iincx;
                for (i = 0; i < nincx; i += iincx)
                    sx[i] = ssa * sx[i];
            }
        }

        return 0;
    } /* dscal_ */

    double uTMv(int n, Vector& u, Vector& M, Vector& v) {
        const int m = n - 4;
        double res = 0;
        int i;
        for (i = 0; i < m; i += 5)
            res += u[i] * M[i] * v[i] + u[i + 1] * M[i + 1] * v[i + 1] + u[i + 2] * M[i + 2] * v[i + 2] +
                   u[i + 3] * M[i + 3] * v[i + 3] + u[i + 4] * M[i + 4] * v[i + 4];
        for (; i < n; i++)
            res += u[i] * M[i] * v[i];
        return res;
    }


    int trpcg(double delta, const ContinuousFunctions& c, const Vector& x, double cg_epsilon, Vector& g, Vector& M, Vector& s, Vector& r, bool *reach_boundary) {
        int i, inc = 1;
        double one = 1;
        int m = c.size();
        Vector d(m,0);
        Vector Hd(m,0);
        Vector z(m,0);
        double zTr, znewTrnew, alpha, beta, cgtol;
        *reach_boundary = false;
        for (i = 0; i < m; i++) {
            s[i] = 0;
            r[i] = -g[i];
            z[i] = r[i] / M[i];
            d[i] = z[i];
        }
        zTr = z * r;
        cgtol = cg_epsilon * sqrt(zTr);
        int cg_iter = 0;
        int max_cg_iter = std::max(m, 5);
        while (cg_iter < max_cg_iter) {
            if (sqrt(zTr) <= cgtol)
                break;
            cg_iter++;
            c.evalHessianVectorProduct(x, d, Hd);
            double temp_vector_mul = d * Hd;
            alpha = zTr / temp_vector_mul;
            daxpy_(&m, &alpha, d, &inc, s, &inc);
            double sMnorm = sqrt(uTMv(m, s, M, s));
            if (sMnorm > delta) {
                //std::cerr << "cg reaches trust region boundary" << std::endl;
                *reach_boundary = true;
                alpha = -alpha;
                daxpy_(&m, &alpha, d, &inc, s, &inc);
                double sTMd = uTMv(m, s, M, d);
                double sTMs = uTMv(m, s, M, s);
                double dTMd = uTMv(m, d, M, d);
                double dsq = delta * delta;
                double rad = sqrt(sTMd * sTMd + dTMd * (dsq - sTMs));
                if (sTMd >= 0)
                    alpha = (dsq - sTMs) / (sTMd + rad);
                else
                    alpha = (rad - sTMd) / dTMd;
                daxpy_(&m, &alpha, d, &inc, s, &inc);
                alpha = -alpha;
                daxpy_(&m, &alpha, Hd, &inc, r, &inc);
                break;
            }
            alpha = -alpha;
            daxpy_(&m, &alpha, Hd, &inc, r, &inc);
            for (i = 0; i < m; i++)
                z[i] = r[i] / M[i];
            znewTrnew = ddot_(&m, z, &inc, r, &inc);
            beta = znewTrnew / zTr;
            dscal_(&m, &beta, d, &inc);
            daxpy_(&m, &one, z, &inc, d, &inc);
            zTr = znewTrnew;
        }

        if (cg_iter == max_cg_iter){}
            //LOG(WARNING) << "WARNING: reaching maximal number of CG steps";
        return cg_iter;
    }

    double dnrm2_(int *n, Vector& x, int *incx) { // 求二范数方法
        long int ix, nn, iincx;
        double norm, scale, absxi, ssq, temp;

/*  DNRM2 returns the euclidean norm of a vector via the function
    name, so that
       DNRM2 := sqrt( x'*x )
    -- This version written on 25-October-1982.
       Modified on 14-October-1993 to inline the call to SLASSQ.
       Sven Hammarling, Nag Ltd.   */
        /* Dereference inputs */

        nn = *n;
        iincx = *incx;
        if (nn > 0 && iincx > 0) {
            if (nn == 1) {
                norm = fabs(x[0]);
            } else {
                scale = 0.0;
                ssq = 1.0;

                /* The following loop is equivalent to this call to the LAPACK
                   auxiliary routine:   CALL SLASSQ( N, X, INCX, SCALE, SSQ ) */

                for (ix = (nn - 1) * iincx; ix >= 0; ix -= iincx) {
                    if (x[ix] != 0.0) {
                        absxi = fabs(x[ix]); // 绝对值函数
                        if (scale < absxi) {
                            temp = scale / absxi;
                            ssq = ssq * (temp * temp) + 1.0;
                            scale = absxi;
                        } else {
                            temp = absxi / scale;
                            ssq += temp * temp;
                        }
                    }
                }
                norm = scale * sqrt(ssq);
            }
        } else
            norm = 0.0;
        return norm;
    } /* dnrm2_ */

    void get_diag_preconditioner(Vector& M, std::vector<SparseFeature>& features, const ContinuousFunctions& c, Vector& D) {
        int sample_num = c.length();
        for (int i = 0; i < c.size(); i++) {
            M[i] = 1;
        }
        for (int i = 0; i < sample_num; i++) {
            for(int j = 0; j < features[i].featureVec.size(); ++j){
                M[features[i].featureIndex[j]] += features[i].featureVec[j] * features[i].featureVec[j] * D[i];
            }
        }
    }

    Vector TRON(const ContinuousFunctions& c, std::vector<SparseFeature>& features, Vector& y, const Vector& x0, const int maxEval, const double TOL, const double cg_epsilon, int verbosity){
        double eta0 = 1e-4, eta1 = 0.25, eta2 = 0.75;
        // Parameters for updating the trust region size delta. 0 < sigma1 < sigma2 < 1 < sigma3
        double tron_epsilon;
        int m = c.size();
        int n = c.length();
        int pos = 0, neg = 0;
        Vector x(x0);
        for(int i = 0; i < n; ++i){
            if(y[i] > 0){
                ++pos;
            }
        }
        neg = n - pos;
        tron_epsilon = TOL * std::max(std::min(pos, neg), 1) / n;
        double sigma1 = 0.25, sigma2 = 0.5, sigma3 = 4;
        int i;
        double delta = 0, sMnorm, one = 1.0;
        double alpha, f, fnew, prered, actred, gs;
        int search = 1, iter = 1, inc = 1;
        Vector D(n,0);
        Vector s(m,0);
        Vector r(m,0);
        Vector g(m,0);
        const double alpha_pcg = 0.01;
        Vector M(m,0);
        Vector I(n,0);
        Vector z(n,0);
        // calculate gradient norm at x=0 for stopping condition.
        Vector x1(m,0);
        c.eval(x1, f, g);
        double gnorm0 = dnrm2_(&m, g, &inc); // 求梯度2范数
        c.eval(x, f, g);
        double gnorm = dnrm2_(&m, g, &inc); // 求梯度2范数
        if (gnorm <= TOL * gnorm0) // 没有循环，仅仅是看一下初次梯度与迭代后的梯度比较关系
            search = 0;
        get_diag_preconditioner(M, features, c, D);
        for (i = 0; i < m; i++)
            M[i] = (1 - alpha_pcg) + alpha_pcg * M[i];
        delta = sqrt(uTMv(m, g, M, g));
        Vector x_new(m,0);
        bool reach_boundary;
        bool delta_adjusted = false;
        while (iter <= maxEval && search) {
            trpcg(delta, c, x, cg_epsilon, g, M, s, r, &reach_boundary);
            x_new = x;
            daxpy_(&m, &one, s, &inc, x_new, &inc);
            gs = ddot_(&m, g, &inc, s, &inc);
            prered = -0.5 * (gs - ddot_(&m, s, &inc, r, &inc));
            fnew = c.eval(x);
            // Compute the actual reduction.
            actred = f - fnew;
            // On the first iteration, adjust the initial step bound.
            sMnorm = sqrt(uTMv(m, s, M, s));
            if (iter == 1 && !delta_adjusted) {
                delta = std::min(delta, sMnorm);
                delta_adjusted = true;
            }
            // Compute prediction alpha*sMnorm of the step.
            if (fnew - f - gs <= 0)
                alpha = sigma3;
            else
                alpha = std::max(sigma1, -0.5 * (gs / (fnew - f - gs)));
            // Update the trust region bound according to the ratio of actual to predicted reduction.
            if (actred < eta0 * prered)
                delta = std::min(alpha * sMnorm, sigma2 * delta);
            else if (actred < eta1 * prered)
                delta = std::max(sigma1 * delta, std::min(alpha * sMnorm, sigma2 * delta));
            else if (actred < eta2 * prered)
                delta = std::max(sigma1 * delta, std::min(alpha * sMnorm, sigma3 * delta));
            else {
                if (reach_boundary)
                    delta = sigma3 * delta;
                else
                    delta = std::max(delta, std::min(alpha * sMnorm, sigma3 * delta));
            }

            if (actred > eta0 * prered) {
                ++iter;
                x = x_new;
                f = fnew;
                g = c.evalGradient(x);
                get_diag_preconditioner(M, features, c, D);
                for (i = 0; i < m; i++)
                    M[i] = (1 - alpha_pcg) + alpha_pcg * M[i];
                gnorm = dnrm2_(&m, g, &inc);
                if (gnorm <= tron_epsilon * gnorm0)
                    break;
            }
            if (f < -1.0e+32) {
                //LOG(WARNING) << "WARNING: f < -1.0e+32";
                break;
            }
            if (prered <= 0) {
                //LOG(WARNING) << "WARNING: prered <= 0";
                break;
            }
            if (fabs(actred) <= 1.0e-12 * fabs(f) && fabs(prered) <= 1.0e-12 * fabs(f)) {
                //LOG(WARNING) << "WARNING: actred and prered too small";
                break;
            }
        }
        return x;
    }
}
