#include <cstdarg>
#include "dcd.h"

template <class S, class T> static inline void clone(T*& dst, S* src, int n)
{
    dst = new T[n];
    memcpy((void *)dst,(void *)src,sizeof(T)*n);
}
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL

static void print_string_stdout(const char *s)
{
    fputs(s,stdout);
    fflush(stdout);
}

static void (*liblinear_print_string) (const char *) = &print_string_stdout;


#if 1
static void info(const char *fmt,...)
{
    char buf[BUFSIZ];
    va_list ap;
    va_start(ap,fmt);
    vsprintf(buf,fmt,ap);
    va_end(ap);
    (*liblinear_print_string)(buf);
}
#else
static void info(const char *fmt,...)
{
	printf(fmt);
}
#endif


DCD::DCD(const Problem *prob, double *w, double * u, double *alpha,
         double *z,  double eps, double Cp, double Cn, double rho,int f) {
    if(f==1){ 
       parallel_solve_proximity_l1l2_svc(prob,w,u,alpha,z,eps,Cp,Cn,rho);
    }
    else if(f==2){
       batch_solve_proximity_l1l2_svc(prob,w,u,alpha,z,eps,Cp,Cn,rho);
    }
    else if(f==3){
    	  solve_proximity_l1l2_svc(prob,w,u,alpha,z,eps,Cp,Cn,rho);
    }
}


#undef GETI
#define GETI(i) (y[i]+1)
// To support weights for instances, use GETI(i) (i)

void DCD::solve_proximity_l1l2_svc(
        const Problem *prob, double *w, double * u, double *alpha,
        double *z, double eps, double Cp, double Cn, double rho)
{
    int l = prob->l;
    int w_size = prob->n;
    int i, s, iter = 0;
    double C, d, G;
    double *QD = new double[l];
    int max_iter = 50;
    int *index = new int[l];
    schar *y = new schar[l]; // This should be problematic if the number of class exceed 128
    int active_size = l;

    for (int i =0; i< w_size; i++)
        u[i] = u[i]/rho;

    // PG: projected gradient, for shrinking and stopping
    double PG;
    double PGmax_old = INF;
    double PGmin_old = -INF;
    double PGmax_new, PGmin_new;

    // default solver_type: L2R_L2LOSS_SVC_DUAL
    double diag[3] = {0.5/Cn, 0, 0.5/Cp};
    double upper_bound[3] = {INF, 0, INF};
    /*if(solver_type == L2R_L1LOSS_SVC_DUAL)
    {
        diag[0] = 0;
        diag[2] = 0;
        upper_bound[0] = Cn;
        upper_bound[2] = Cp;
    }*/
    
    for(i=0; i<w_size; i++)
        w[i] = z[i] - u[i];
    for(i=0; i<l; i++)
    {
        if(prob->y[i] > 0)
        {
            y[i] = +1;
        }
        else
        {
            y[i] = -1;
        }
        QD[i] = diag[GETI(i)]; // Actually can be cached for further improvement

        FeatureNode *xi = prob->x[i];
        while (xi->index != -1)
        {
            QD[i] += (xi->value)*(xi->value);
            w[xi->index-1]+= y[i]* (xi->value) * alpha[i];
            //assert(xi->index-1>=0 && xi->index-1<w_size);	 //Debug
            xi++;
        }
        index[i] = i;
    }


    while (iter < max_iter)
    {
        PGmax_new = -INF;
        PGmin_new = INF;

        for (i=0; i<active_size; i++)
        {
            int j = i+rand()%(active_size-i);
            swap(index[i], index[j]);
        }

        for (s=0; s<active_size; s++)
        {
            i = index[s];
            G = 0;
            schar yi = y[i];

            FeatureNode *xi = prob->x[i];
            while(xi->index!= -1)
            {
                G += w[xi->index-1]*(xi->value);
                xi++;
            }
            G = G*yi-1;

            C = upper_bound[GETI(i)];
            G += alpha[i]*diag[GETI(i)];

            PG = 0;
            if (alpha[i] == 0)
            {
                if (G > PGmax_old)
                {
                    active_size--;
                    swap(index[s], index[active_size]);
                    s--;
                    continue;
                }
                else if (G < 0)
                    PG = G;
            }
            else if (alpha[i] == C)
            {
                if (G < PGmin_old)
                {
                    active_size--;
                    swap(index[s], index[active_size]);
                    s--;
                    continue;
                }
                else if (G > 0)
                    PG = G;
            }
            else
                PG = G;

            PGmax_new = max(PGmax_new, PG);
            PGmin_new = min(PGmin_new, PG);

            if(fabs(PG) > 1.0e-12)
            {
                double alpha_old = alpha[i];
                alpha[i] = min(max(alpha[i] - G/QD[i], 0.0), C);
                d = (alpha[i] - alpha_old)*yi;
                xi = prob->x[i];
                while (xi->index != -1)
                {
                    w[xi->index-1] += d*xi->value;
                    xi++;
                }
            }
        }

        iter++;
        //if (!param->inner_mute){
            //if(iter % 10 == 0)
                //info(".");
        //}

        if(PGmax_new - PGmin_new <= eps)
        {
            if(active_size == l)
                break;
            else
            {
                active_size = l;
               // if (!param->inner_mute){
                    //info("*");
               // }
                PGmax_old = INF;
                PGmin_old = -INF;
                continue;
            }
        }
        PGmax_old = PGmax_new;
        PGmin_old = PGmin_new;
        if (PGmax_old <= 0)
            PGmax_old = INF;
        if (PGmin_old >= 0)
            PGmin_old = -INF;
    }
    for (int i =0; i< w_size; i++)
        u[i] = u[i]*rho;

    delete [] QD;
    delete [] y;
    delete [] index;
}

void DCD::parallel_solve_proximity_l1l2_svc(
        const Problem *prob, double *w, double * u, double *alpha,
        double *z, double eps, double Cp, double Cn, double rho)
{
    int l = prob->l;
    int w_size = prob->n;
    int i, s, iter = 0;
    double C, d, G;
    double *QD = new double[l];
    int max_iter = 50;
    int *index = new int[l];
    schar *y = new schar[l]; // This should be problematic if the number of class exceed 128
    int active_size = l;

    size_t total_inner_iter = 0;
    size_t total_dots = 0;

    double eps1 = 0.1;

    int initB=1024;
    int B = initB;
    int maxB = 4096;

    double *Grad = new double [maxB];
    int *workingset = new int[maxB];

    for (int i =0; i< w_size; i++)
        u[i] = u[i]/rho;

    // PG: projected gradient, for shrinking and stopping
    double PG;
    double PGmax_old = INF;
    double PGmin_old = -INF;
    double PGmax_new, PGmin_new;
    int PGmax_new_index, PGmin_new_index;

    int round = 20;
    int num_updates_one_iter = 0;

    double G_sum=0;

    // default solver_type: L2R_L2LOSS_SVC_DUAL
    double diag[3] = {0.5/Cn, 0, 0.5/Cp};
    double upper_bound[3] = {INF, 0, INF};
    /*if(solver_type == L2R_L1LOSS_SVC_DUAL)
    {
        diag[0] = 0;
        diag[2] = 0;
        upper_bound[0] = Cn;
        upper_bound[2] = Cp;
    }*/

    for(i=0; i<w_size; i++)
        w[i] = z[i] - u[i];

    //for(i=0;i<l;i++)
        //alpha[i] = 0;

    for(i=0; i<l; i++)
    {
        if(prob->y[i] > 0)
        {
            y[i] = +1;
        }
        else
        {
            y[i] = -1;
        }
        QD[i] = diag[GETI(i)]; // Actually can be cached for further improvement

        FeatureNode *xi = prob->x[i];
        while (xi->index != -1)
        {
            QD[i] += (xi->value)*(xi->value);
            w[xi->index-1]+= y[i]* (xi->value) * alpha[i];
            //assert(xi->index-1>=0 && xi->index-1<w_size);	 //Debug
            xi++;
        }
        index[i] = i;
    }


    while (iter < max_iter)
    {
        PGmax_new = -INF;
        PGmin_new = INF;

        for (i=0; i<active_size; i++)
        {
            int j = i+rand()%(active_size-i);
            swap(index[i], index[j]);
        }

        num_updates_one_iter = 0;
        int t = 0;
        while (t < active_size)
        {
            int send = min(B,active_size-t);
        //double G_t1=0;
        //double G_t2=0;
        //G_t1=omp_get_wtime();
#pragma omp parallel for private(s,i) schedule(static)
            for (s = 0; s < send; s++)
            {
                i = index[t+s];
                Grad[s] = y[i]*sparse_operator::dot(w,prob->x[i])-1 + alpha[i]*diag[GETI(i)];
            }
        //G_t2=omp_get_wtime();
        //G_sum+=G_t2-G_t1;
        
            total_dots += send;

            int num_cd = 0;

            for(s=0; s<send; s++)
            {
                PG = 0;
                i = index[t+s];
                C = upper_bound[GETI(i)];

                if(alpha[i]==0)
                {
                    if(Grad[s] > PGmax_old)
                    {
                        active_size--;
                        send--;
                        if(t+send == active_size)
                            swap(index[t+s], index[t+send]);
                        else
                        {
                            int r = index[active_size];
                            index[active_size] = index[t+s];
                            index[t+s] = index[t+send];
                            index[t+send] = r;
                        }
                        Grad[s] = Grad[send];
                        s--;
                        continue;
                    }
                    else if(Grad[s] < 0)
                        PG = Grad[s];
                }
                else if(alpha[i] == C)
                {
                    if(Grad[s] < PGmin_old)
                    {
                        active_size--;
                        send--;
                        if(t+send == active_size)
                            swap(index[t+s], index[t+send]);
                        else
                        {
                            int r = index[active_size];
                            index[active_size] = index[t+s];
                            index[t+s] = index[t+send];
                            index[t+send] = r;
                        }
                        Grad[s] = Grad[send];
                        s--;
                        continue;
                    }
                    else if(Grad[s] > 0)
                        PG = Grad[s];
                }
                else
                    PG = Grad[s];

                PGmax_new = max(PGmax_new, PG);
                PGmin_new = min(PGmin_new, PG);

                if(fabs(PG) >= 0.1*eps1)
                {
                    workingset[num_cd] = i;
                    num_cd++;
                }
            }

            if(num_cd == 0)
                B = min((int)(B*1.5),maxB);
            else if(num_cd >= initB)
                B = B/2;

            for(s=0; s<num_cd; s++)
            {
                i = workingset[s];

                const  schar yi=y[i];
                FeatureNode *xi = prob->x[i];

                double G = yi*sparse_operator::dot(w,xi)-1;

                C = upper_bound[GETI(i)];
                G += alpha[i]*diag[GETI(i)];

                double alpha_new = min(max(alpha[i] - G/QD[i], 0.0), C);
                d = alpha_new - alpha[i];
                if(fabs(PG) > 1.0e-15) // or 1.0e-12
                {
                    alpha[i] = alpha_new;
                    sparse_operator::axpy(d*yi,xi,w);
                    num_updates_one_iter++;
                }

                total_inner_iter++;
                //if(total_inner_iter % 50000 == 0)
                    //info(".");
            }

            t += send;
        }

        iter++;

        if(PGmax_new - PGmin_new <= eps)
        {
            if(active_size == l)
            {
                if(eps1 <= eps+1e-12)
                    break;
            }
            else
            {
                active_size = l;
                //info("*");
              
                PGmax_old = INF;
                PGmin_old = -INF;

                eps1 = max(0.1*eps1,eps);
                //info("eps %g new eps1 %g\n",eps,eps1);

                continue;
            }
        }
        //else if(num_updates_one_iter == 0)
        //{
            //info("warning:no update in one outer iteration\n");
        //}

        PGmax_old = PGmax_new;
        PGmin_old = PGmin_new;
        if (PGmax_old <= 0)
            PGmax_old = INF;
        if (PGmin_old >= 0)
            PGmin_old = -INF;
    }
	//info("G_SUM=%d\n",G_sum);
    for (int i =0; i< w_size; i++)
        u[i] = u[i]*rho;

    delete [] Grad;
    delete [] QD;
    delete [] y;
    delete [] index;
    delete [] workingset;
}


void DCD::batch_solve_proximity_l1l2_svc(
        const Problem *prob, double *w, double * u, double *alpha,
        double *z, double eps, double Cp, double Cn, double rho)
{
    int l = prob->l;
    int w_size = prob->n;
    int i, s, iter = 0;
    double C, d, G;
    double *QD = new double[l];
    int max_iter = 50;
    int *index = new int[l];
    schar *y = new schar[l]; // This should be problematic if the number of class exceed 128
    int active_size = l;

    size_t total_inner_iter = 0;
    size_t total_dots = 0;

    double eps1 = 0.1;

    int initB=1024;
    int B = initB;
    int maxB = 4096;

    double *Grad = new double [maxB];
    int *workingset = new int[maxB];
    for (int i=0; i<l; i++) {
        alpha[i]=0;
    }
    for (int i =0; i< w_size; i++)
        u[i] = u[i]/rho;

    // PG: projected gradient, for shrinking and stopping
    double PG;
    double PGmax_old = INF;
    double PGmin_old = -INF;
    double PGmax_new, PGmin_new;
    int PGmax_new_index, PGmin_new_index;

    int round = 20;
    int num_updates_one_iter = 0;

    // default solver_type: L2R_L2LOSS_SVC_DUAL
    double diag[3] = {0.5/Cn, 0, 0.5/Cp};
    double upper_bound[3] = {INF, 0, INF};
    /*if(solver_type == L2R_L1LOSS_SVC_DUAL)
    {
        diag[0] = 0;
        diag[2] = 0;
        upper_bound[0] = Cn;
        upper_bound[2] = Cp;
    }*/

    for(i=0; i<w_size; i++)
        w[i] = z[i] - u[i];
    for(i=0; i<l; i++)
    {
        if(prob->y[i] > 0)
        {
            y[i] = +1;
        }
        else
        {
            y[i] = -1;
        }
        QD[i] = diag[GETI(i)]; // Actually can be cached for further improvement

        FeatureNode *xi = prob->x[i];
        while (xi->index != -1)
        {
            QD[i] += (xi->value)*(xi->value);
            w[xi->index-1]+= y[i]* (xi->value) * alpha[i];
            //assert(xi->index-1>=0 && xi->index-1<w_size);	 //Debug
            xi++;
        }
        index[i] = i;
    }


    while (iter < max_iter)
    {
        PGmax_new = -INF;
        PGmin_new = INF;

        for (i=0; i<active_size; i++)
        {
            int j = i+rand()%(active_size-i);
            swap(index[i], index[j]);
        }

        num_updates_one_iter = 0;
        int t = 0;
        while (t < active_size)
        {
            int send = min(B,active_size-t);

//#pragma omp parallel for private(s,i) schedule(static)
            for (s = 0; s < send; s++)
            {
                i = index[t+s];
                Grad[s] = y[i]*sparse_operator::dot(w,prob->x[i])-1 + alpha[i]*diag[GETI(i)];
            }
            total_dots += send;

            int num_cd = 0;

            for(s=0; s<send; s++)
            {
                PG = 0;
                i = index[t+s];
                C = upper_bound[GETI(i)];

                if(alpha[i]==0)
                {
                    if(Grad[s] > PGmax_old)
                    {
                        active_size--;
                        send--;
                        if(t+send == active_size)
                            swap(index[t+s], index[t+send]);
                        else
                        {
                            int r = index[active_size];
                            index[active_size] = index[t+s];
                            index[t+s] = index[t+send];
                            index[t+send] = r;
                        }
                        Grad[s] = Grad[send];
                        s--;
                        continue;
                    }
                    else if(Grad[s] < 0)
                        PG = Grad[s];
                }
                else if(alpha[i] == C)
                {
                    if(Grad[s] < PGmin_old)
                    {
                        active_size--;
                        send--;
                        if(t+send == active_size)
                            swap(index[t+s], index[t+send]);
                        else
                        {
                            int r = index[active_size];
                            index[active_size] = index[t+s];
                            index[t+s] = index[t+send];
                            index[t+send] = r;
                        }
                        Grad[s] = Grad[send];
                        s--;
                        continue;
                    }
                    else if(Grad[s] > 0)
                        PG = Grad[s];
                }
                else
                    PG = Grad[s];

                PGmax_new = max(PGmax_new, PG);
                PGmin_new = min(PGmin_new, PG);

                if(fabs(PG) >= 0.1*eps1)
                {
                    workingset[num_cd] = i;
                    num_cd++;
                }
            }

            if(num_cd == 0)
                B = min((int)(B*1.5),maxB);
            else if(num_cd >= initB)
                B = B/2;

            for(s=0; s<num_cd; s++)
            {
                i = workingset[s];

                const  schar yi=y[i];
                FeatureNode *xi = prob->x[i];

                double G = yi*sparse_operator::dot(w,xi)-1;

                C = upper_bound[GETI(i)];
                G += alpha[i]*diag[GETI(i)];

                double alpha_new = min(max(alpha[i] - G/QD[i], 0.0), C);
                d = alpha_new - alpha[i];
                if(fabs(PG) > 1.0e-15) // or 1.0e-12
                {
                    alpha[i] = alpha_new;
                    sparse_operator::axpy(d*yi,xi,w);
                    num_updates_one_iter++;
                }

                total_inner_iter++;
                //if(total_inner_iter % 50000 == 0)
                    //info(".");
            }

            t += send;
        }

        iter++;

        if(PGmax_new - PGmin_new <= eps)
        {
            if(active_size == l)
            {
                if(eps1 <= eps+1e-12)
                    break;
            }
            else
            {
                active_size = l;
                // if (!param->inner_mute){
                info("*");
                // }
                PGmax_old = INF;
                PGmin_old = -INF;

                eps1 = max(0.1*eps1,eps);
                info("eps %g new eps1 %g\n",eps,eps1);

                continue;
            }
        }
        else if(num_updates_one_iter == 0)
        {
            //info("warning:no update in one outer iteration\n");
        }

        PGmax_old = PGmax_new;
        PGmin_old = PGmin_new;
        if (PGmax_old <= 0)
            PGmax_old = INF;
        if (PGmin_old >= 0)
            PGmin_old = -INF;
    }
    for (int i =0; i< w_size; i++)
        u[i] = u[i]*rho;

    delete [] Grad;
    delete [] QD;
    delete [] y;
    delete [] index;
    delete [] workingset;
}


/*
void DCD::solve_l2r_lr_dual(const Problem *prob, double *w, double *yi, double *zi,double eps, double Cp, double Cn) {
    int l = prob->l;
    int w_size = prob->n;
    int i, s, iter = 0;
    double *xTx = new double[l];
    int max_iter = 1000;
    int *index = new int[l];
    double *alpha = new double[2*l]; // store alpha and C - alpha
    schar *y = new schar[l];
    int max_inner_iter = 100; // for inner Newton
    double innereps = 1e-2;
    double innereps_min = min(1e-8, eps);
    double upper_bound[3] = {Cn, 0, Cp};

    for(i=0; i<l; i++){
        if(prob->y[i] > 0){
            y[i] = +1;
        }
        else{
            y[i] = -1;
        }
    }

    // Initial alpha can be set here. Note that
    // 0 < alpha[i] < upper_bound[GETI(i)]
    // alpha[2*i] + alpha[2*i+1] = upper_bound[GETI(i)]
    for(i=0; i<l; i++)
    {
        alpha[2*i] = min(0.001*upper_bound[GETI(i)], 1e-8);
        alpha[2*i+1] = upper_bound[GETI(i)] - alpha[2*i];
    }

    for(i=0; i<w_size; i++)
        w[i] = 0;
    for(i=0; i<l; i++)
    {
        FeatureNode * const xi = prob->x[i];
        xTx[i] = sparse_operator::nrm2_sq(xi);
        sparse_operator::axpy(y[i]*alpha[2*i], xi, w);
        index[i] = i;
    }

    while (iter < max_iter)
    {
        for (i=0; i<l; i++)
        {
            int j = i+rand_int(l-i);
            swap(index[i], index[j]);
        }
        int newton_iter = 0;
        double Gmax = 0;
        for (s=0; s<l; s++){
            i = index[s];
            const schar yi = y[i];
            double C = upper_bound[GETI(i)];
            double ywTx = 0, xisq = xTx[i];
            FeatureNode * const xi = prob->x[i];
            ywTx = yi*sparse_operator::dot(w, xi);
            double a = xisq, b = ywTx;

            // Decide to minimize g_1(z) or g_2(z)
            int ind1 = 2*i, ind2 = 2*i+1, sign = 1;
            if(0.5*a*(alpha[ind2]-alpha[ind1])+b < 0){
                ind1 = 2*i+1;
                ind2 = 2*i;
                sign = -1;
            }

            //  g_t(z) = z*log(z) + (C-z)*log(C-z) + 0.5a(z-alpha_old)^2 + sign*b(z-alpha_old)
            double alpha_old = alpha[ind1];
            double z = alpha_old;
            if(C - z < 0.5 * C)
                z = 0.1*z;
            double gp = a*(z-alpha_old)+sign*b+log(z/(C-z));
            Gmax = max(Gmax, fabs(gp));

            // Newton method on the sub-problem
            const double eta = 0.1; // xi in the paper
            int inner_iter = 0;
            while (inner_iter <= max_inner_iter)
            {
                if(fabs(gp) < innereps)
                    break;
                double gpp = a + C/(C-z)/z;
                double tmpz = z - gp/gpp;
                if(tmpz <= 0)
                    z *= eta;
                else // tmpz in (0, C)
                    z = tmpz;
                gp = a*(z-alpha_old)+sign*b+log(z/(C-z));
                newton_iter++;
                inner_iter++;
            }

            if(inner_iter > 0) // update w
            {
                alpha[ind1] = z;
                alpha[ind2] = C-z;
                sparse_operator::axpy(sign*(z-alpha_old)*yi, xi, w);
            }
        }

        iter++;
        /*if(iter % 10 == 0)
            printf(".");*/
/*
        if(Gmax < eps)
            break;

        if(newton_iter <= l/10)
            innereps = max(innereps_min, 0.1*innereps);

    }

    //printf("\noptimization finished, #iter = %d\n",iter);
    if (iter >= max_iter)
        printf("\nWARNING: reaching max number of iterations\nUsing -s 0 may be faster (also see FAQ)\n\n");

    // calculate objective value

    double v = 0;
    for(i=0; i<w_size; i++)
        v += w[i] * w[i];
    v *= 0.5;
    for(i=0; i<l; i++)
        v += alpha[2*i] * log(alpha[2*i]) + alpha[2*i+1] * log(alpha[2*i+1])
             - upper_bound[GETI(i)] * log(upper_bound[GETI(i)]);
    //printf("Objective value = %lf\n", v);

    delete [] xTx;
    delete [] alpha;
    delete [] y;
    delete [] index;
}
*/

