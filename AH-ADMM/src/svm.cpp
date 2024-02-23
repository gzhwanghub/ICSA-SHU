#include "../include/svm.h"
#include "../include/sparse_operator.h"

l2r_l2_svc_fun::l2r_l2_svc_fun(const Problem *prob, double rho)
{
	int l=prob->l;

	this->prob = prob;

	z = new double[l];

	I = new int[l];
	rho_=rho;
}

l2r_l2_svc_fun::~l2r_l2_svc_fun()
{
	delete[] z;
	delete[] I;
	//delete reduce_vectors;
}

double l2r_l2_svc_fun::fun(double *w,double *yi,double *zi)
{
	int i;
	double f=0;
	double *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();

	Xv(w, z);
	for(i=0;i<l;i++)
	{
		z[i] = y[i]*z[i];
		double d = 1-z[i];
		if (d > 0)
			f += d*d;
	}

//#pragma omp parallel for private(i) reduction(+:f) schedule(static)
    for (i = 0; i < w_size; ++i) {
        f+=yi[i]*(w[i]-zi[i])+0.5*rho_*(w[i]-zi[i])*(w[i]-zi[i]);
    }

	return(f);
}

void l2r_l2_svc_fun::grad(double *w, double *g,double *yi,double *zi)
{
	int i;
	double *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();

	sizeI = 0;
	for (i=0;i<l;i++)
		if (z[i] < 1)
		{
			z[sizeI] = y[i]*(z[i]-1);
			I[sizeI] = i;
			sizeI++;
		}
	XTv(z, g);

//#pragma omp parallel for private(i) schedule(static)
	for(i=0;i<w_size;i++)
		g[i] = 2*g[i];

//#pragma omp parallel for private(i) schedule(static)
    for(i=0;i<w_size;++i)
        g[i]+=rho_ *(w[i]-zi[i]) + yi[i];

}

int l2r_l2_svc_fun::get_nr_variable(void)
{
	return prob->n;
}
 
void l2r_l2_svc_fun::get_diagH(double *M)
{
	int i;
	int w_size=get_nr_variable();
	FeatureNode **x = prob->x;

	for (i=0; i<w_size; i++)
		M[i] = 1;

	for (i=0; i<sizeI; i++)
	{
		int idx = I[i];
		FeatureNode *s = x[idx];
		while (s->index!=-1)
		{
			M[s->index-1] += s->value*s->value*2;
			s++;
		}
	}
}

void l2r_l2_svc_fun::Hv(double *s, double *Hs)
{
	int i;
	int w_size=get_nr_variable();
	FeatureNode **x=prob->x;

//	reduce_vectors->init();
    for(i=0;i<w_size;i++)
		Hs[i] = 0; 
//#pragma omp parallel for private(i) schedule(guided)
	for(i=0;i<sizeI;i++)
	{
        FeatureNode * const xi=x[I[i]];
		double xTs = sparse_operator::dot(s, xi);
       // xTs = D[i]*xTs;

		sparse_operator::axpy(xTs, xi, Hs);
		//reduce_vectors->sum_scale_x(xTs, xi);
	}
	
//	reduce_vectors->reduce_sum(Hs);
//#pragma omp parallel for private(i) schedule(static)
	for(i=0;i<w_size;i++)
		Hs[i] = rho_*s[i] + 2*Hs[i];
}

void l2r_l2_svc_fun::Xv(double *v, double *Xv)
{
	int i;
	int l=prob->l;
    FeatureNode **x=prob->x;

	for(i=0;i<l;i++)
		Xv[i]=sparse_operator::dot(v, x[i]);
}

void l2r_l2_svc_fun::XTv(double *v, double *XTv)
{
	int i;
    int l=prob->l;
	int w_size=get_nr_variable();
    FeatureNode **x=prob->x;

	for(i=0;i<w_size;i++)
		XTv[i]=0;
	//for(i=0;i<l;i++)
	//	sparse_operator::axpy(v[i], x[i], XTv); 
    for(i=0;i<sizeI;i++)
		sparse_operator::axpy(v[i], x[I[i]], XTv);
}