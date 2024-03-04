#include <mpi.h>
#include <iostream>
#include <string>
#include <vector>
#include <stdio.h>
#include <cstring>
#include <fstream>
#include <math.h>
#include <cmath>
#include <unistd.h>
#include <stdarg.h>
#include <time.h>
#include "train_framework.h"

using namespace std;

problem::problem(const char *filename) {
    read_problem(filename);
}

void problem::read_problem(const char *filename) {
    int max_index, inst_max_index, i;
    size_t elements, j;
    FILE *fp = fopen(filename, "r");
    char *endptr;
    char *idx, *val, *label;

    if (fp == NULL) {
        fprintf(stderr, "can't open input file %s\n", filename);
        exit(1);
    }

    l = 0;
    elements = 0;
    max_line_len = 1024;
    line = Malloc(char, max_line_len);
    while (readline(fp) != NULL) {
        char *p = strtok(line, " \t"); // label

        // features
        while (1) {
            p = strtok(NULL, " \t");
            if (p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
                break;
            elements++;
        }
        elements++; // for bias term
        l++;
    }
    rewind(fp);

    bias = -1;

    y = Malloc(double, l);
    x = Malloc(struct feature_node *, l);
    x_space = Malloc(struct feature_node, elements + l);

    max_index = 0;
    j = 0;
    for (i = 0; i < l; i++) {
        inst_max_index = 0; // strtol gives 0 if wrong format
        readline(fp);
        x[i] = &x_space[j];
        label = strtok(line, " \t\n");
        if (label == NULL) // empty line
            exit_input_error(i + 1);

        y[i] = strtod(label, &endptr);
        if (y[i] == 0)
            y[i] = -1;
        if (endptr == label || *endptr != '\0')
            exit_input_error(i + 1);

        while (1) {
            idx = strtok(NULL, ":");
            val = strtok(NULL, " \t");

            if (val == NULL)
                break;

            errno = 0;
            x_space[j].index = (int) strtol(idx, &endptr, 10);
            if (endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
                exit_input_error(i + 1);
            else
                inst_max_index = x_space[j].index;

            errno = 0;
            x_space[j].value = strtod(val, &endptr);
            if (endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
                exit_input_error(i + 1);

            ++j;
        }

        if (inst_max_index > max_index)
            max_index = inst_max_index;

        if (bias >= 0)
            x_space[j++].value = bias;

        x_space[j++].index = -1;
    }

    if (bias >= 0) {
        n = max_index + 1;
        for (i = 1; i < l; i++)
            (x[i] - 2)->index = n;
        x_space[j - 2].index = n;
    } else
        n = max_index;

    fclose(fp);
}

char *problem::readline(FILE *input) {
    int len;

    if (fgets(line, max_line_len, input) == NULL)
        return NULL;

    while (strrchr(line, '\n') == NULL) {
        max_line_len *= 2;
        line = (char *) realloc(line, max_line_len);
        len = (int) strlen(line);
        if (fgets(line + len, max_line_len - len, input) == NULL)
            break;
    }
    return line;
}

void problem::exit_input_error(int line_num) {
    fprintf(stderr, "Wrong input format at line %d\n", line_num);
    exit(1);
}

std::string &LeftTrim(std::string &s) {
    auto it = s.begin();
    for (; it != s.end() && std::isspace(*it); ++it);
    s.erase(s.begin(), it);
    return s;
}

std::string &RightTrim(std::string &s) {
    auto it = s.end() - 1;
    for (; it != s.begin() - 1 && std::isspace(*it); --it);
    s.erase(it + 1, s.end());
    return s;
}

std::string &Trim(std::string &s) {
    return RightTrim(LeftTrim(s));
}

Properties::Properties(int &argc, char **&argv) {
    // 第一个参数是程序名，因此从i=1开始循环
    // 如果一个命令行参数以-开头，那么我们认为这是一个有效的参数
    // 因此有效参数的格式为 -key1 value1 -key2 value2 other
    // 当遇到不以-开头的命令行参数，停止解析
    int i = 1;
    for (; i < argc && argv[i][0] == '-';) {
        std::string key(argv[i] + 1);
        // 如果命令行参数中有 -file path，那么将会从path读取配置文件
        // 因此file为保留的参数名，用户无法使用
        if (key == "file") {
            ParseFromFile(argv[i + 1]);
        } else {
            properties_[key] = argv[i + 1];
        }
        i += 2;
    }
    // 将剩余的命令行参数挪动位置，并修改argc
    int j = 1, k = i;
    for (; k > 1 && k < argc;) {
        argv[j++] = argv[k++];
    }
    argc -= (i - 1);
}

Properties::Properties(const std::string &path) {
    ParseFromFile(path);
}

void Properties::ParseFromFile(const std::string &path) {
    std::ifstream reader(path);
    if (reader.fail()) {
        std::cout << "无法打开配置文件，文件名：" << path << std::endl;
    }

    // 新建一个map临时存放属性值
    std::map<std::string, std::string> temp;
    std::string line;
    while (std::getline(reader, line)) {
        // 每一行中#号后面的内容为注释，因此删去这些内容
        std::size_t pos = line.find_first_of('#');
        if (pos != std::string::npos) {
            line.erase(pos);
        }
        // 除去每一行内容的前后空格
        Trim(line);
        if (line.empty()) {
            continue;
        }
        // 每一行内容的格式为key:value，冒号两边可以有空格
        pos = line.find_first_of(':');
        if (pos == std::string::npos || pos == 0 || pos == line.length() - 1) {
            std::cout << "格式错误，应该为key:value格式，" << line << std::endl;
        }
        std::string key = line.substr(0, pos);
        std::string value = line.substr(pos + 1);
        Trim(key);
        Trim(value);
        temp[key] = value;
    }
    reader.close();

    //命令行参数的优先程度大于配置文件的参数，因此只把properties中没有的参数复制过去
    for (auto it = temp.begin(); it != temp.end(); ++it) {
        if (properties_.count(it->first) == 0) {
            properties_[it->first] = it->second;
        }
    }
}

std::string Properties::GetString(const std::string &property_name) {
    return properties_.at(property_name);
}

int Properties::GetInt(const std::string &property_name) {
    return Convert<int, std::string>(properties_.at(property_name));
}

double Properties::GetDouble(const std::string &property_name) {
    return Convert<double, std::string>(properties_.at(property_name));
}

bool Properties::GetBool(const std::string &property_name) {
    if (properties_.at(property_name) == "true") {
        return true;
    } else if (properties_.at(property_name) == "false") {
        return false;
    }
    std::cout << property_name << "的值必须为true或者false" << std::endl;
    return false;
}

bool Properties::HasProperty(const std::string &property_name) {
    return properties_.count(property_name) != 0;
}

void Properties::CheckProperty(const std::string &property_name) {
    if (!HasProperty(property_name)) {
        std::cout << "缺少参数\n";
    }
}

void Properties::Print() {
    std::cout << "**************************************" << std::endl;
    for (auto it = properties_.begin(); it != properties_.end(); ++it) {
        std::cout << it->first << ":" << it->second << std::endl;
    }
    std::cout << "**************************************" << std::endl;
}

args_t::args_t(int rank, int size, Properties properties) {
    myid = rank;
    procnum = size;
    min_barrier = properties.GetInt("min_barrier");
    max_delay = properties.GetInt("max_delay");
    max_iterations = properties.GetInt("max_iterations");
    filter_type = properties.GetInt("filter_type");
    rho = properties.GetDouble("rho");
    l1reg = properties.GetDouble("l1reg");
    l2reg = properties.GetDouble("l2reg");
    ABSTOL = properties.GetDouble("ABSTOL");
    RELTOL = properties.GetDouble("RELTOL");
    worker_per_group_ = properties.GetInt("worker_per_group");
    group_type = properties.GetInt("group_type");
    leader_num_ = procnum / worker_per_group_;
//    if(procnum == (int)sqrt(procnum) * (int)sqrt(procnum)){
//        sqrt_procnum_ = sqrt(procnum);
//    }else{
//        fprintf(stderr,"Unable to perform torus allreduce algorithm！\n");
//    }
    if (leader_num_ == (int) sqrt(leader_num_) * (int) sqrt(leader_num_)) {
        sqrt_leader_ = sqrt(leader_num_);
    } else {
        fprintf(stderr, "Unable to perform hierarchical torus algorithm！\n");
    }
    //thread_num = properties.GetInt("thread_num");
}

void ReduceSum(double *dst, double *src, int size){
    for (int i = 0; i < size; ++i) {
        dst[i] += src[i];
    }
}

void ReduceReplace(double *dst, double *src, int size){
    for (int i = 0; i < size; ++i) {
        dst[i] = src[i];
    }
}

int Next(int rank, int size){
    return ((rank + 1) % size);
}

int Prev (int rank , int size){
    return ((size + rank - 1) % size);
}

Collective::Collective(args_t *args, problem *problem) {
    comm_rank_ = args->myid;
    comm_size_ = args->procnum;
    dim_ = problem->n;
    datatype_ = MPI_DOUBLE;
    sqrt_leader_ = args->sqrt_leader_;
    worker_per_group_ = args->worker_per_group_;
    sqrt_procnum_ = args->sqrt_procnum_;
}

void Collective::RingAllreduce(double *data, int count ,MPI_Comm communicator){
    int comm_rank;
    int comm_size;
    MPI_Comm_rank(communicator, &comm_rank);
    MPI_Comm_size(communicator, &comm_size);
    int segment_size = count / comm_size;
    int residual = count % comm_size;
    int *segment_sizes = (int *) malloc(sizeof(int) * comm_size);
    int *segment_start_ptr = (int *) malloc(sizeof(int) * comm_size);
    int segment_ptr = 0;
    for(int i = 0; i < comm_size; i++){
        segment_start_ptr[i] = segment_ptr;
        segment_sizes[i] = segment_size;
        if(i < residual){
            segment_sizes[i]++;
        }
        segment_ptr += segment_sizes[i];
    }
    if(segment_start_ptr[comm_size - 1] + segment_sizes[comm_size - 1] != count){
        MPI_Abort(MPI_COMM_WORLD, MPI_ERR_COUNT);
    }
    MPI_Status recv_status;
    MPI_Request recv_req;
    double *buffer = (double *) malloc(sizeof(double) * segment_sizes[0]);
    for(int iter = 0; iter < comm_size - 1; iter++){
        int recv_chunk = (comm_rank - iter - 1 + comm_size) % comm_size;
        int send_chunk = (comm_rank - iter + comm_size) % comm_size;
        double *sending_segment = &(data[segment_start_ptr[send_chunk]]);
        MPI_Irecv(buffer, segment_sizes[recv_chunk], datatype_,
                  Prev(comm_rank, comm_size), 0, communicator, &recv_req);
        MPI_Send(sending_segment, segment_sizes[send_chunk], datatype_,
                 Next(comm_rank, comm_size), 0, communicator);
        double *updating_segment = &(data[segment_start_ptr[recv_chunk]]);
        MPI_Wait(&recv_req, &recv_status);
        ReduceSum(updating_segment, buffer, segment_sizes[recv_chunk]);
    }
    MPI_Barrier(communicator);
    for(int iter = 0; iter < comm_size - 1; iter++){
        int recv_chunk = (comm_rank - iter + comm_size) % comm_size;
        int send_chunk = (comm_rank - iter + 1 + comm_size) % comm_size;
        double *sending_segment = &(data[segment_start_ptr[send_chunk]]);
        double *updating_segment = &(data[segment_start_ptr[recv_chunk]]);
        MPI_Sendrecv(sending_segment, segment_sizes[send_chunk], datatype_,
                     Next(comm_rank, comm_size), 0, updating_segment,
                     segment_sizes[recv_chunk], datatype_,
                     Prev(comm_rank, comm_size), 0, communicator, &recv_status);
    }
    free(buffer);
    free(segment_sizes);
    free(segment_start_ptr);
}

void Collective::TorusAllreduce(double *data, int worker_per_group, int group_count, MPI_Comm communicator, int *nbrs){
    int comm_rank;
    int comm_size;
    MPI_Comm_rank(communicator, &comm_rank);
    MPI_Comm_size(communicator, &comm_size);
    comm_size /= worker_per_group;
    int segment_size = dim_ / comm_size;
    int residual = dim_ % comm_size;
    int *segment_sizes = (int *) malloc(sizeof(int) * comm_size);
    int *segment_start_ptr = (int *) malloc(sizeof(int) * comm_size);
    int segment_ptr = 0;
    for(int i = 0; i < comm_size; i++){
        segment_start_ptr[i] = segment_ptr;
        segment_sizes[i] = segment_size;
        if(i < residual){
            segment_sizes[i]++;
        }
        segment_ptr += segment_sizes[i];
    }
    if(segment_start_ptr[comm_size - 1] + segment_sizes[comm_size - 1] != dim_){
        MPI_Abort(MPI_COMM_WORLD, MPI_ERR_COUNT);
    }
    MPI_Status recv_status;
    MPI_Request recv_req;
    double *buffer = (double *) malloc(sizeof(double) * segment_sizes[0]);
    //scatter-reduce
    for(int iter = 0; iter < comm_size - 1; iter++){
        int recv_chunk = (comm_rank - iter - 1 + comm_size) % comm_size;
        int send_chunk = (comm_rank - iter + comm_size) % comm_size;
        double *sending_segment = &(data[segment_start_ptr[send_chunk]]);
        MPI_Irecv(buffer, segment_sizes[recv_chunk], datatype_,
                  nbrs[0], 0, communicator, &recv_req);
        MPI_Send(sending_segment, segment_sizes[send_chunk], datatype_,
                 nbrs[1], 0, communicator);
        double *updating_segment = &(data[segment_start_ptr[recv_chunk]]);
        MPI_Wait(&recv_req, &recv_status);
        ReduceSum(updating_segment, buffer, segment_sizes[recv_chunk]);
    }
    MPI_Barrier(communicator);
    //segment ringallreduce
    int color = comm_rank % group_count;
    MPI_Comm subgrp_comm; //intra-group
    MPI_Comm_split(communicator, color, comm_rank, &subgrp_comm);
    int subgrp_rank, subgrp_size;
    MPI_Comm_rank(subgrp_comm, &subgrp_rank);
    MPI_Comm_size(subgrp_comm, &subgrp_size);
    int reduce_chunk = (color + 1) % group_count;
    double *reduce_segment = &(data[segment_start_ptr[reduce_chunk]]);
    RingAllreduce(reduce_segment, segment_sizes[reduce_chunk], subgrp_comm);
//    MPI_Allreduce(reduce_segment, buffer,segment_sizes[reduce_chunk], datatype_, MPI_SUM, subgrp_comm);
//    ReduceReplace(reduce_segment,buffer,segment_sizes[reduce_chunk]);
    //allgather
    for(int iter = 0; iter < comm_size - 1; iter++){
        int recv_chunk = (comm_rank - iter + comm_size) % comm_size;
        int send_chunk = (comm_rank - iter + 1 + comm_size) % comm_size;
        double *sending_segment = &(data[segment_start_ptr[send_chunk]]);
        double *updating_segment = &(data[segment_start_ptr[recv_chunk]]);
        MPI_Sendrecv(sending_segment, segment_sizes[send_chunk], datatype_,
                     nbrs[1], 0, updating_segment,
                     segment_sizes[recv_chunk], datatype_,
                     nbrs[0], 0, communicator, &recv_status);
    }
    free(buffer);
    free(segment_sizes);
    free(segment_start_ptr);
}

void Collective::HierarchicalTorus(double *data, MPI_Comm &MAINGRP_COMM, MPI_Comm &SUBGRP_COMM, int *nbrs){
    RingAllreduce(data, dim_, SUBGRP_COMM);
    if (MAINGRP_COMM != MPI_COMM_NULL){
        int main_size;
        MPI_Comm_size(MAINGRP_COMM, &main_size);
        MPI_Comm RING_COMM;
        CreateTorus(MAINGRP_COMM, RING_COMM, sqrt(main_size), main_size, nbrs);
        TorusAllreduce(data, sqrt(main_size),sqrt(main_size), RING_COMM, nbrs);
    }
    MPI_Bcast(data, dim_, datatype_, 0, SUBGRP_COMM);
}

void Collective::HierarchicalAllreduce(double *data, MPI_Comm &MAINGRP_COMM, MPI_Comm &SUBGRP_COMM){
//    double* recv_data;
//    recv_data = new double[dim_];
//    MPI_Reduce(data,recv_data,dim_, datatype_, MPI_SUM,0, SUBGRP_COMM);
    RingAllreduce(data, dim_,SUBGRP_COMM);
    if(MAINGRP_COMM != MPI_COMM_NULL){
        RingAllreduce(data, dim_,MAINGRP_COMM);
    }
    MPI_Bcast(data, dim_, datatype_, 0, SUBGRP_COMM);
}

void Collective::CreateTorus(MPI_Comm OLD_COMM, MPI_Comm &TORUS_COMM, int worker_per_group, int main_size, int *nbrs){
    int comm_rank, comm_size;
    int dims[2] = {worker_per_group, main_size / worker_per_group}, periods[2] = {0,1}, reorder = 1, coords[2];
    MPI_Status status;
    MPI_Cart_create(OLD_COMM, 2, dims, periods, 1, &TORUS_COMM);
    MPI_Comm_rank(TORUS_COMM, &comm_rank);
    MPI_Cart_coords(TORUS_COMM, comm_rank,2, coords);
    MPI_Cart_shift(TORUS_COMM, 1, 1, &nbrs[LEFT], &nbrs[RIGHT]);
    MPI_Cart_shift(TORUS_COMM, 0, 1, &nbrs[UP], &nbrs[DOWN]);
}

static double dnrm2_(int *n, double *x, int *incx)
{
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

    if( nn > 0 && iincx > 0 )
    {
        if (nn == 1)
        {
            norm = fabs(x[0]);
        }
        else
        {
            scale = 0.0;
            ssq = 1.0;

            /* The following loop is equivalent to this call to the LAPACK
               auxiliary routine:   CALL SLASSQ( N, X, INCX, SCALE, SSQ ) */

            for (ix=(nn-1)*iincx; ix>=0; ix-=iincx)
            {
                if (x[ix] != 0.0)
                {
                    absxi = fabs(x[ix]);
                    if (scale < absxi)
                    {
                        temp = scale / absxi;
                        ssq = ssq * (temp * temp) + 1.0;
                        scale = absxi;
                    }
                    else
                    {
                        temp = absxi / scale;
                        ssq += temp * temp;
                    }
                }
            }
            norm = scale * sqrt(ssq);
        }
    }
    else
        norm = 0.0;

    return norm;

} /* dnrm2_ */

static double ddot_(int *n, double *sx, int *incx, double *sy, int *incy)
{
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
    if (nn > 0)
    {
        if (iincx == 1 && iincy == 1) /* code for both increments equal to 1 */
        {
            m = nn-4;
            for (i = 0; i < m; i += 5)
                stemp += sx[i] * sy[i] + sx[i+1] * sy[i+1] + sx[i+2] * sy[i+2] +
                         sx[i+3] * sy[i+3] + sx[i+4] * sy[i+4];

            for ( ; i < nn; i++)        /* clean-up loop */
                stemp += sx[i] * sy[i];
        }
        else /* code for unequal increments or equal increments not equal to 1 */
        {
            ix = 0;
            iy = 0;
            if (iincx < 0)
                ix = (1 - nn) * iincx;
            if (iincy < 0)
                iy = (1 - nn) * iincy;
            for (i = 0; i < nn; i++)
            {
                stemp += sx[ix] * sy[iy];
                ix += iincx;
                iy += iincy;
            }
        }
    }

    return stemp;
} /* ddot_ */

static int daxpy_(int *n, double *sa, double *sx, int *incx, double *sy,
                  int *incy)
{
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

    if( nn > 0 && ssa != 0.0 )
    {
        if (iincx == 1 && iincy == 1) /* code for both increments equal to 1 */
        {
            m = nn-3;
            for (i = 0; i < m; i += 4)
            {
                sy[i] += ssa * sx[i];
                sy[i+1] += ssa * sx[i+1];
                sy[i+2] += ssa * sx[i+2];
                sy[i+3] += ssa * sx[i+3];
            }
            for ( ; i < nn; ++i) /* clean-up loop */
                sy[i] += ssa * sx[i];
        }
        else /* code for unequal increments or equal increments not equal to 1 */
        {
            ix = iincx >= 0 ? 0 : (1 - nn) * iincx;
            iy = iincy >= 0 ? 0 : (1 - nn) * iincy;
            for (i = 0; i < nn; i++)
            {
                sy[iy] += ssa * sx[ix];
                ix += iincx;
                iy += iincy;
            }
        }
    }

    return 0;
} /* daxpy_ */

static int dscal_(int *n, double *sa, double *sx, int *incx)
{
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

    if (nn > 0 && iincx > 0)
    {
        if (iincx == 1) /* code for increment equal to 1 */
        {
            m = nn-4;
            for (i = 0; i < m; i += 5)
            {
                sx[i] = ssa * sx[i];
                sx[i+1] = ssa * sx[i+1];
                sx[i+2] = ssa * sx[i+2];
                sx[i+3] = ssa * sx[i+3];
                sx[i+4] = ssa * sx[i+4];
            }
            for ( ; i < nn; ++i) /* clean-up loop */
                sx[i] = ssa * sx[i];
        }
        else /* code for increment not equal to 1 */
        {
            nincx = nn * iincx;
            for (i = 0; i < nincx; i += iincx)
                sx[i] = ssa * sx[i];
        }
    }

    return 0;
} /* dscal_ */

static void default_print(const char *buf)
{
    fputs(buf,stdout);
    fflush(stdout);
}

static double uTMv(int n, double *u, double *M, double *v)
{
    const int m = n-4;
    double res = 0;
    int i;
    for (i=0; i<m; i+=5)
        res += u[i]*M[i]*v[i]+u[i+1]*M[i+1]*v[i+1]+u[i+2]*M[i+2]*v[i+2]+
               u[i+3]*M[i+3]*v[i+3]+u[i+4]*M[i+4]*v[i+4];
    for (; i<n; i++)
        res += u[i]*M[i]*v[i];
    return res;
}

void TRON::info(const char *fmt,...)
{
    char buf[BUFSIZ];
    va_list ap;
    va_start(ap,fmt);
    vsprintf(buf,fmt,ap);
    va_end(ap);
    (*tron_print_string)(buf);
}

TRON::TRON(const Function *fun_obj, double eps, double eps_cg, int max_iter)
{
    this->fun_obj=const_cast<Function *>(fun_obj);
    this->eps=eps;
    this->eps_cg=eps_cg;
    this->max_iter=max_iter;
    tron_print_string = default_print;
}
TRON::~TRON()
{
}

void TRON::tron(double *w,double *yi,double *zi, double *statistical_time, int &tron_iteration, int *cg_iter_array)
{
    // module computing time
    double object_function_begin_time(0), object_function_end_time(0),
    gradient_begin_time(0), gradient_end_time(0), CG_begin_time(0), CG_end_time(0),
     diagonal_begin_time(0), diagonal_end_time(0);
    statistical_time[2] = 0;
    statistical_time[4] = 0;
    // Parameters for updating the iterates.
    double eta0 = 1e-4, eta1 = 0.25, eta2 = 0.75;
    // Parameters for updating the trust region size delta.
    double sigma1 = 0.25, sigma2 = 0.5, sigma3 = 4;
    int n = fun_obj->get_nr_variable();
    int i;
    double delta=0, sMnorm, one=1.0;
    double alpha, f, fnew, prered, actred, gs;
    int search = 1, inc = 1;
    tron_iteration = 1;
    double *s = new double[n];
    double *r = new double[n];
    double *g = new double[n];
    const double alpha_pcg = 0.01;
    double *M = new double[n];
    // calculate gradient norm at w=0 for stopping condition.
    double *w0 = new double[n];
    for (i=0; i<n; i++)
        w0[i] = 0;
    object_function_begin_time = MPI_Wtime();
    fun_obj->fun(w0,yi,zi);
    object_function_end_time = MPI_Wtime();
    statistical_time[0] = object_function_end_time - object_function_begin_time;
    gradient_begin_time = MPI_Wtime();
    fun_obj->grad(w0, g,yi,zi);
    gradient_end_time = MPI_Wtime();
    statistical_time[1] = gradient_end_time - gradient_begin_time;
    double gnorm0 = dnrm2_(&n, g, &inc);
    delete [] w0;
    f = fun_obj->fun(w,yi,zi);
    fun_obj->grad(w, g,yi,zi);
    double gnorm = dnrm2_(&n, g, &inc);
    if (gnorm <= eps * gnorm0)
        search = 0;
    diagonal_begin_time = MPI_Wtime();
    fun_obj->get_diagH(M);
    diagonal_end_time = MPI_Wtime();
    statistical_time[3] = diagonal_end_time - diagonal_begin_time;
    for(i=0; i<n; i++)
        M[i] = (1-alpha_pcg) + alpha_pcg*M[i];
    delta = sqrt(uTMv(n, g, M, g));
    int cg_iter = 0;
    double *w_new = new double[n];
    bool reach_boundary;
    bool delta_adjusted = false;
    while (tron_iteration <= max_iter && search)
    {
        CG_begin_time = MPI_Wtime();
        i = 0;
        cg_iter_array[tron_iteration] = trpcg(delta, g, M, s, r, &reach_boundary, statistical_time[4]);
        CG_end_time = MPI_Wtime();
        statistical_time[2] += CG_end_time - CG_begin_time;
        memcpy(w_new, w, sizeof(double)*n);
        daxpy_(&n, &one, s, &inc, w_new, &inc);
        gs = ddot_(&n, g, &inc, s, &inc);
        prered = -0.5*(gs-ddot_(&n, s, &inc, r, &inc));
        fnew = fun_obj->fun(w_new,yi,zi);
        // Compute the actual reduction.
        actred = f - fnew;
        // On the first iteration, adjust the initial step bound.
        sMnorm = sqrt(uTMv(n, s, M, s));
        if (tron_iteration == 1 && !delta_adjusted)
        {
            delta = min(delta, sMnorm);
            delta_adjusted = true;
        }
        // Compute prediction alpha*sMnorm of the step.
        if (fnew - f - gs <= 0)
            alpha = sigma3;
        else
            alpha = max(sigma1, -0.5*(gs/(fnew - f - gs)));
        // Update the trust region bound according to the ratio of actual to predicted reduction.
        if (actred < eta0*prered)
            delta = min(alpha*sMnorm, sigma2*delta);
        else if (actred < eta1*prered)
            delta = max(sigma1*delta, min(alpha*sMnorm, sigma2*delta));
        else if (actred < eta2*prered)
            delta = max(sigma1*delta, min(alpha*sMnorm, sigma3*delta));
        else
        {
            if (reach_boundary)
                delta = sigma3*delta;
            else
                delta = max(delta, min(alpha*sMnorm, sigma3*delta));
        }

//        info("iter %2d act %5.3e pre %5.3e delta %5.3e f %5.3e |g| %5.3e CG %3d\n", iter, actred, prered, delta, f, gnorm, cg_iter);
        if (actred > eta0*prered)
        {
            tron_iteration++;
            memcpy(w, w_new, sizeof(double)*n);
            f = fnew;
            fun_obj->grad(w, g,yi,zi);
            fun_obj->get_diagH(M);
            for(i=0; i<n; i++)
                M[i] = (1-alpha_pcg) + alpha_pcg*M[i];
            gnorm = dnrm2_(&n, g, &inc);
			//info("gnorm: %f***%f\n",gnorm,gnorm0);
            if (gnorm <= eps*gnorm0)
                break;
        }
        if (f < -1.0e+32)
        {
           // info("WARNING: f < -1.0e+32\n");
            break;
        }
        if (prered <= 0)
        {
          //  info("WARNING: prered <= 0\n");
            break;
        }
        if (fabs(actred) <= 1.0e-12*fabs(f) &&
            fabs(prered) <= 1.0e-12*fabs(f))
        {
          //  info("WARNING: actred and prered too small\n");
            break;
        }
    }
  //  info("iteration time %d\n",iter);
    delete[] g;
    delete[] r;
    delete[] w_new;
    delete[] s;
    delete[] M;
}

int TRON::trpcg(double delta, double *g, double *M, double *s, double *r, bool *reach_boundary, double &hassian_time){
    int i, inc = 1;
    int n = fun_obj->get_nr_variable();
    double one = 1;
    double *d = new double[n];
    double *Hd = new double[n];
    double zTr, znewTrnew, alpha, beta, cgtol;
    double *z = new double[n];
    double hassian_begin_time(0), hassian_end_time(0);
    *reach_boundary = false;
    for (i=0; i<n; i++){
        //d and r: direction s:信赖域半径
        s[i] = 0;
        r[i] = -g[i];
        z[i] = r[i] / M[i];
        d[i] = z[i];
    }
    zTr = ddot_(&n, z, &inc, r, &inc);//-g[i] * -g[i] / M[i],梯度二范数
    cgtol = eps_cg*sqrt(zTr);//eps_cg:epsilon_k
    int cg_iter = 0;
    while (1){
        if (sqrt(zTr) <= cgtol)
            break;
        cg_iter++;
        hassian_begin_time = MPI_Wtime();
        //Hd:hassian矩阵
        fun_obj->Hv(d, Hd);
        hassian_end_time = MPI_Wtime();
        hassian_time += hassian_end_time - hassian_begin_time;
        alpha = zTr/ddot_(&n, d, &inc, Hd, &inc);
        daxpy_(&n, &alpha, d, &inc, s, &inc);//更新牛顿方向

        double sMnorm = sqrt(uTMv(n, s, M, s));
        if (sMnorm > delta)
        {
           // info("cg reaches trust region boundary\n");
            *reach_boundary = true;
            alpha = -alpha;
            daxpy_(&n, &alpha, d, &inc, s, &inc);
            double sTMd = uTMv(n, s, M, d);
            double sTMs = uTMv(n, s, M, s);
            double dTMd = uTMv(n, d, M, d);
            double dsq = delta*delta;
            double rad = sqrt(sTMd*sTMd + dTMd*(dsq-sTMs));
            if (sTMd >= 0)
                alpha = (dsq - sTMs)/(sTMd + rad);
            else
                alpha = (rad - sTMd)/dTMd;
            daxpy_(&n, &alpha, d, &inc, s, &inc);
            alpha = -alpha;
            daxpy_(&n, &alpha, Hd, &inc, r, &inc);
            break;
        }
        alpha = -alpha;
        daxpy_(&n, &alpha, Hd, &inc, r, &inc);

        for (i=0; i<n; i++)
            z[i] = r[i] / M[i];
        znewTrnew = ddot_(&n, z, &inc, r, &inc);
        beta = znewTrnew/zTr;
        dscal_(&n, &beta, d, &inc);
        daxpy_(&n, &one, z, &inc, d, &inc);
        zTr = znewTrnew;
    }

    delete[] d;
    delete[] Hd;
    delete[] z;
    return(cg_iter);
}

double TRON::norm_inf(int n, double *x){
    double dmax = fabs(x[0]);
    for (int i=1; i<n; i++)
        if (fabs(x[i]) >= dmax)
            dmax = fabs(x[i]);
    return(dmax);
}

void TRON::set_print_string(void (*print_string) (const char *buf)){
    tron_print_string = print_string;
}

l2r_lr_fun::l2r_lr_fun(const problem *prob, double C)
{
	int l=prob->l;

	this->prob = prob;

	z = new double[l];
	D = new double[l];
	this->C = C;
}

l2r_lr_fun::~l2r_lr_fun()
{
	delete[] z;
	delete[] D;
}

double l2r_lr_fun::fun(double *w,double *yi, double *zi)
{
	int i;
	double f=0;
	double *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();

	Xv(w, z);

	for(i=0;i<w_size;i++) {
        double temp = w[i] - zi[i] + yi[i] / C;
        f += temp * temp;
    }
	f *= (C / 2.0);
	for(i=0;i<l;i++)
	{
        f += std::log(
                1 + std::exp(-prob->y[i] * z[i]));
	}

	return(f);
}

void l2r_lr_fun::grad(double *w, double *g,double *yi, double *zi)
{

	int i;
	double *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();

	for(i=0;i<l;i++)
	{
		z[i] = 1/(1 + exp(-y[i]*z[i]));
		D[i] = z[i]*(1-z[i]);
		z[i] = (z[i]-1)*y[i];
	}
	XTv(z, g);

	for(i=0;i<w_size;i++)
		g[i] = g[i] + C *(w[i]-zi[i]) + yi[i];
}

int l2r_lr_fun::get_nr_variable(void)
{
	return prob->n;
}
int l2r_lr_fun::get_number()
{
	return prob->l;
}
void l2r_lr_fun::get_diagH(double *M){
    int i;
    int l = prob->l;
    int w_size=get_nr_variable();
    feature_node **x = prob->x;

    for (i=0; i<w_size; i++)
        M[i] = 1;

    for (i=0; i<l; i++){
        feature_node *s = x[i];
        while (s->index!=-1){
            M[s->index-1] += s->value*s->value*D[i];
            s++;
        }
    }
}
////迭代法计算hassian矩阵与向量s乘积，Hs:hassian矩阵
void l2r_lr_fun::Hv(double *s, double *Hs)
{
    //\nablaf(w)s Hs:\nablaf(w)

	int i;
	int l=prob->l;
	int w_size=get_nr_variable();
	feature_node **x=prob->x;

	for(i=0;i<w_size;i++)
		Hs[i] = 0;
	for(i=0;i<l;i++)
	{
		feature_node * const xi=x[i];
		double xTs = sparse_operator::dot(s, xi);

		xTs = D[i]*xTs;

		sparse_operator::axpy(xTs, xi, Hs);//X^TDX
	}
	for(i=0;i<w_size;i++)
	    //s + CX^T(D(Xs))
		Hs[i] = C*s[i] + Hs[i];
}

void l2r_lr_fun::Xv(double *v, double *Xv)
{
	int i;
	int l=prob->l;
	feature_node **x=prob->x;

	for(i=0;i<l;i++)
		Xv[i]=sparse_operator::dot(v, x[i]);
}

void l2r_lr_fun::XTv(double *v, double *XTv)
{
	int i;
	int l=prob->l;
	int w_size=get_nr_variable();
	feature_node **x=prob->x;

	for(i=0;i<w_size;i++)
		XTv[i]=0;
	for(i=0;i<l;i++)
		sparse_operator::axpy(v[i], x[i], XTv);
}
void l2r_lr_fun::one_grad(double *g,double *w,int data_index){
    //1.求h(xi)
    feature_node **x=prob->x;
    double *y=prob->y;
    double zi=sparse_operator::dot(w,x[data_index]);//w向量与x[data_index]点乘
    zi=1/(1+exp(-y[data_index]*zi));
    zi=(zi-1)*y[data_index];
    sparse_operator::axpy(zi,x[data_index],g);//zi标量与xx[data_index]向量相乘
}

void l2r_lr_fun::batch_grad(double *g,double *w,double *y,double *z,vector<int> &batch_vals){
    for (int j = 0; j < get_nr_variable(); ++j) {
        g[j]=0.0;
    }
    int i;
    vector<int>::iterator batch_begin=batch_vals.begin(),batch_end;
	feature_node **x=prob->x;
    double *label=prob->y;
   for(;batch_begin!=batch_vals.end();batch_begin++)
   {
	   //one_grad(g,w,*batch_begin);
	    int index=*batch_begin;
		//double zi=sparse_operator::dot(w,x[index]);
		//zi=1/(1+exp(-label[index]*zi));
        //zi=(zi-1)*label[index];
		z[index]=1/(1+exp(-label[index]*z[index]));
		z[index]=(z[index]-1)*label[index];
        sparse_operator::axpy(z[index],x[index],g);
   }
    for (i = 0; i < get_nr_variable(); ++i) {
        g[i]+=(C*(w[i]-z[i]) + y[i]);
    }
}

ADMM::ADMM(args_t *args, problem *prob1, string test_file_path, Collective *collective) {
    prob_ = prob1;
    data_num_ = prob_->l;
    collective_ = collective;
    dim_ = prob_->n;
    myid_ = args->myid;
    procnum_ = args->procnum;
    barrier_size_ = args->min_barrier;
    delta = args->max_delay;
    max_iterations_ = args->max_iterations;
    filter_type_ = args->filter_type;
    rho_ = args->rho;
    ABSTOL = args->ABSTOL;
    RELTOL = args->RELTOL;
    l2reg_ = args->l2reg;
    l1reg_ = args->l1reg;
    worker_per_group_ = args->worker_per_group_;
    if (l1reg_ > 0) hasL1reg_ = true;
    else hasL1reg_ = false;
    lemada_ = 0.01;
    x_ = new double[dim_];
    y_ = new double[dim_];
    z_ = new double[dim_]; //C = new double[dim];
    z_pre_ = new double[dim_];
    w_ = new double[dim_];
    sum_w_ = new double[dim_];
    cg_iter_ = new int[10]();
    group_num_ = procnum_ / worker_per_group_;
    maingrp_rank_ = (int *) malloc(sizeof(int) * group_num_);
    for (int i = 0; i < dim_; i++) {
        x_[i] = 0;
        y_[i] = 0;
        z_[i] = 0; //C[i] = rho;
        w_[i] = 0;
        sum_w_[i] = 0;
    }
    eps_ = 0.001;
    eps_cg_ = 0.1;
    int pos = 0;
    int neg = 0;
    for (int i = 0; i < prob_->l; i++)
        if (prob_->y[i] > 0)
            pos++;
    neg = prob_->l - pos;
    primal_solver_tol_ = eps_ * max(min(pos, neg), 1) / prob_->l;
    //MPI_Barrier(MPI_COMM_WORLD);
    if (myid_ == 0) {
        cout << "procnum=" << procnum_ << ",synchronize!!!" << endl;
        char *outfile = new char[100];
        sprintf(outfile, "../out/results_%d_%d_%d.dat", procnum_, barrier_size_,
                filter_type_);
        of_.open(outfile);
        of_ << "procnum=" << procnum_ << ",dim=" << dim_ << ",synchronize!!!"
           << endl;
        char filename[100];
        sprintf(filename, test_file_path.c_str(),procnum_, myid_);
        string dataname(filename);
        string inputfile = dataname;
        predprob_ = new problem(inputfile.c_str());
    }
}
ADMM::~ADMM() {
    delete[]w_;
    delete[]x_;
    delete[]y_;
    delete[]z_;
    // delete []C;
    delete[]z_pre_;
}
double ADMM::GetObjectValue() {
    int instance_num = predprob_->l;
    vector<double> hypothesis(instance_num, 0);
    //计算每个数据中的sigmoid函数的值
    //计算costFunction
    double costFunction = 0.0;
    for (int i = 0; i < instance_num; i++) {
        int j = 0;
        while (true) {
            if (predprob_->x[i][j].index == -1)
                break;
            hypothesis[i] +=
                    predprob_->x[i][j].value * z_[predprob_->x[i][j].index - 1];
            j++;
        }
        costFunction += std::log(
                1 + std::exp(-predprob_->y[i] * hypothesis[i]));
    }
    costFunction = costFunction / instance_num;
    return costFunction;
}

double ADMM::GetSVMObjectValue(int type) {
    int datanumber = prob_->l;
    vector<double> hypothesis(datanumber, 0);
    for (int i = 0; i < datanumber; i++) {
        int j = 0;
        while (true) {
            if (prob_->x[i][j].index == -1)
                break;
            hypothesis[i] += prob_->x[i][j].value * z_[prob_->x[i][j].index - 1];
            j++;
        }
    }
    //计算costFunction
    double costFunction = 0.0;
    if (type == 0)         // L2-SVM
    {
        for (int i = 0; i < datanumber; i++) {
            double d = 1 - prob_->y[i] * hypothesis[i];
            if (d > 0)
                costFunction += d * d;
        }
    } else {                                   //L1-SVM
        for (int i = 0; i < datanumber; i++) {
            double d = 1 - prob_->y[i] * hypothesis[i];
            if (d > 0)
                costFunction += d;
        }
    }
    costFunction = costFunction / datanumber;
    return costFunction;
}
void ADMM::x_update() {
    //  omp_set_num_threads(th_num);
    subproblem_tron();
//    scd_l1_svm(0.01,x,1,1);
//    subproblem_lbfgs();
//    subproblem_gd();
}

//void ADMM::subproblem_lbfgs() {
////    fun_obj_ = new l2r_l2_svc_fun(prob_,rho_);
//    fun_obj_ = new l2r_lr_fun(prob_, rho_);
//    lbfgs_obj_ = new LBFGS_OPT(fun_obj_);
//    lbfgs_obj_->optimizer(x_, y_, z_, statistical_time_);
//    delete fun_obj_;
//    delete lbfgs_obj_;
//}
//
//void ADMM::subproblem_gd() {
//    fun_obj_ = new l2r_l2_svc_fun(prob_,rho_);
////    fun_obj_ = new l2r_lr_fun(prob_, rho_);
//    gd_obj_ = new GD_OPT(fun_obj_);
//    gd_obj_->optimizer(x_, y_, z_, statistical_time_);
//    delete fun_obj_;
//    delete gd_obj_;
//}

void ADMM::subproblem_tron() {
    fun_obj_ = new l2r_lr_fun(prob_, rho_);
//    fun_obj_ = new l2r_l2_svc_fun(prob_,rho_);
    tron_obj_ = new TRON(fun_obj_, primal_solver_tol_, eps_cg_, 1000);
    tron_obj_->tron(x_, y_, z_, statistical_time_, tron_iteraton_, cg_iter_);
    //free(fun_obj); free(tron_obj);
    delete fun_obj_;
    delete tron_obj_;
}
void ADMM::y_update() {
    int i;
    for (i = 0; i < dim_; i++) {
        y_[i] += rho_ * (x_[i] - z_[i]);
    }
}
void ADMM::softThreshold(double t, double *z) {
    for (size_t i = 0; i < dim_; i++) {
        if (z[i] > t)
            z[i] -= t;
        else if (z[i] <= t && z[i] >= -t) {
            z[i] = 0.0;
        } else
            z[i] += t;
    }
}
void ADMM::z_update() {
    double s = 1.0 / (rho_ * procnum_ + 2 * l2reg_);
    double t = s * l1reg_;
    for (int i = 0; i < dim_; i++) {
        z_pre_[i] = z_[i];
    }
    double tmp;
    for (int i = 0; i < dim_; i++) {
        z_[i] = sum_w_[i] * s;
//        z_[i] = w_[i] * s;
    }
    if (hasL1reg_)
        softThreshold(t, z_);
}
bool ADMM::is_stop() {
    double *send = new double[3];
    double *rcv = new double[3];
    for (int i = 0; i < 3; i++) {
        send[i] = 0;
        rcv[i] = 0;
    }
    for (int i = 0; i < dim_; i++) {
        send[0] += (x_[i] - z_[i]) * (x_[i] - z_[i]);
        send[1] += x_[i] * x_[i];
        send[2] += y_[i] * y_[i];
    }
    MPI_Allreduce(send, rcv, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double prires = sqrt(rcv[0] / procnum_);
    double nxstack = sqrt(rcv[1] / procnum_);
    double nystack = sqrt(rcv[2] / procnum_);
    double zdiff = 0.0;
    double z_squrednorm = 0.0;
    for (int i = 0; i < dim_; i++) {
        zdiff += (z_[i] - z_pre_[i]) * (z_[i] - z_pre_[i]);
        z_squrednorm += z_[i] * z_[i];
    }
    double z_norm = sqrt(z_squrednorm);
    double dualres = rho_ * sqrt(zdiff);
    double eps_pri = sqrt(dim_) * ABSTOL + RELTOL * fmax(nxstack, z_norm);
    double eps_dual = sqrt(dim_) * ABSTOL + RELTOL * nystack;
    //if(myid==0)printf("%10.4f %10.4f %10.4f %10.4f\t", prires, eps_pri, dualres, eps_dual);
    //of<<prires<<"\t"<<eps_pri<<"\t"<<dualres<<"\t"<<eps_dual<<"\t";
    if (send != NULL) delete[] send;
    if (rcv != NULL) delete[] rcv;
    if (prires <= eps_pri && dualres <= eps_dual) {
        return true;
    }
    return false;
}
void ADMM::CreateGroup() {
    int color = myid_ / worker_per_group_;
    MPI_Comm_split(MPI_COMM_WORLD, color, myid_, &SUBGRP_COMM_);
    int subgrp_rank, subgrp_size;
    MPI_Comm_rank(SUBGRP_COMM_, &subgrp_rank);
    MPI_Comm_size(SUBGRP_COMM_, &subgrp_size);
    MPI_Group main_grp, world_grp;
    MPI_Comm_group(MPI_COMM_WORLD, &world_grp);// 返回通信域
    int group_num = procnum_ / worker_per_group_;
    int *maingrp_ranks = (int *) malloc(sizeof(int) * group_num);
    for (int i = 0; i < group_num; ++i) {
        maingrp_ranks[i] = i * worker_per_group_;
    }
    MPI_Group_incl(world_grp, group_num, maingrp_ranks, &main_grp);
    MPI_Comm_create_group(MPI_COMM_WORLD, main_grp, 0, &MAINGRP_COMM_);
}
void ADMM::train() {
    //initialization
    double begin_time, end_time, begin_synchronization, end_synchronization, start_calculation;
    double communication_time = 0;
    double total_synchronization_time(0);
    double synchronization_time(0);
    double max_synchronization_time(0);
    double update_time = 0;
    double max_update_time(0);
    double min_update_time(0);
    double total_average_update_time(0);
    double max_wait_time(0);
    double total_average_wait_time(0);
    double close_time = 0;
    double average_updata_time(0);
    int k = 0;
    double total_time(0), single_iteration_time(0);
    double average_wait_time = 0;
    int flag = 0;
    if(myid_ == 0)
        printf("%-3s %-4s %-11s %-10s %-10s %-10s %-10s \n", "#", "RANK", "ObjectValue", "Accuracy", "MUpdateTime", "MWaitTime", "SynchTime");
    MPI_Barrier(MPI_COMM_WORLD);
    CreateGroup();
    collective_->CreateTorus(MPI_COMM_WORLD, TORUS_COMM_, worker_per_group_, procnum_, nbrs_);
    begin_time = MPI_Wtime();
    while (k < max_iterations_) {
        //initialization
        /*for(int i = 0; i < 10; i++){
            cg_iter_[i] = 0;
        }*/
        double accuracy = 0.0;
        double object_value = 0.0;
        start_calculation = MPI_Wtime();
        x_update();
        for (int i = 0; i < dim_; i++) {
            w_[i] = rho_ * x_[i] + y_[i];
        }
        MPI_Barrier(MPI_COMM_WORLD);
        begin_synchronization = MPI_Wtime();
        MPI_Allreduce(w_, sum_w_, dim_, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
//        collective_->TorusAllreduce(w_, worker_per_group_, group_num_, TORUS_COMM_, nbrs_);
        collective_->RingAllreduce(w_, dim_, MPI_COMM_WORLD);
//        collective_->HierarchicalAllreduce(w_, MAINGRP_COMM_, SUBGRP_COMM_);
//        collective_->HierarchicalTorus(w_, MAINGRP_COMM_, SUBGRP_COMM_, nbrs_);
        end_synchronization = MPI_Wtime();
        z_update();
        /*if (is_stop()){
            flag = 1;
        }*/
        y_update();
        end_time = MPI_Wtime();
        k++;
        if(myid_ == 0){
            accuracy = predict(k);
            object_value = GetObjectValue();
        }
        single_iteration_time = end_time - start_calculation;
        /*calculate update_time/communication_time/synchronization_time*/
        synchronization_time = end_synchronization - begin_synchronization;
        update_time = begin_synchronization - start_calculation;
        MPI_Allreduce(&synchronization_time, &max_synchronization_time, 1, MPI_DOUBLE, MPI_MAX,
                      MPI_COMM_WORLD);
        /*MPI_Allreduce(&update_time, &min_update_time, 1, MPI_DOUBLE, MPI_MIN,
                      MPI_COMM_WORLD);
        MPI_Allreduce(&update_time, &max_update_time, 1, MPI_DOUBLE, MPI_MAX,
                      MPI_COMM_WORLD);*/
        MPI_Allreduce(&update_time, &average_updata_time, 1, MPI_DOUBLE,
                      MPI_SUM, MPI_COMM_WORLD);
        average_updata_time /= procnum_;
        total_synchronization_time += max_synchronization_time;
        average_wait_time = single_iteration_time - average_updata_time;
        total_average_wait_time += average_wait_time;
        total_average_update_time += average_updata_time;
        if(myid_ == 0){
            printf("%-3d %-4d %-11f %-10f %-10f %-10f %-10f \n",
                   k, myid_, object_value, accuracy,  update_time, average_wait_time, max_synchronization_time);
        }
    }
    close_time = MPI_Wtime();
    int training_time = close_time - begin_time;

    //include calculation communicaiton and wait time.
    if (myid_ == 0) {
        cout << "total training time is: " << training_time << "\n" << endl;
        cout << "total average update time is: " << total_average_update_time << "\n" << endl;
        cout << "total average wait time is: " << total_average_wait_time << "\n" << endl;
        cout << "total max synchronization time is: " << total_synchronization_time << "\n" << endl;
    }
}
double ADMM::predict(int last_iter) {
    //MPI_Allreduce(x,w,dim,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    int acnum = 0;
    int tp = 0;
    int fp = 0;
    int fn = 0;
    int tn = 0;
    int instance_num = predprob_->l;
    for (int i = 0; i < instance_num; i++) {
        feature_node **element = predprob_->x;
        double res = 1.0 / (1 + exp (-1 * sparse_operator::dot(z_, element[i])));
        if(predprob_->y[i] == 1 && res >= 0.5){
            ++acnum;
        }
        if(predprob_->y[i] == -1 && res < 0.5){
            ++acnum;
        }
        /*
        if(last_iter == max_iterations_){
            if (res >= 0.5 && (predprob_->y[i] > 0)) tp++;
            if (res >= 0.5 && (predprob_->y[i] < 0)) fp++;
            if (res < 0.5 && (predprob_->y[i] < 0)) tn++;
            if (res < 0.5 && (predprob_->y[i] > 0)) fn++;
            ofstream file("../roc_auc_webspam_128_30.csv",ios::app);
            if(file){
                file << predprob_->y[i] << "," << res << "\n";
            }
        }
         */
    }

    double p = (double) tp / (tp + fp);
    double r = (double) tp / (tp + fn);
    double ftr = (double) fp / (fp + tn);
    double acrate = (double) acnum / instance_num;
    return acrate;
}
void ADMM::draw() {
    if (myid_ == 0) {
        int acnum = 0;
        int tp = 0;
        int fp = 0;
        int fn = 0;
        int tn = 0;
        int datalen = predprob_->l;
        // int datalen = prob->l;
        double *gradValue = new double[51];
        for (int j = 0; j < 51; j++) {
            gradValue[j] = 0.02 * j;
        }
        for (int j = 0; j < 51; j++) {
            for (int i = 0; i < datalen; i++) {
                feature_node **data = predprob_->x;
                double res =
                        1.0 / (1 + exp(-1 * sparse_operator::dot(z_, data[i])));
                if (res > gradValue[j] && (predprob_->y[i] > 0)) tp++;
                if (res > gradValue[j] && (predprob_->y[i] < 0)) fp++;
                if (res <= gradValue[j] && (predprob_->y[i] <= 0)) tn++;
                if (res <= gradValue[j] && (predprob_->y[i] > 0)) fn++;
            }
            double p = (double) tp / (tp + fp);
            double r = (double) tp / (tp + fn);
            double ftr = (double) fp / (fp + tn);
            cout << p << "\t" << r << "\t" << ftr << "\n";
        }
    }
}
//
void test_main(MPI_Comm comm) {
    int myid, procnum;
    double begin_time, end_time;
    MPI_Init(NULL, NULL);
    MPI_Comm_size(comm, &procnum);
    MPI_Comm_rank(comm, &myid);
    /*write a optimizer selection*/
    Properties properties("../conf/admm.conf");  //读取默认的配置文件
    string train_data_path = properties.GetString("train_data_path");
    string test_data_path = properties.GetString("test_data_path");
    int maxdim;
    maxdim = properties.GetInt("dimension");
    if (myid == 0)
        cout << "dim=" << maxdim << endl;
    char filename[100];
    problem *prob;
    sprintf(filename, train_data_path.c_str(), procnum, myid);
    string dataname(filename);
    string inputfile = dataname;
    prob = new problem(inputfile.c_str());
    prob->n = maxdim;
    args_t *args = new args_t(myid, procnum, properties);
    Collective *collective = new Collective(args, prob);
    begin_time=MPI_Wtime(); //clock();
    ADMM admm(args, prob, test_data_path, collective);
    admm.train();                 //未过滤
    end_time=MPI_Wtime();//clock();
    //admm.draw();
    if(myid == 0)
    {    cout << "Total train time is " <<  (double)(end_time-begin_time)<< " second." << endl;
        // cout<<"accracy:"<<admm.predict()<<endl;
    }
    MPI_Finalize();
}

void test_main3(MPI_Comm comm) {
    int rank, size;
    MPI_Init(NULL, NULL); /* starts MPI */
    MPI_Comm_rank(comm, &rank); /* get current process id */
    MPI_Comm_size(comm, &size); /* get number of processes */
    printf("Hello world from process %d of %d\n", rank, size);
    MPI_Finalize();
}
