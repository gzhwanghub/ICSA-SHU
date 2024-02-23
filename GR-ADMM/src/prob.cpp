//
// Created by cluster on 2020/10/14.
//

#include "../include/prob.h"

problem::problem(const char *filename)
{
    read_problem(filename);
    //cout<<filename<<endl;
}

void problem::read_problem(const char *filename)
{
    int max_index, inst_max_index, i;
    size_t elements, j;
    FILE *fp = fopen(filename,"r");
    char *endptr;
    char *idx, *val, *label;

    if(fp == NULL)
    {
        fprintf(stderr,"can't open input file %s\n",filename);
        exit(1);
    }

    l = 0;//行数
    elements = 0;//总的元素数
    max_line_len = 1024;
    line = Malloc(char,max_line_len);
    while(readline(fp)!=NULL)
    {
        //分解字符串为一组字符串。line为要分解的字符串，delim为分隔符字符串。
        char *p = strtok(line," \t"); // label

        // features
        while(1)
        {
            p = strtok(NULL," \t");
            if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
                break;
            elements++;
        }
        elements++; // for bias term
        l++;
    }

    rewind(fp);//功能是将文件内部的指针重新指向一个流的开头
    bias=-1;

    y = Malloc(double,l);
    x = Malloc(struct feature_node *,l);//** 二维数组
    x_space = Malloc(struct feature_node,elements+l); //一维数组

    max_index = 0;
    j=0;
    for(i=0;i<l;i++)
    {
        //数据格式  标签（1或0）  列数：值
        inst_max_index = 0; // strtol gives 0 if wrong format
        readline(fp);
        x[i] = &x_space[j];
        label = strtok(line," \t\n");
            if(label == NULL) // empty line
            exit_input_error(i+1);

        y[i] = strtod(label,&endptr);//功能是将字符串转换成浮点数
        if(endptr == label || *endptr != '\0')
            exit_input_error(i+1);

        while(1)
        {
            idx = strtok(NULL,":");  //列数
            val = strtok(NULL," \t"); //值

            if(val == NULL)
                break;

            errno = 0;
            //x_space存储的是 feature_node包括index和value
            x_space[j].index = (int) strtol(idx,&endptr,10);//将参数nptr字符串根据参数base来转换成长整型数 base进制
            if(endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
                exit_input_error(i+1);
            else
                inst_max_index = x_space[j].index;

            errno = 0;
            x_space[j].value = strtod(val,&endptr);
            if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
                exit_input_error(i+1);

            ++j;
        }

        if(inst_max_index > max_index)
            max_index = inst_max_index;

        if(bias >= 0)
            x_space[j++].value = bias;

        x_space[j++].index = -1;
    }

    if(bias >= 0)
    {
        n=max_index+1;
        for(i=1;i<l;i++)
            (x[i]-2)->index = n;
        x_space[j-2].index = n;
    }
    else
        n=max_index;

    fclose(fp);
}
/// 读取一行数据
/// \param input
/// \return
char * problem::readline(FILE *input)
{
    int len;
    //读取1024个 或者到\n
    if(fgets(line,max_line_len,input) == NULL)//从指定的流 stream 读取一行，并把它存储在 str 所指向的字符串内。当读取 (n-1) 个字符时，或者读取到换行符时，或者到达文件末尾时，它会停止
        return NULL;
    //加入1024个中没有\n 说明没有一行 继续读取
    while(strrchr(line,'\n') == NULL)//查找字符在指定字符串中从右面开始的第一次出现的位置，如果成功，返回该字符以及其后面的字符，如果失败，则返回 NULL。
    {
        max_line_len *= 2;
        line = (char *) realloc(line,max_line_len);

        len = (int) strlen(line);
        if(fgets(line+len,max_line_len-len,input) == NULL)
            break;
    }
    return line;
}

void problem::exit_input_error(int line_num)
{
    fprintf(stderr,"Wrong input format at line %d\n", line_num);
    exit(1);
}
