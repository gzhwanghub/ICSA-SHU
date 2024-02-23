#include "prob.h"


problem::problem(const char *filename)
{
    read_problem(filename);
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

	l = 0;
	elements = 0;
	max_line_len = 1024;
	line = Malloc(char,max_line_len);
	while(readline(fp)!=NULL)
	{
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
	rewind(fp);

	bias=-1;

	y = Malloc(double,l);
	x = Malloc(struct feature_node *,l);
	x_space = Malloc(struct feature_node,elements+l);

	max_index = 0;
	j=0;
	for(i=0;i<l;i++)
	{
		inst_max_index = 0; // strtol gives 0 if wrong format
		readline(fp);
		x[i] = &x_space[j];
		label = strtok(line," \t\n");
		if(label == NULL) // empty line
			exit_input_error(i+1);

		y[i] = strtod(label,&endptr);
		if(y[i] == 0)
			y[i] = -1;
		if(endptr == label || *endptr != '\0')
			exit_input_error(i+1);

		while(1)
		{
			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;

			errno = 0;
			x_space[j].index = (int) strtol(idx,&endptr,10);
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

char * problem::readline(FILE *input)
{
    int len;

	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
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
