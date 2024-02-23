#ifndef COMMUNICATION_H
#define COMMUNICATION_H

#undef SEEK_SET
#undef SEEK_END
#undef SEEK_CUR

 
#include<mpi.h>
#include<iostream>
#include<cmath>
#include<fstream>
#include<time.h>
#include<stdio.h>
 

void sparse_split_scannter(double* buffer, int procnum,int myid,int dim)   //split-ring-reduce
{
    int worker_number= procnum;
	if(worker_number==1) return;  
	int block_size = dim / worker_number;
    std::map<int, std::pair<int, int>> blocks;
    for (int i = 0; i < worker_number; ++i) {
        blocks[i].first = i * block_size;
        blocks[i].second = block_size;
    }
    blocks[worker_number - 1].second = (dim - (worker_number - 1) * block_size);
    int left = -1, right = -1;
    
   

    bool need_check = true;
    int isize = sizeof(int);
    int vsize = sizeof(double);
    int *index_buffer = new int[dim];
    double *value_buffer = new double[dim];
    double *recv_buffer = new double[dim];
   
    for (int i = 0; i < worker_number - 1; ++i) {
	left=(myid-i-1+worker_number)%worker_number;
	right=(myid+i+1)%worker_number;
        MPI_Status statuses[2];
        MPI_Request requests[2];
        //double *base = buffer + blocks[right].first;
        block_size = blocks[right].second;
	double *base=new double[block_size];
	int start=blocks[right].first;
	int nnz = 0;
        for (int j = 0; j < block_size; ++j)
	{
	    base[j]=buffer[j+start];
            if (base[j] != 0) 
	    {
                 ++nnz;
            }
        }
       // if(myid==0) printf("non zero number:%d\n",nnz);
        //如果上一次迭代接收到了稠密的块，那么本次传输就不用检查了，必然稠密
        if (need_check) {
           /* int nnz = 0;
            for (int j = 0; j < block_size; ++j) {
                if (base[j] != 0) {
                    ++nnz;
                }
            }*/
            //如果满足稀疏传输条件则采用稀疏传输
            if (nnz * (isize + vsize) < block_size * vsize) {
                int k = 0;
                for (int j = 0; j < block_size; ++j) {
                    if (base[j] != 0) {
                        value_buffer[k] = base[j];
                        index_buffer[k] = j;
                        ++k;
                    }
                }
                //把元素的值和元素的索引放到同一个内存空间上，一同发送，不分两次发送了
                memcpy(value_buffer + nnz, index_buffer, isize * nnz);
                MPI_Isend(value_buffer, nnz * (isize + vsize), MPI_CHAR, right,0, MPI_COMM_WORLD, &requests[0]);
            } else {
                MPI_Isend(base, block_size * vsize, MPI_CHAR, right, 0, MPI_COMM_WORLD,
                          &requests[0]);
            }
        } else {
            MPI_Isend(base, block_size * vsize, MPI_CHAR, right, 0, MPI_COMM_WORLD,
                      &requests[0]);
        }
        //base = buffer + blocks[myid].first;
        block_size = blocks[myid].second;
		start= blocks[myid].first;
        MPI_Irecv(recv_buffer, block_size * vsize, MPI_CHAR, left, 0, MPI_COMM_WORLD, &requests[1]);
        MPI_Waitall(2, requests, statuses);
        nnz = 0;
        MPI_Get_count(&statuses[1], MPI_CHAR, &nnz);
        //如果接收到的字节数少于这个块应有的大小，说明左边进程发送的是稀疏数据
        if (nnz < block_size * vsize) {
           
            nnz = nnz / (isize + vsize);
            //分离值和索引
            memcpy(index_buffer, recv_buffer + nnz, nnz * isize);
            for (int j = 0; j < nnz; ++j) {
               // Reduce<O>(base[index_buffer[j]], recv_buffer[j]);
				buffer[index_buffer[j]+start]+=recv_buffer[j];
            }
           // need_check = true;
        } else {
            
            for (int j = 0; j < block_size; ++j) {
               // Reduce<O>(base[j], recv_buffer[j]);
				buffer[j+start]+=recv_buffer[j];
            }
           // need_check = false;
        }
         delete []base;
        
    }
    need_check = true;
    for (int i = 0; i < worker_number - 1; ++i) {
        MPI_Status statuses[2];
        MPI_Request requests[2];
        
		right=(myid+1)%worker_number;
		left=(myid+worker_number-1)%worker_number;
		int sendBlockId=(myid+worker_number-i)%worker_number;
		int recvBlockId=(myid+worker_number-i-1)%worker_number;
		//double *base = buffer + blocks[sendBlockId].first;
        block_size = blocks[sendBlockId].second;
		int start=blocks[sendBlockId].first;
		double *base=new double[block_size];
		int nnz = 0;
        for (int j = 0; j < block_size; ++j) {
			base[j]=buffer[start+j];
            if (base[j] != 0) {
                    ++nnz;
            }
        }
        if (need_check) {
            
            if (nnz * (isize + vsize) < block_size * vsize) {
                int k = 0;
                for (int j = 0; j < block_size; ++j) {
                    if (base[j] != 0) {
                        value_buffer[k] = base[j];
                        index_buffer[k] = j;
                        ++k;
                    }
                }
                memcpy(value_buffer + nnz, index_buffer, isize * nnz);
                MPI_Isend(value_buffer, nnz * (isize + vsize), MPI_CHAR, right,
                          1, MPI_COMM_WORLD, &requests[0]);
            } else {
                MPI_Isend(base, block_size * vsize, MPI_CHAR, right, 1, MPI_COMM_WORLD,
                          &requests[0]);
            }
        } else {
            MPI_Isend(base, block_size * vsize, MPI_CHAR, right, 1, MPI_COMM_WORLD,
                      &requests[0]);
        }
        //base = buffer + blocks[recvBlockId].first;
        block_size = blocks[recvBlockId].second;
		start = blocks[recvBlockId].first;
        MPI_Irecv(recv_buffer, block_size * vsize, MPI_CHAR, left, 1, MPI_COMM_WORLD, &requests[1]);
        MPI_Waitall(2, requests, statuses);
        nnz = 0;
        MPI_Get_count(&statuses[1], MPI_CHAR, &nnz);
        if (nnz < block_size * vsize) {
            
            nnz = nnz / (isize + vsize);
            memcpy(index_buffer, recv_buffer + nnz, nnz * isize);
          
            for (int j = 0; j < nnz; ++j) {
                //base[index_buffer[j]] = recv_buffer[j];
				buffer[index_buffer[j]+start] = recv_buffer[j];
            }
           // need_check = true;
        } else {
            
            for (int j = 0; j < block_size; ++j) {
                //base[j] = recv_buffer[j];
				buffer[j+start] = recv_buffer[j];
            }
            //need_check = false;
        }
         delete []base;
        //send_block_index = recv_block_index;
       // recv_block_index = (recv_block_index - 1 + worker_number) % worker_number;
    }

    delete[] recv_buffer;
    delete[] value_buffer;
    delete[] index_buffer;	
}

void sparse_heir_allgather(double* buffer, int procnum,int myid,int dim) 
{
	if (procnum == 1) return;
    int total_num=1;
	int number=pow(2,total_num);
	while(number<procnum)
	{
		total_num++;
		number=pow(2,total_num);
	}
	int dis=1;
    int thred=2;
    int commId=0;
	bool need_check=true;
	int isize = sizeof(int);
    int vsize = sizeof(double);
	int *index_buffer = new int[dim];
	double *base=buffer;
    double *value_buffer = new double[2*dim];
	double *recv_buffer = new double[dim];
		
	for(int k=total_num;k>0;k--)       //  recursive doubling
    {
        MPI_Status statuses[2];
        MPI_Request requests[2];   
	    dis=pow(2,k-1);
	    thred=dis*2;
        if(myid%thred+dis<thred)
        {
            commId=myid+dis;
        }else{
         
            commId=myid-dis;
        }
        			
        if(need_check)
		{
			int nnz=0;
			for(int j=0;j<dim;j++)
			{
				if(base[j]!=0)
				{
					nnz++;
				}
			}
			if(nnz*(isize + vsize) < dim * vsize)
			{
				int l=0;
				for(int j=0;j<dim;j++)
				{
					if(base[j]!=0)
					{
						value_buffer[l] = base[j];
                        index_buffer[l] = j;
                        ++l;
					}						
				}
				memcpy(value_buffer + nnz, index_buffer, isize * nnz);
                MPI_Isend(value_buffer, nnz * (isize + vsize), MPI_CHAR, commId,0, MPI_COMM_WORLD, &requests[0]);
			}else{
				MPI_Isend(base, dim * vsize, MPI_CHAR, commId, 0, MPI_COMM_WORLD,&requests[0]);
			}
		}else{
			MPI_Isend(base, dim * vsize, MPI_CHAR, commId, 0, MPI_COMM_WORLD,&requests[0]);
		}
		//if(myid==0) cout<<"******"<<endl;
        int rcv_acnt=0;
		MPI_Status recvstatus;
		MPI_Probe(commId,MPI_ANY_TAG,MPI_COMM_WORLD,&recvstatus);
		MPI_Get_count(&recvstatus, MPI_CHAR, &rcv_acnt);  
		//char *recv_buf=new char[rcv_acnt];
		MPI_Irecv(recv_buffer,rcv_acnt,MPI_CHAR,commId,0,MPI_COMM_WORLD,&requests[1]);
		MPI_Waitall(2, requests, statuses);
		if(rcv_acnt<dim * vsize)
		{
			//分离值和索引
			int indexNum=rcv_acnt/(isize + vsize);
			memcpy(index_buffer, recv_buffer + indexNum, indexNum * isize);
			for(int j=0;j<indexNum;j++)
			{
				//if(buffer[index_buffer[j]]==0) nnz++;
				buffer[index_buffer[j]]+=recv_buffer[j];
			}
		}else{
			for (int j = 0; j < dim; ++j) {
               // Reduce<O>(base[j], recv_buf[j]);
				buffer[j]+=recv_buffer[j];
            }
			need_check = false;
		}
		//if(myid==0) cout<<"##########"<<endl;
    } 
	 
	delete []recv_buffer;
    delete[] value_buffer;
    delete[] index_buffer;
	//if(myid==0) cout<<"%%%%%%%%%%%%"<<endl;
}
void sparse_allgather(double* buffer, int procnum,int myid,int dim) 
{
	if (procnum == 1) return;
    int total_num=1;
	int number=pow(2,total_num);
	while(number<procnum)
	{
		total_num++;
		number=pow(2,total_num);
	}
	int dis=1;
    int thred=2;
    int commId=0;
	bool need_check=true;
	int isize = sizeof(int);
    int vsize = sizeof(double);
	int *index_buffer = new int[dim];
	double *base=buffer;
    double *value_buffer = new double[2*dim];
	double *recv_buffer = new double[dim];
	for(int k=0;k<total_num;k++)       //  recursive doubling
    {
        MPI_Status statuses[2];
        MPI_Request requests[2];   
	    dis=pow(2,k);
        thred=dis*2;
        if(myid%thred+dis<thred)
        {
            commId=myid+dis;
        }else{
         
            commId=myid-dis;
        }
        int nnz=0;
        for(int j=0;j<dim;j++)
     	{
			if(base[j]!=0)
			{
				nnz++;
			}
		}			
        if(need_check)
		{
			if(nnz*(isize + vsize) < dim * vsize)
			{
				int k=0;
				for(int j=0;j<dim;j++)
				{
					if(base[j]!=0)
					{
						value_buffer[k] = base[j];
                        index_buffer[k] = j;
                        ++k;
					}						
				}
				memcpy(value_buffer + nnz, index_buffer, isize * nnz);
                MPI_Isend(value_buffer, nnz * (isize + vsize), MPI_CHAR, commId,0, MPI_COMM_WORLD, &requests[0]);
			}else{
				MPI_Isend(base, dim * vsize, MPI_CHAR, commId, 0, MPI_COMM_WORLD,&requests[0]);
			}
		}else{
			MPI_Isend(base, dim * vsize, MPI_CHAR, commId, 0, MPI_COMM_WORLD,&requests[0]);
		}
		//if(myid==0) cout<<"******"<<endl;
        int rcv_acnt=0;
		MPI_Status recvstatus;
		MPI_Probe(commId,MPI_ANY_TAG,MPI_COMM_WORLD,&recvstatus);
		MPI_Get_count(&recvstatus, MPI_CHAR, &rcv_acnt);  
		//char *recv_buf=new char[rcv_acnt];
		MPI_Irecv(recv_buffer,rcv_acnt,MPI_CHAR,commId,0,MPI_COMM_WORLD,&requests[1]);
		MPI_Waitall(2, requests, statuses);
		if(rcv_acnt<dim * vsize)
		{
			//分离值和索引
			int indexNum=rcv_acnt/(isize + vsize);
			memcpy(index_buffer, recv_buffer + indexNum, indexNum * isize);
			for(int j=0;j<indexNum;j++)
			{
				buffer[index_buffer[j]]+=recv_buffer[j];
			}
		}else{
			for (int j = 0; j < dim; ++j) {
               // Reduce<O>(base[j], recv_buf[j]);
				buffer[j]+=recv_buffer[j];
            }
			need_check = false;
		}
		//if(myid==0) cout<<"##########"<<endl;
    } 
	//delete []base;
	delete []recv_buffer;
    delete[] value_buffer;
    delete[] index_buffer;
	//if(myid==0) cout<<"%%%%%%%%%%%%"<<endl;
}
void sparse_split_allgather(double* buffer, int procnum,int myid,int dim)  //split-allgather
{
    int worker_number= procnum;
	if(worker_number==1) return;
	
	int block_size = dim / worker_number;
    std::map<int, std::pair<int, int>> blocks;
	for (int i = 0; i < worker_number; ++i) {
        blocks[i].first = i * block_size;
        blocks[i].second = block_size;
    } 
	blocks[worker_number-1].second = dim - (worker_number - 1) * block_size;
	int right=-1;
	int left=-1;
	bool need_check = true;
	int isize = sizeof(int);
    int vsize = sizeof(double);
	double *send_buffer=new double[dim];
	double *value_buffer = new double[dim];
	int *index_buffer=new int[dim];
	 
	double *recv_buffer=new double[dim];
	double *base;
	//split 阶段
	for(int i=0;i<worker_number-1;++i)
	{
		left=(myid-i-1+worker_number)%worker_number;
	 	right=(myid+i+1)%worker_number;
		MPI_Status statuses[2];
        MPI_Request requests[2];
		base=buffer+blocks[right].first;
		block_size=blocks[right].second;
 
		if(need_check)//是否过滤标志
		{
			int nnz=0;
			for(int j=0;j<block_size;j++)
			{
				if(base[j]!=0)
				{
					++nnz;
				}
			}
			 
			if(nnz*(isize+vsize)<block_size * vsize)//满足稀疏传送条件
			{
				int k=0;
				for(int j=0;j<block_size;++j)
				{
					if(base[j]!=0)
					{
						value_buffer[k]=base[j];
						index_buffer[k]=j;
						++k;
					}					
				}
				memcpy(value_buffer+nnz,index_buffer,isize*nnz); //将对应索引值放在value值的后面
				//发送稀疏数据
				MPI_Isend(value_buffer,nnz * (isize + vsize), MPI_CHAR,right, 0,MPI_COMM_WORLD,&requests[0]);
			}else{
				MPI_Isend(base, block_size * vsize, MPI_CHAR, right, 0, MPI_COMM_WORLD,&requests[0]);
			}
		}else{
			MPI_Isend(base, block_size * vsize, MPI_CHAR, right, 0, MPI_COMM_WORLD,&requests[0]);
		}
		 base=buffer+blocks[myid].first;
		 block_size = blocks[myid].second;
 
		MPI_Irecv(recv_buffer, block_size * vsize, MPI_CHAR, left, 0, MPI_COMM_WORLD, &requests[1]);
        MPI_Waitall(2, requests, statuses);
		int nnz=0;
		MPI_Get_count(&statuses[1], MPI_CHAR, &nnz);
		 
		if(nnz<block_size * vsize)//接收到稀疏数据
		{
		    //CHECK_EQ(nnz % (isize + vsize), 0);
			nnz = nnz / (isize + vsize);
			memcpy(index_buffer, recv_buffer + nnz, nnz * isize);
			for(int j=0;j<nnz;j++)
			{
				base[index_buffer[j]]+=recv_buffer[j];
				 
			}
		}else{//接收到稠密数据
		   for (int j = 0; j < block_size; ++j) {
                base[j]+=recv_buffer[j];
				 
            }
		}
	 
	}
	//allgather 阶段
	need_check=true;
	int total_num=1;
	int number=pow(2,total_num);
	while(number<worker_number)
	{
		total_num++;
		number=pow(2,total_num);
	}
	
	//cout<<"********"<<endl;
	int sendBlockSize= blocks[myid].second;
	base=buffer+blocks[myid].first;
	int start_index=blocks[myid].first;
	int recvBlockSize=0;
	int dis=1; 
	int thred=2;
	double *sBuf=new double[dim];
	int *index=new int[dim];
	int nnz_cnt=0;
	for(int i=0;i<sendBlockSize;i++)
	{
		if(base[i]!=0)
		{
			sBuf[nnz_cnt]=base[i];
			index[nnz_cnt]=i+start_index;
			nnz_cnt++;
		}
		//sBuf[i]=base[i];
	}
	 
	double *send_buf=new double[dim*2];
	double *recv_buf=new double[dim];
	 
	for(int i=0;i<total_num;i++)
	{
		MPI_Status statuses[2];
        MPI_Request requests[2];
		
		dis=pow(2,i);
	    thred=dis*2;
		 
	    if(myid%thred+dis<thred)
	    {
		    right= myid+dis;
	    }else{
		
		    right=myid-dis;
	    }
        
		memcpy(send_buf,sBuf,nnz_cnt*vsize); 
		memcpy(send_buf+nnz_cnt,index,nnz_cnt*isize);
		
		MPI_Isend(send_buf,nnz_cnt*(vsize+isize),MPI_CHAR,right,1,MPI_COMM_WORLD,&requests[0]);
		
		int rcv_acnt=0;
		MPI_Status recvstatus;
		MPI_Probe(right,MPI_ANY_TAG,MPI_COMM_WORLD,&recvstatus);
		MPI_Get_count(&recvstatus, MPI_CHAR, &rcv_acnt);
		char *recv_buf=new char[rcv_acnt];
		MPI_Irecv(recv_buf,rcv_acnt,MPI_CHAR,right,1,MPI_COMM_WORLD,&requests[1]);
		MPI_Wait(&requests[0],&statuses[0]); 
        MPI_Wait(&requests[1],&statuses[1]);
		
		recvBlockSize=rcv_acnt/(vsize+isize);
		memcpy(sBuf+nnz_cnt,recv_buf,recvBlockSize*vsize);
		memcpy(index+nnz_cnt,recv_buf+recvBlockSize*vsize,recvBlockSize*isize);
		nnz_cnt+=recvBlockSize;
		delete[] recv_buf;
	}
 	 
	for(int i=0;i<nnz_cnt;i++)
	{
		 
		  buffer[index[i]]=sBuf[i];
	}
	 
	delete[] send_buf;
	delete[] index;
	delete[] sBuf;
	delete[] recv_buffer;
    delete[] send_buffer;
    delete[] index_buffer;
}
void heir_split_allgather(double* buffer, int procnum,int myid,int dim)  //split-allgather
{
    int worker_number= procnum;
	if(worker_number==1) return;
	
	int block_size = dim / worker_number;
    std::map<int, std::pair<int, int>> blocks;
	for (int i = 0; i < worker_number; ++i) {
        blocks[i].first = i * block_size;
        blocks[i].second = block_size;
    } 
	blocks[worker_number-1].second = dim - (worker_number - 1) * block_size;
	int right=-1;
	int left=-1;
	bool need_check = true;
	int isize = sizeof(int);
    int vsize = sizeof(double);
	double *send_buffer=new double[dim];
	double *value_buffer = new double[dim];
	int *index_buffer=new int[dim];
	 
	double *recv_buffer=new double[dim];
	double *base;
	//split 阶段
	for(int i=0;i<worker_number-1;++i)
	{
		left=(myid-i-1+worker_number)%worker_number;
	 	right=(myid+i+1)%worker_number;
		MPI_Status statuses[2];
        MPI_Request requests[2];
		base=buffer+blocks[right].first;
		block_size=blocks[right].second;
 
		if(need_check)//是否过滤标志
		{
			int nnz=0;
			for(int j=0;j<block_size;j++)
			{
				if(base[j]!=0)
				{
					++nnz;
				}
			}
			 
			if(nnz*(isize+vsize)<block_size * vsize)//满足稀疏传送条件
			{
				int k=0;
				for(int j=0;j<block_size;++j)
				{
					if(base[j]!=0)
					{
						value_buffer[k]=base[j];
						index_buffer[k]=j;
						++k;
					}					
				}
				memcpy(value_buffer+nnz,index_buffer,isize*nnz); //将对应索引值放在value值的后面
				//发送稀疏数据
				MPI_Isend(value_buffer,nnz * (isize + vsize), MPI_CHAR,right, 0,MPI_COMM_WORLD,&requests[0]);
			}else{
				MPI_Isend(base, block_size * vsize, MPI_CHAR, right, 0, MPI_COMM_WORLD,&requests[0]);
			}
		}else{
			MPI_Isend(base, block_size * vsize, MPI_CHAR, right, 0, MPI_COMM_WORLD,&requests[0]);
		}
		 base=buffer+blocks[myid].first;
		 block_size = blocks[myid].second;
 
		MPI_Irecv(recv_buffer, block_size * vsize, MPI_CHAR, left, 0, MPI_COMM_WORLD, &requests[1]);
        MPI_Waitall(2, requests, statuses);
		int nnz=0;
		MPI_Get_count(&statuses[1], MPI_CHAR, &nnz);
		 
		if(nnz<block_size * vsize)//接收到稀疏数据
		{
		    //CHECK_EQ(nnz % (isize + vsize), 0);
			nnz = nnz / (isize + vsize);
			memcpy(index_buffer, recv_buffer + nnz, nnz * isize);
			for(int j=0;j<nnz;j++)
			{
				base[index_buffer[j]]+=recv_buffer[j];
				 
			}
		}else{//接收到稠密数据
		   for (int j = 0; j < block_size; ++j) {
                base[j]+=recv_buffer[j];
				 
            }
		}
	 
	}
	//allgather 阶段
	need_check=true;
	int total_num=1;
	int number=pow(2,total_num);
	while(number<worker_number)
	{
		total_num++;
		number=pow(2,total_num);
	}
	
	//cout<<"********"<<endl;
	int sendBlockSize= blocks[myid].second;
	base=buffer+blocks[myid].first;
	int start_index=blocks[myid].first;
	int recvBlockSize=0;
	int dis=1; 
	int thred=2;
	double *sBuf=new double[dim];
	int *index=new int[dim];
	int nnz_cnt=0;
	for(int i=0;i<sendBlockSize;i++)
	{
		if(base[i]!=0)
		{
			sBuf[nnz_cnt]=base[i];
			index[nnz_cnt]=i+start_index;
			nnz_cnt++;
		}
		//sBuf[i]=base[i];
	}
	 
	double *send_buf=new double[dim*2];
	double *recv_buf=new double[dim];
	 
	for(int i=total_num;i>0;i--)  //先节点间再节点内通信
	{
		MPI_Status statuses[2];
        MPI_Request requests[2];
		
		dis=pow(2,i-1);
	    thred=dis*2;
		 
	    if(myid%thred+dis<thred)
	    {
		    right= myid+dis;
	    }else{
		
		    right=myid-dis;
	    }
        
		memcpy(send_buf,sBuf,nnz_cnt*vsize); 
		memcpy(send_buf+nnz_cnt,index,nnz_cnt*isize);
		
		MPI_Isend(send_buf,nnz_cnt*(vsize+isize),MPI_CHAR,right,1,MPI_COMM_WORLD,&requests[0]);
		
		int rcv_acnt=0;
		MPI_Status recvstatus;
		MPI_Probe(right,MPI_ANY_TAG,MPI_COMM_WORLD,&recvstatus);
		MPI_Get_count(&recvstatus, MPI_CHAR, &rcv_acnt);
		char *recv_buf=new char[rcv_acnt];
		MPI_Irecv(recv_buf,rcv_acnt,MPI_CHAR,right,1,MPI_COMM_WORLD,&requests[1]);
		MPI_Wait(&requests[0],&statuses[0]); 
        MPI_Wait(&requests[1],&statuses[1]);
		
		recvBlockSize=rcv_acnt/(vsize+isize);
		memcpy(sBuf+nnz_cnt,recv_buf,recvBlockSize*vsize);
		memcpy(index+nnz_cnt,recv_buf+recvBlockSize*vsize,recvBlockSize*isize);
		nnz_cnt+=recvBlockSize;
		delete[] recv_buf;
	}
 	 
	for(int i=0;i<nnz_cnt;i++)
	{
		 
		  buffer[index[i]]=sBuf[i];
	}
	 
	delete[] send_buf;
	delete[] index;
	delete[] sBuf;
	delete[] recv_buffer;
    delete[] send_buffer;
    delete[] index_buffer;
} 
void RingAllreduce(float* buffer, int procnum,int myid,int dim) {
    //worker_list包含了参与此次Allreduce的所有进程id
    int worker_number = procnum;
    if (worker_number == 1) return;
    

    //对数组进行分块，在blocks中记录每一块的起始地址和块大小
    int block_size = dim / worker_number;
    std::map<int, std::pair<int, int>> blocks;
    for (int i = 0; i < worker_number; ++i) {
        blocks[i].first = i * block_size;
        blocks[i].second = block_size;
    }
    //由于数组不一定能均分，因此最后一块的大小需要计算
    blocks[worker_number - 1].second = (dim - (worker_number - 1) * block_size);
    //确定本进程在worker_list中的索引，并确定左右两个进程的id
    int left = -1, right = -1;

    left=(myid-1+worker_number)%worker_number;
    right=(myid+1)%worker_number;
    bool need_check = true;
    int isize = sizeof(int);
    int vsize = sizeof(float);
    float *recv_buffer = new float[dim];
    int *index_buffer = new int[dim];
    float *value_buffer = new float[dim];
    //每个进程从发送my_index这个块开始，因为每个进程的索引肯定不同，因此发送的是不同的块
    //每个进程是从左边进程接收数据，左边进程发送的是my_index-1这个块
    int send_block_index = myid;
    int recv_block_index = (send_block_index - 1 + worker_number) % worker_number;
    //一共需要worker_number-1次迭代，从左边进程接收数据，发送数据给右边进程
    for (int i = 0; i < worker_number - 1; ++i) {
        MPI_Request requests[2];
	MPI_Status statuses[2];
	int nnz = 0;
	int start = blocks[send_block_index].first;
	int block_size = blocks[send_block_index].second;
	float *temp = new float[block_size];
	for(int j = 0; j < block_size; ++j){
	    temp[j] = buffer[start+j];
	    if(temp[j] != 0)
		++nnz;
	}
	if(need_check){
	    if(nnz * (isize +vsize) < block_size * vsize){
		int k = 0;
		for(int j = 0; j < block_size; ++j){
		    if(temp[j] != 0){
			value_buffer[k] = temp[j];
			index_buffer[k] = j;
			++k;
		    }
		}
		memcpy(value_buffer + nnz, index_buffer, isize * nnz);
		MPI_Isend(value_buffer, nnz * (isize + vsize), MPI_CHAR, right, 0, MPI_COMM_WORLD, &requests[0]);
	    } else{
		MPI_Isend(temp, block_size * vsize, MPI_CHAR, right, 0, MPI_COMM_WORLD, &requests[0]);
	    }
	} else{
	    MPI_Isend(temp, block_size * vsize, MPI_CHAR, right, 0, MPI_COMM_WORLD, &requests[0]);
	}
	block_size = blocks[recv_block_index].second;
	start = blocks[recv_block_index].first;
	MPI_Irecv(recv_buffer, block_size * vsize, MPI_CHAR, left, 0, MPI_COMM_WORLD, &requests[1]);
	MPI_Waitall(2, requests, statuses);
	nnz = 0;
	MPI_Get_count(&statuses[1], MPI_CHAR, &nnz);
	if(nnz < block_size * vsize){
	    nnz = nnz / (isize + vsize);
	    memcpy(index_buffer, recv_buffer + nnz, nnz * isize);
	    for(int j = 0; j < nnz; ++j)
		buffer[index_buffer[j]+start] += recv_buffer[j];
	} else{
	    for(int j = 0; j < block_size; ++ j)
		buffer[start+j] += recv_buffer[j];
	}
	delete []temp;

        /*MPI_Isend(buffer + blocks[send_block_index].first, blocks[send_block_index].second, MPI_FLOAT,right,
              0, MPI_COMM_WORLD, &requests[0]);
        MPI_Irecv(recv_buffer, blocks[recv_block_index].second, MPI_FLOAT, left, 0, MPI_COMM_WORLD,
              &requests[1]);
        MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
        float *base = buffer + blocks[recv_block_index].first;
        block_size = blocks[recv_block_index].second;
        //接收到块后要Reduce到本进程对应位置的块上
        for (int j = 0; j < block_size; ++j) {
            //Reduce<O>(base[j], recv_buffer[j]);
            base[j]+=recv_buffer[j];
        }*/
        //下一次迭代发送本次迭代接收到的块
        send_block_index = recv_block_index;
        recv_block_index = (recv_block_index - 1 + worker_number) % worker_number;
    }

    for (int i = 0; i < worker_number - 1; ++i) {
        MPI_Request requests[2];
        MPI_Isend(buffer + blocks[send_block_index].first, blocks[send_block_index].second, MPI_FLOAT,right,
              1, MPI_COMM_WORLD, &requests[0]);
        // 由于不需要计算，因此直接覆盖buf上的数据即可
        MPI_Irecv(buffer + blocks[recv_block_index].first, blocks[recv_block_index].second, MPI_FLOAT,left,
              1, MPI_COMM_WORLD, &requests[1]);
        MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
        send_block_index = recv_block_index;
        recv_block_index = (recv_block_index - 1 + worker_number) % worker_number;
    }

    delete[] recv_buffer;
}
#endif // ASYC_ADMM_H
