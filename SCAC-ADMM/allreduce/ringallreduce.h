//
// Created by cluster on 2020/10/14.
//

#ifndef GR_ADMM_RINGALLREDUCE_H
#define GR_ADMM_RINGALLREDUCE_H

#include <mpi.h>
#include <vector>
#include <map>
#include <math.h>
#include "reduceoperator.h"
#include "p2p_communicator.h"
#include "common.h"

#define random(x)(rand()%x)
using namespace spar;

template<class O, class T>

void RingAllreduce(T *buffer, int count, int id, std::vector<int> &worker_list, MPI_Comm comm, int QuantifyPart) {
    //worker_list包含了参与此次Allreduce的所有进程id
    int worker_number = worker_list.size();
    double extra_time = 0.0;
    double QBegin = 0, Q_end = 0;
    if (worker_number == 1) return;

    //对数组进行分块，在blocks中记录每一块的起始地址和块大小
    int block_size = count / worker_number;
    std::map<int, std::pair<int, int>> blocks;
    for (int i = 0; i < worker_number; ++i) {
        blocks[i].first = i * block_size;
        blocks[i].second = block_size;
    }
    //由于数组不一定能均分，因此最后一块的大小需要计算
    blocks[worker_number - 1].second = (count - (worker_number - 1) * block_size);
    //确定本进程在worker_list中的索引，并确定左右两个进程的id
    int left = -1, right = -1, my_index = -1;
    for (int i = 0; i < worker_number; ++i) {
        if (worker_list[i] == id) {
            my_index = i;
            left = worker_list[(i - 1 + worker_number) % worker_number];
            right = worker_list[(i + 1) % worker_number];
            break;
        }
    }
    //CHECK(left != -1 && right != -1 && my_index != -1);
    T *recv_buffer = new T[count];
    //每个进程从发送my_index这个块开始，因为每个进程的索引肯定不同，因此发送的是不同的块
    //每个进程是从左边进程接收数据，左边进程发送的是my_index-1这个块
    int send_block_index = my_index;
    int recv_block_index = (send_block_index - 1 + worker_number) % worker_number;
    //一共需要worker_number-1次迭代，从左边进程接收数据，发送数据给右边进程
    for (int i = 0; i < worker_number - 1; ++i) {
        if (QuantifyPart == 0 || QuantifyPart == 2) {
            MPI_Request requests[2];
            Isend(buffer + blocks[send_block_index].first, blocks[send_block_index].second, right,
                  MessageType::kScatterReduce, comm, &requests[0]);
            Irecv(recv_buffer, blocks[recv_block_index].second, left, MessageType::kScatterReduce, comm,
                  &requests[1]);
            MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);//必须等到数据传输完成后才能reduce操作
        } else if (QuantifyPart == 1 || QuantifyPart == 3) {

            MPI_Request requests[2];//
            double max_min[2];
            uint16_t *sendMsg, *recvMsg;
            sendMsg = new uint16_t[blocks[send_block_index].second + 8];
            recvMsg = new uint16_t[blocks[recv_block_index].second + 8];
            max_min[0] = buffer[blocks[send_block_index].first];
            max_min[1] = buffer[blocks[send_block_index].first];
            //cout<<buffer[blocks[send_block_index].first]<<" "<<blocks[send_block_index].first<<endl;
            //最大和最小值
            for (size_t u = blocks[send_block_index].first;
                 u < blocks[send_block_index].first + blocks[send_block_index].second; u++) {
                max_min[0] = max_min[0] > buffer[u] ? max_min[0] : buffer[u];
                max_min[1] = max_min[1] < buffer[u] ? max_min[1] : buffer[u];
            }
            //        cout<<blocks[send_block_index].first<<" "<<blocks[send_block_index].first+blocks[send_block_index].second<<endl;
            //量化
            double unit = (max_min[0] - max_min[1]) / (pow(2, 16) - 1);
            //cout<<unit<<" ";
            int temp = 0;
            for (size_t u = blocks[send_block_index].first;
                 u < blocks[send_block_index].first + blocks[send_block_index].second; u++) {
                if (unit == 0) {
                    sendMsg[temp] = 0;
                } else {
                    srand((int) time(0));
                    sendMsg[temp] = (uint16_t)((buffer[u] - max_min[1]) / unit + random(1000001) / 1000000);
                }
                //cout<<buffer[u]<<" "<<max_min[1]<<sendMsg[temp]<<" "<<endl;
                temp += 1;
            }
            //将 max_min 和sendMSg拼起来
            memcpy(sendMsg + blocks[send_block_index].second, max_min, (sizeof(double)) * 2);
//            Isend(max_min, 2, right,
//                  MessageType::kmaxandmin, comm, &requests[0]);
            Isend(sendMsg, blocks[send_block_index].second + 8, right,
                  MessageType::kScatterReduce, comm, &requests[0]);
//            Irecv(max_min, 2, left, MessageType::kmaxandmin, comm,
//                  &requests[2]);
            Irecv(recvMsg, blocks[recv_block_index].second + 8, left, MessageType::kScatterReduce, comm,
                  &requests[1]);
            MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);//必须等到数据传输完成后才能reduce操作

            memcpy(max_min, recvMsg + blocks[recv_block_index].second, (sizeof(double)) * 2);
            //解量化
            double util = (max_min[0] - max_min[1]) / (pow(2, 16) - 1);
            for (int j = 0; j < blocks[recv_block_index].second; ++j) {
                if (util == 0)
                    recv_buffer[j] = 0;
                else
                    recv_buffer[j] = recvMsg[j] * util + max_min[1];
            }
            delete [] recvMsg;
            delete [] sendMsg;

        }

        T *base = buffer + blocks[recv_block_index].first;
        block_size = blocks[recv_block_index].second;
        //接收到块后要Reduce到本进程对应位置的块上
        for (int j = 0; j < block_size; ++j) {
            Reduce<O>(base[j], recv_buffer[j]);
        }
        //下一次迭代发送本次迭代接收到的块
        send_block_index = recv_block_index;
        recv_block_index = (recv_block_index - 1 + worker_number) % worker_number;

//        delete sendMsg;
//        delete recvMsg;
    }

    //前面半个步骤使得每个节点存有一部分数据
    //下面使所有节点都有完整的数据
    for (int i = 0; i < worker_number - 1; ++i) {
        if (QuantifyPart == 0 || QuantifyPart == 1) {
            MPI_Request requests[2];
            Isend(buffer + blocks[send_block_index].first, blocks[send_block_index].second, right,
                  MessageType::kAllGather, comm, &requests[0]);
            // 由于不需要计算，因此直接覆盖buf上的数据即可
            Irecv(buffer + blocks[recv_block_index].first, blocks[recv_block_index].second, left,
                  MessageType::kAllGather, comm, &requests[1]);
            MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
        } else if (QuantifyPart == 2 || QuantifyPart == 3) {
            MPI_Request requests[2];//由2变成4 因为要传输max和min
            double max_min[2];
            uint16_t *sendMsg, *recvMsg;
//            delete [] sendMsg;delete []recvMsg;
            sendMsg = new uint16_t[blocks[send_block_index].second + 8];
            recvMsg = new uint16_t[blocks[recv_block_index].second + 8];
            max_min[0] = buffer[blocks[send_block_index].first];
            max_min[1] = buffer[blocks[send_block_index].first];
//        //cout<<buffer[blocks[send_block_index].first]<<" "<<blocks[send_block_index].first<<endl;
//        //最大和最小值
            for (size_t u = blocks[send_block_index].first;
                 u < blocks[send_block_index].first + blocks[send_block_index].second; u++) {
                max_min[0] = max_min[0] > buffer[u] ? max_min[0] : buffer[u];
                max_min[1] = max_min[1] < buffer[u] ? max_min[1] : buffer[u];
            }
            //cout<<max_min[0]<<"   "<<max_min[1]<<endl;
////        cout<<blocks[send_block_index].first<<" "<<blocks[send_block_index].first+blocks[send_block_index].second<<endl;
//
            //量化
            double unit = (max_min[0] - max_min[1]) / (pow(2, 16) - 1);
            //cout<<unit<<" ";
            int temp = 0;
            for (size_t u = blocks[send_block_index].first;
                 u < blocks[send_block_index].first + blocks[send_block_index].second; u++) {
                if (unit == 0) {
                    sendMsg[temp] = 0;
                } else {
                    srand((int) time(0));
                    sendMsg[temp] = (uint16_t)((buffer[u] - max_min[1]) / unit + random(1000001) / 1000000);
                }
                //cout<<buffer[u]<<" "<<max_min[1]<<sendMsg[temp]<<" "<<endl;
                temp += 1;
            }

            //将 max_min 和sendMSg拼起来
            memcpy(sendMsg + blocks[send_block_index].second, max_min, (sizeof(double)) * 2);

//            Isend(max_min, 2, right,
//                  MessageType::kmaxandmin, comm, &requests[0]);
            Isend(sendMsg, blocks[send_block_index].second + 8, right,
                  MessageType::kAllGather, comm, &requests[0]);
//            Irecv(max_min, 2, left, MessageType::kmaxandmin, comm,
//                  &requests[2]);
            Irecv(recvMsg, blocks[recv_block_index].second + 8, left, MessageType::kAllGather, comm,
                  &requests[1]);
            MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);//必须等到数据传输完成后才能reduce操作

            memcpy(max_min, recvMsg + blocks[send_block_index].second, (sizeof(double)) * 2);
            //解量化
            double util = (max_min[0] - max_min[1]) / (pow(2, 16) - 1);
            T *base = buffer + blocks[recv_block_index].first;
            block_size = blocks[recv_block_index].second;
            for (int j = 0; j < block_size; ++j) {
                if (util == 0)
                    base[j] = 0;
                else
                    base[j] = recvMsg[j] * util + max_min[1];
                //cout<<recvMsg[j]<<" ";
            }
            delete [] recvMsg;
            delete [] sendMsg;
        }

        send_block_index = recv_block_index;
        recv_block_index = (recv_block_index - 1 + worker_number) % worker_number;
    }

    delete[] recv_buffer;
}


//稀疏版本的算法和稠密版本的步骤基本一致，只是在发送前判断是否满足稀疏传输条件
template<class O, class T>
void SparseRingAllreduce(T *buffer, int count, int id, std::vector<int> &worker_list, MPI_Comm comm) {
    int worker_number = worker_list.size();
    if (worker_number == 1) return;
//    if (count < 512 * worker_number) {
//        SimpleAllreduce<O>(buffer, count, id, worker_list, comm);
//        return;
//    }

    int block_size = count / worker_number;
    std::map<int, std::pair<int, int>> blocks;
    for (int i = 0; i < worker_number; ++i) {
        blocks[i].first = i * block_size;
        blocks[i].second = block_size;
    }
    blocks[worker_number - 1].second = (count - (worker_number - 1) * block_size);
    int left = -1, right = -1, my_index = -1;
    for (int i = 0; i < worker_number; ++i) {
        if (worker_list[i] == id) {
            my_index = i;
            left = worker_list[(i - 1 + worker_number) % worker_number];
            right = worker_list[(i + 1) % worker_number];
            break;
        }
    }
//    CHECK(left != -1 && right != -1 && my_index != -1);

    bool need_check = true;
    int isize = sizeof(int);
    int vsize = sizeof(T);
    int *index_buffer = new int[count];
    T *value_buffer = new T[count];
    T *recv_buffer = new T[count];
    int send_block_index = my_index;
    int recv_block_index = (send_block_index - 1 + worker_number) % worker_number;
    for (int i = 0; i < worker_number - 1; ++i) {
        MPI_Status statuses[2];
        MPI_Request requests[2];
        T *base = buffer + blocks[send_block_index].first;
        block_size = blocks[send_block_index].second;
        //如果上一次迭代接收到了稠密的块，那么本次传输就不用检查了，必然稠密
        if (need_check) {
            int nnz = 0;
            for (int j = 0; j < block_size; ++j) {
                if (base[j] != 0) {
                    ++nnz;
                }
            }
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
                MPI_Isend(value_buffer, nnz * (isize + vsize), MPI_CHAR, right,
                          MessageType::kScatterReduce, comm, &requests[0]);
            } else {
                MPI_Isend(base, block_size * vsize, MPI_CHAR, right, MessageType::kScatterReduce, comm,
                          &requests[0]);
            }
        } else {
            MPI_Isend(base, block_size * vsize, MPI_CHAR, right, MessageType::kScatterReduce, comm,
                      &requests[0]);
        }
        base = buffer + blocks[recv_block_index].first;
        block_size = blocks[recv_block_index].second;
        MPI_Irecv(recv_buffer, block_size * vsize, MPI_CHAR, left, MessageType::kScatterReduce, comm, &requests[1]);
        MPI_Waitall(2, requests, statuses);
        int nnz = 0;
        MPI_Get_count(&statuses[1], MPI_CHAR, &nnz);
        //如果接收到的字节数少于这个块应有的大小，说明左边进程发送的是稀疏数据
        if (nnz < block_size * vsize) {
//            CHECK_EQ(nnz % (isize + vsize), 0);
            nnz = nnz / (isize + vsize);
            //分离值和索引
            memcpy(index_buffer, recv_buffer + nnz, nnz * isize);
            for (int j = 0; j < nnz; ++j) {
                Reduce<O>(base[index_buffer[j]], recv_buffer[j]);
            }
            need_check = true;
        } else {
//            CHECK_EQ(nnz, block_size * vsize);
            for (int j = 0; j < block_size; ++j) {
                Reduce<O>(base[j], recv_buffer[j]);
            }
            need_check = false;
        }

        // 下一次循环发送这次循环接收到的segment
        send_block_index = recv_block_index;
        recv_block_index = (recv_block_index - 1 + worker_number) % worker_number;
    }

    for (int i = 0; i < worker_number - 1; ++i) {
        MPI_Status statuses[2];
        MPI_Request requests[2];
        T *base = buffer + blocks[send_block_index].first;
        block_size = blocks[send_block_index].second;
        if (need_check) {
            int nnz = 0;
            for (int j = 0; j < block_size; ++j) {
                if (base[j] != 0) {
                    ++nnz;
                }
            }
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
                          MessageType::kAllGather, comm, &requests[0]);
            } else {
                MPI_Isend(base, block_size * vsize, MPI_CHAR, right, MessageType::kAllGather, comm,
                          &requests[0]);
            }
        } else {
            MPI_Isend(base, block_size * vsize, MPI_CHAR, right, MessageType::kAllGather, comm,
                      &requests[0]);
        }
        base = buffer + blocks[recv_block_index].first;
        block_size = blocks[recv_block_index].second;
        MPI_Irecv(recv_buffer, block_size * vsize, MPI_CHAR, left, MessageType::kAllGather, comm, &requests[1]);
        MPI_Waitall(2, requests, statuses);
        int nnz = 0;
        MPI_Get_count(&statuses[1], MPI_CHAR, &nnz);
        if (nnz < block_size * vsize) {
//            CHECK_EQ(nnz % (isize + vsize), 0);
            nnz = nnz / (isize + vsize);
            memcpy(index_buffer, recv_buffer + nnz, nnz * isize);
            for (int j = 0; j < block_size; ++j) {
                base[j] = 0;
            }
            for (int j = 0; j < nnz; ++j) {
                base[index_buffer[j]] = recv_buffer[j];
            }
            need_check = true;
        } else {
//            CHECK_EQ(nnz, block_size * vsize);
            for (int j = 0; j < block_size; ++j) {
                base[j] = recv_buffer[j];
            }
            need_check = false;
        }

        send_block_index = recv_block_index;
        recv_block_index = (recv_block_index - 1 + worker_number) % worker_number;
    }

    delete[] recv_buffer;
    delete[] value_buffer;
    delete[] index_buffer;
}

#endif //GR_ADMM_RINGALLREDUCE_H
