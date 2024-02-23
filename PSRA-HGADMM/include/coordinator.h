#ifndef SPARSEALLREDUCE_COORDINATOR_H
#define SPARSEALLREDUCE_COORDINATOR_H

#include <map>
#include <vector>

using namespace std;
namespace spar {

    class Coordinator {
    public:
        static Coordinator *GetInstance();

        void Set_max_iterations(int max_iterations) { max_iterations_ = max_iterations; };

        void Set_repeatIter(int repeat_iter){repeatIter = repeat_iter;};

        void Set_DynamicGroup(int Dynamic_Group) { DynamicGroup = Dynamic_Group; };

        void Set_nodesOfGroup(int nodes_Group) { nodesOfGroup = nodes_Group; };

        void Run();

    private:

        int id_;
        int worker_number_;
        int repeatIter;
        int max_iterations_;
        int DynamicGroup ;
        int nodesOfGroup;
        //所有Leader进程的id
        std::vector<int> leader_list;

        /*   std::map<int, int> worker_delay_;
           std::vector<int> ready_worker_list_;*/

        Coordinator();

        void CreateGroup();

        void Terminate();

        /*intercommunicate group*/
        vector<int> exchangeElement(vector<int> data, int GroupNum, int Group1, int Group2, int part);

        vector<vector<int>> divideGroup(vector<int> nodes, int groupNums);

        int position(double *vec, int size, int index);

        vector<int> findFastNodes(double *time, vector<int> group, int node, int numsofGrup, int size);

        void changeGroup(vector<vector<int>> &data, int node, vector<int> fastVec, int numsOfgroup, int iter);

        void MasterNodes();
    };

}

#endif //SPARSEALLREDUCE_COORDINATOR_H
