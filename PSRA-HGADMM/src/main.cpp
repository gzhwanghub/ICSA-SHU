#include "../include/admm.h"
#include "../include/other/properties.h"

int main(int argc, char **argv) {
    Properties properties(argc, argv);
    //LOG(FATAL) << "ADMM Begin 1" ;
    int dimension = properties.GetInt("dimension");
    //int min_barrier = properties.GetInt("min_barrier");
    //int max_delay = properties.GetInt("max_delay");
    int max_iterations = properties.GetInt("max_iterations");
    //int interval = properties.GetInt("interval");

    int repeatIter = properties.GetInt("repeatIter");
    int DynamicGroup = properties.GetInt("DynamicGroup");
    int nodesOfGroup = properties.GetInt("nodesOfGroup");

    double rho = properties.GetDouble("rho");
    double l2reg = properties.GetDouble("l2reg");
    double ABSTOL = properties.GetDouble("ABSTOL");
    double RELTOL = properties.GetDouble("RELTOL");
    std::string train_data_path = properties.GetString("train_data_path");
    std::string test_data_path = properties.GetString("test_data_path");

    ADMM admm(dimension, max_iterations, rho, l2reg, ABSTOL, RELTOL, train_data_path,
              test_data_path,repeatIter,DynamicGroup,nodesOfGroup);
    if (admm.GetID() == 0) {
        properties.Print();
    }
    admm.Run();
    return 0;
}
