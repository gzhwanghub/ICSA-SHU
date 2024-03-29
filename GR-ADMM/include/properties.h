//
// Created by cluster on 2020/10/14.
//

#ifndef GR_ADMM_PROPERTIES_H
#define GR_ADMM_PROPERTIES_H

#include <map>
#include <string>

class Properties {
public:
    Properties(int &argc, char **&argv);

    std::string GetString(const std::string &property_name);

    int GetInt(const std::string &property_name);

    double GetDouble(const std::string &property_name);

    bool GetBool(const std::string &property_name);

    bool HasProperty(const std::string &property_name);

    void CheckProperty(const std::string &property_name);

    void Print();

private:
    std::map<std::string, std::string> properties_;

    void ParseFromFile(const std::string &path);
};


#endif //GR_ADMM_PROPERTIES_H
