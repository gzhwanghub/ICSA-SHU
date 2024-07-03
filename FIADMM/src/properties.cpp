/*************************************************************************
    > File Name: properties.cpp
    > Description: Parameter parsing tool
    > Author: Jinyang Xie
    > Created Time: 2020-10-14
 ************************************************************************/
#include <fstream>
#include "../include/properties.h"
#include "../include/string_util.h"
#include "../include/type_convert.h"

Properties::Properties(int &argc, char **&argv) {
    // The first parameter is the program name, so the loop starts with i=1.
    // If a command line parameter starts with '-',then we think it is a valid parameter.
    // Therefore, the format of valid parameters is -key1 value1 -key2 value2 other.
    // Stop parsing when encountering a command line parameter that does not start with '-'
    int i = 1;
    for (; i < argc && argv[i][0] == '-';) {
        std::string key(argv[i] + 1);
        // If there is -file path in the command line parameter, the configuration file will be read from the path.
        // Therefore, file is a reserved parameter name and cannot be used by users.
        if (key == "file") {
            ParseFromFile(argv[i + 1]);
        } else {
            properties_[key] = argv[i + 1];
        }
        i += 2;
    }
    // Move the remaining command line parameters and modify argc.
    int j = 1, k = i;
    for (; k > 1 && k < argc;) {
        argv[j++] = argv[k++];
    }
    argc -= (i - 1);
}

void Properties::ParseFromFile(const std::string &path) {
    std::ifstream reader(path);
    if (reader.fail()) {
        //LOG(FATAL) << "Unable to open configuration file, file name:" << path;
    }

    // Create a new map to temporarily store attribute values.
    std::map<std::string, std::string> temp;
    std::string line;
    while (std::getline(reader, line)) {
        // The contents after the # sign in each line are comments, so these contents are deleted.
        std::size_t pos = line.find_first_of('#');
        if (pos != std::string::npos) {
            line.erase(pos);
        }
        // Remove the spaces before and after each line.
        Trim(line);
        if (line.empty()) {
            continue;
        }
        // The format of each line is key:value, and there can be spaces on both sides of the colon.
        pos = line.find_first_of(':');
        if (pos == std::string::npos || pos == 0 || pos == line.length() - 1) {
            //LOG(FATAL) << "Wrong format, it should be in key:value format." << line;
        }
        std::string key = line.substr(0, pos);
        std::string value = line.substr(pos + 1);
        Trim(key);
        Trim(value);
        temp[key] = value;
    }
    reader.close();
    // Command-line parameters take precedence over configuration file parameters, so only those parameters that are not in properties are copied.
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
    //LOG(FATAL) << property_name << " must be true or false." << std::endl;
    return false;
}

bool Properties::HasProperty(const std::string &property_name) {
    return properties_.count(property_name) != 0;
}

void Properties::CheckProperty(const std::string &property_name) {
    if (!HasProperty(property_name)) {
        //LOG(FATAL) << "Missing parameter " << property_name << std::endl;
    }
}

void Properties::Print() {
    //LOG(INFO) << "**************************************";
    for (auto it = properties_.begin(); it != properties_.end(); ++it) {
        //LOG(INFO) << it->first << ":" << it->second;
    }
    //LOG(INFO) << "**************************************";
}