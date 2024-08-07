/*************************************************************************
    > File Name: string_util.cpp
    > Description: Cut string tool
    > Author: Guozheng Wang
    > Mail: gzh.wang@outlook.com
    > Created Time: 2020-10-14
 ************************************************************************/

#include "include/string_util.h"

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