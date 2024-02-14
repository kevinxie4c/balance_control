#ifndef IOUTIL_H
#define IOUTIL_H

#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Core>
#include <osg/Vec3>
#include <nlohmann/json.hpp>

template<typename T> std::vector<T> readListFrom(const std::string &filename);
Eigen::VectorXd readVectorXdFrom(const std::string &filename);
std::vector<Eigen::VectorXd> readVectorXdListFrom(const std::string &filename);
Eigen::MatrixXd readMatrixXFrom(const std::string &filename);
osg::Vec3 json2Vec3(const nlohmann::json &json);

template<typename T>
std::vector<T> readListFrom(const std::string &filename)
{
    std::ifstream input(filename);
    if (input.fail())
	throw std::ios_base::failure("cannot open " + filename);
    std::vector<T> list;
    T d;
    while (input >> d)
	list.push_back(d);
    input.close();
    return list;
}

#endif
