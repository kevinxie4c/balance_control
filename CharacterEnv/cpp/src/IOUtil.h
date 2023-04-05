#ifndef IOUTIL_H
#define IOUTIL_H

#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Core>

std::vector<double> readListFrom(const std::string &filename);
Eigen::VectorXd readVectorXdFrom(const std::string &filename);
std::vector<Eigen::VectorXd> readVectorXdListFrom(const std::string &filename);
Eigen::MatrixXd readMatrixXFrom(const std::string &filename);

#endif
