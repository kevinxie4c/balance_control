#include <cstring>
#include "IOUtil.h"

Eigen::VectorXd readVectorXdFrom(const std::string &filename)
{
    std::vector<double> list = readListFrom<double>(filename);
    Eigen::VectorXd v(list.size());
    memcpy(v.data(), list.data(), list.size() * sizeof(double));
    return v;
}

std::vector<Eigen::VectorXd> readVectorXdListFrom(const std::string &filename)
{
    std::ifstream input(filename);
    if (input.fail())
	throw std::ios_base::failure("cannot open " + filename);
    std::vector<Eigen::VectorXd> result;
    std::string line;
    while (std::getline(input, line))
    {
	std::stringstream strStream(line);
	std::vector<double> list;
	double d;
	while (strStream >> d)
	    list.push_back(d);
	Eigen::VectorXd v(list.size());
	memcpy(v.data(), list.data(), list.size() * sizeof(double));
	result.push_back(v);
    }
    input.close();
    return result;
}

Eigen::MatrixXd readMatrixXFrom(const std::string &filename)
{
    Eigen::MatrixXd m;
    std::vector<Eigen::VectorXd> vlist = readVectorXdListFrom(filename);
    if (vlist.size() > 0 && vlist[0].size() > 0)
    {
	m = Eigen::MatrixXd(vlist.size(), vlist[0].size());
	for (size_t i = 0; i < vlist.size(); ++i)
	    m.row(i) = vlist[i];
    }
    return m;
}
