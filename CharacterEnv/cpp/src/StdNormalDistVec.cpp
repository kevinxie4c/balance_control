#include <cmath>
#include <chrono>
#include "StdNormalDistVec.h"

using namespace std;
using namespace Eigen;

StdNormalDistVec::StdNormalDistVec(size_t n): v(VectorXd(n)), generator(chrono::system_clock::now().time_since_epoch().count()), distribution(0.0, 1.0) {}

VectorXd StdNormalDistVec::operator()()
{
    for (size_t i = 0; i < v.size(); ++i)
        v[i] = distribution(generator);
    return v;
}

double StdNormalDistVec::logProbability(VectorXd x)
{
    return -(x.array().square().sum() / 2) - x.size() * 0.5 * log(2 * M_PI);
}
