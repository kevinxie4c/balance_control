#ifndef STD_NORMAL_DIST_VEC_H
#define STD_NORMAL_DIST_VEC_H

#include <random>
#include <vector>
#include <Eigen/Core>

// class for generating n-D standard normal distribution

class StdNormalDistVec
{
    public:
        Eigen::VectorXd v;
        std::default_random_engine generator;
        std::normal_distribution<double> distribution;

        StdNormalDistVec(size_t n);
        Eigen::VectorXd operator()();
        static double logProbability(Eigen::VectorXd x);

};

#endif
