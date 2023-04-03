#ifndef SIMCHARACTER_H
#define SIMCHARACTER_H

#include <string>
#include <nlohmann/json.hpp>
#include <dart/dart.hpp>


class SimCharacter
{
    public:
	SimCharacter(std::string filename);

	dart::dynamics::SkeletonPtr skeleton;

    private:
	void createJoint(nlohmann::json json, dart::dynamics::BodyNodePtr parent);
};

#endif
