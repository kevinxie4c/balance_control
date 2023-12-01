#include <string>
#include <stdexcept>
#include <dart/dart.hpp>
#include <nlohmann/json.hpp>
#include "CharacterEnv.h"
#include "MimicEnv.h"
#include "SimpleEnv.h"

using namespace std;
using namespace dart::dynamics;

CharacterEnv* CharacterEnv::makeEnv(const char *cfgFilename)
{
    ifstream input(cfgFilename);
    if (input.fail())
        throw ios_base::failure(string("cannot open ") + cfgFilename);
    nlohmann::json json;
    input >> json;
    input.close();
    if (json["env"] == "mimic")
        return new MimicEnv(cfgFilename);
    else if (json["env"] == "simple")
        return new SimpleEnv(cfgFilename);
    else
        throw runtime_error(string("unknow env: ") + json["env"].get<string>());
}

double CharacterEnv::getTime()
{
    return world->getTime();
}

double  CharacterEnv::getTimeStep()
{
    return world->getTimeStep();
}

/*
void CharacterEnv::setTimeStep(double h)
{
    world->setTimeStep(h);
}
*/

void CharacterEnv::print_info()
{
    cout << "BodyNode:" << endl;
    for (size_t i = 0; i < skeleton->getNumBodyNodes(); ++i)
    {
        const BodyNode *bn = skeleton->getBodyNode(i);
        cout << bn->getName() << endl;
    }
    cout << "Dofs:" << endl;
    for (const DegreeOfFreedom *dof: skeleton->getDofs())
        cout << dof->getName() << endl;
}
