#include <string>
#include <stdexcept>
#include <dart/dart.hpp>
#include <nlohmann/json.hpp>
#include "IOUtil.h"
#include "CharacterEnv.h"
#include "MimicEnv.h"
#include "SimpleEnv.h"

using namespace std;
using namespace dart::dynamics;

CharacterEnv* CharacterEnv::makeEnv(const char *cfgFilename)
{
    CharacterEnv *env = nullptr;
    ifstream input(cfgFilename);
    if (input.fail())
        throw ios_base::failure(string("cannot open ") + cfgFilename);
    nlohmann::json json;
    input >> json;
    input.close();
    if (json["env"] == "mimic")
        env = new MimicEnv(cfgFilename);
    else if (json["env"] == "simple")
        env = new SimpleEnv(cfgFilename);
    else
        throw runtime_error(string("unknow env: ") + json["env"].get<string>());

    env->eye = osg::Vec3(0, 1, 5);
    env->center = osg::Vec3(0, 1, 0);
    env->up = osg::Vec3(0, 1, 0);
    if (json.contains("camera"))
    {
        nlohmann::json camera = json["camera"];
        if (camera.contains("eye"))
            env->eye = json2Vec3(camera["eye"]);
        if (camera.contains("center"))
            env->center = json2Vec3(camera["center"]);
        if (camera.contains("up"))
            env->up = json2Vec3(camera["up"]);
    }

    return env;
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

void CharacterEnv::create_viewer()
{
    worldNode = new dart::gui::osg::RealTimeWorldNode(world);
    viewer = new dart::gui::osg::ImGuiViewer(osg::Vec4(0.0, 0.5, 1.0, 1.0));
    viewer->addWorldNode(worldNode.get());
    viewer->getCameraManipulator()->setHomePosition(eye, center, up);
    viewer->switchHeadlights(false);
    viewer->setUpwardsDirection(up);
}

void CharacterEnv::run_viewer()
{
    if (viewer == nullptr)
        create_viewer();
    viewer->run();
}

void CharacterEnv::render_viewer()
{
    if (viewer == nullptr)
        create_viewer();
    viewer->frame();
}

bool CharacterEnv::viewer_done()
{
    if (viewer == nullptr)
        create_viewer();
    return viewer->done();
}
