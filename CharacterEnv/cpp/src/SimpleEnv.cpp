#include "SimCharacter.h"
#include "SimpleEnv.h"
#include "IOUtil.h"

using namespace std;
using namespace Eigen;
using namespace dart::dynamics;

SimpleEnv::SimpleEnv(const char *cfgFilename)
{
    ifstream input(cfgFilename);
    if (input.fail())
        throw ios_base::failure(string("cannot open ") + cfgFilename);
    nlohmann::json json;
    input >> json;
    input.close();

    SimCharacter character(json["character"]);
    skeleton = character.skeleton;
    world = dart::simulation::World::create();
    //skeleton->setGravity(Vector3d(0, -9.8, 0)); // Why it doesn't work?
    world->setGravity(Vector3d(0, -9.8, 0));
    world->addSkeleton(skeleton);
    world->setTimeStep(1.0 / forceRate);
    //world->getConstraintSolver()->setCollisionDetector(dart::collision::DARTCollisionDetector::create());
    //world->getConstraintSolver()->setCollisionDetector(dart::collision::FCLCollisionDetector::create());

    floor = Skeleton::create("floor");
    BodyNodePtr body = floor->createJointAndBodyNodePair<WeldJoint>(nullptr).second;
    // Deprecated
    //body->setFrictionCoeff(json["friction_coeff"].get<double>());
    //body->setRestitutionCoeff(json["restitution_coeff"].get<double>());
    body->setName("floor");
    double floor_width = 1e8;
    double floor_height = 1;
    shared_ptr<BoxShape> box(new BoxShape(Vector3d(floor_width, floor_height, floor_width)));
    auto shapeNode = body->createShapeNodeWith<CollisionAspect, DynamicsAspect>(box);
    shapeNode->getDynamicsAspect()->setFrictionCoeff(json["friction_coeff"].get<double>());
    shapeNode->getDynamicsAspect()->setRestitutionCoeff(json["restitution_coeff"].get<double>());
    Isometry3d tf(Isometry3d::Identity());
    tf.translation() = Vector3d(0.0, -0.501, 0.0);
    body->getParentJoint()->setTransformFromParentBodyNode(tf);
    world->addSkeleton(floor);

    scales = readVectorXdFrom(json["scales"]);
    state = VectorXd(skeleton->getNumDofs() * 2);
    action = VectorXd(skeleton->getNumDofs());

    skeleton->setPositionLowerLimits(readVectorXdFrom(json["lower_limits"]));
    skeleton->setPositionUpperLimits(readVectorXdFrom(json["upper_limits"]));

    VectorXd limits = 1000 * VectorXd::Ones(skeleton->getNumDofs());
    skeleton->setVelocityLowerLimits(-limits);
    skeleton->setVelocityUpperLimits(limits);

    for (Joint *joint: skeleton->getJoints())
    {
        joint->setActuatorType(Joint::FORCE);
        joint->setLimitEnforcement(true);
    }

    reset();
}

void SimpleEnv::reset()
{
    world->reset();
    VectorXd zeros = VectorXd::Zero(skeleton->getNumDofs());
    skeleton->setPositions(zeros);
    skeleton->setVelocities(zeros);
    updateState();
}

void SimpleEnv::step()
{
    VectorXd force = action.array() * scales.array();
    force.head(3).setZero();
    //force.setZero();
    //cout << force.transpose() << endl;
    for (size_t i = 0; i < forceRate / actionRate; ++i)
    {
        skeleton->setForces(force);
        world->step();
    }
    updateState();
    //cout << skeleton->getPositions().transpose() << endl;
}

void SimpleEnv::updateState()
{
    state << skeleton->getPositions(), skeleton->getVelocities();
    const BodyNode *root = skeleton->getRootBodyNode();
    Vector3d v = root->getCOMLinearVelocity();
    Vector3d v_bar;
    v_bar << 1, 0, 0;
    reward = exp(-(v - v_bar).norm());
    done = root->getCOM().y() < 0.5;
}
