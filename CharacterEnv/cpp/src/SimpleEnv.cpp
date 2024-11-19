#include <dart/collision/bullet/BulletCollisionDetector.hpp>
#include <dart/collision/ode/OdeCollisionDetector.hpp>
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
    world->getConstraintSolver()->setCollisionDetector(dart::collision::BulletCollisionDetector::create());
    //world->getConstraintSolver()->setCollisionDetector(dart::collision::OdeCollisionDetector::create());
    //((dart::constraint::BoxedLcpConstraintSolver*)world->getConstraintSolver())->setBoxedLcpSolver(std::unique_ptr<dart::constraint::BoxedLcpSolver>(new dart::constraint::DantzigBoxedLcpSolver()));
    ((dart::constraint::BoxedLcpConstraintSolver*)world->getConstraintSolver())->setBoxedLcpSolver(std::unique_ptr<dart::constraint::BoxedLcpSolver>(new dart::constraint::PgsBoxedLcpSolver()));

    floor = Skeleton::create("floor");
    BodyNodePtr body = floor->createJointAndBodyNodePair<WeldJoint>(nullptr).second;
    // Deprecated
    //body->setFrictionCoeff(json["friction_coeff"].get<double>());
    //body->setRestitutionCoeff(json["restitution_coeff"].get<double>());
    body->setName("floor");
    double floor_width = 1e2;
    double floor_height = 1;
    shared_ptr<BoxShape> box(new BoxShape(Vector3d(floor_width, floor_height, floor_width)));
    //auto shapeNode = body->createShapeNodeWith<CollisionAspect, DynamicsAspect>(box);
    auto shapeNode = body->createShapeNodeWith<CollisionAspect, DynamicsAspect, VisualAspect>(box);
    shapeNode->getDynamicsAspect()->setFrictionCoeff(json["friction_coeff"].get<double>());
    shapeNode->getDynamicsAspect()->setRestitutionCoeff(json["restitution_coeff"].get<double>());
    shapeNode->getVisualAspect()->setColor(Eigen::Vector3d(1.0, 1.0, 1.0));
    Isometry3d tf(Isometry3d::Identity());
    tf.translation() = Vector3d(0.0, -0.502, 0.0);
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
    prev_com = skeleton->getRootBodyNode()->getCOM();
    updateState();
}

void SimpleEnv::step()
{
    prev_com = skeleton->getRootBodyNode()->getCOM();
    VectorXd force = (action.array().min(1).max(-1) * scales.array()).matrix();
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
    Eigen::Vector3d curr_com = skeleton->getRootBodyNode()->getCOM();
    done = curr_com.y() < 0.50 || curr_com.y() > 1.70;
    reward = 5 * (curr_com.x() - prev_com.x()) + exp(-action.norm()) + (done ? 0 : 1);
    //cout << curr_com.transpose() << endl;
}
