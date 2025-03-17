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
    //world->setGravity(Vector3d(0, 0, 0));
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
    //state = VectorXd(skeleton->getNumDofs() * 2);
    state = VectorXd(skeleton->getNumDofs() * 2 + 3);
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

    refMotion = readVectorXdListFrom(json["ref_motion"]);
    frameRate = json["frame_rate"].get<int>();

    reset();
}

SimpleEnv::~SimpleEnv()
{
}

void SimpleEnv::reset()
{
    world->reset();
    VectorXd zeros = VectorXd::Zero(skeleton->getNumDofs());
    skeleton->setPositions(zeros);
    skeleton->setVelocities(zeros);
    prev_com = skeleton->getRootBodyNode()->getCOM();
    done = false;
    updateState();
}

void SimpleEnv::step()
{
    prev_com = skeleton->getRootBodyNode()->getCOM();
    VectorXd force = action.array() * scales.array();
    force.head(3).setZero();
    VectorXd dq = skeleton->getVelocities();
    VectorXd ddq = skeleton->getAccelerations();
    VectorXd ds(state.size());
    ds << dq, ddq, 0, 0, 0;
    ds = (ds.array() / (normalizerStd.array() + 1e-8)).matrix();
    VectorXd df = ((policyJacobian * ds).array() * scales.array()).matrix();
    df.head(3).setZero();
    //df.setZero();
    double dt = 1.0 / forceRate;
    //force.setZero();
    //cout << force.transpose() << endl;
    done = false;
    for (size_t i = 0; i < forceRate / actionRate; ++i)
    {
        VectorXd f = force + df * dt * i;
        f_forces << f.transpose() << endl;
        //skeleton->setForces(force);
        skeleton->setForces(f);
        world->step();
    }
    updateState();
    //cout << skeleton->getPositions().transpose() << endl;
}

void SimpleEnv::updateState()
{
    vector<BodyNode*> bns = skeleton->getBodyNodes();
    BodyNode *lFoot = bns[3], *rFoot = bns[6], *spNode;
    if (lFoot->getCOM().y() < rFoot->getCOM().y())
        spNode = lFoot;
    else
        spNode = rFoot;
    Vector3d r_IP = skeleton->getCOM() - spNode->getCOM();
    VectorXd q = skeleton->getPositions();
    VectorXd dq = skeleton->getVelocities();
    state << q, dq, r_IP;
    Eigen::Vector3d curr_com = skeleton->getRootBodyNode()->getCOM();
    done = done || (curr_com.y() < 0.50 || curr_com.y() > 1.70);

    double r_com_vel, r_ref, r_norm;
    r_com_vel = 1 * exp(-5 * abs((curr_com.x() - prev_com.x()) * actionRate - 1.2));

    //double theta = sin(getTime() / 4 * M_PI) * M_PI / 6;
    //r_ref = 10 * (exp(-5 * abs(q[3] - theta)) + exp(-5 * abs(q[6] + theta)));
    VectorXd ref = refMotion[(size_t)round(getTime() * frameRate) % refMotion.size()];
    r_ref = 10 * exp(-2 * (q.tail(6) - ref).norm());

    r_norm = 0.5 * exp(-0.5 * action.norm());

    reward = r_com_vel + r_ref + r_norm;
    //cout << r_com_vel << " " << r_ref << " " << r_norm << endl;
    //cout << curr_com.transpose() << endl;

    /*
    if (isnan(reward) || state.array().isNaN().any())
    {
        //cout << state << endl;
        //cout << reward << endl;
        state.setOnes();
        reward = -100;
        done = true;
    }
    */
}
