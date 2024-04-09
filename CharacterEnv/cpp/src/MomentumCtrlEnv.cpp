#include <cmath>
#include <dart/collision/bullet/BulletCollisionDetector.hpp>
#include <dart/collision/ode/OdeCollisionDetector.hpp>
#include "SimCharacter.h"
#include "MomentumCtrlEnv.h"
#include "IOUtil.h"

using namespace std;
using namespace Eigen;
using namespace dart::dynamics;

MomentumCtrlEnv::MomentumCtrlEnv(const char *cfgFilename)
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
    //world->getConstraintSolver()->setCollisionDetector(dart::collision::BulletCollisionDetector::create());
    world->getConstraintSolver()->setCollisionDetector(dart::collision::OdeCollisionDetector::create());
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

    kp = readVectorXdFrom(json["kp"]);
    kd = readVectorXdFrom(json["kd"]);
    mkp = MatrixXd::Zero(kp.size(), kp.size());
    mkd = MatrixXd::Zero(kd.size(), kd.size());
    mkp.diagonal() = kp;
    mkd.diagonal() = kd;

    scales = readVectorXdFrom(json["scales"]);
    //state = VectorXd(skeleton->getNumDofs() * 2);
    state = VectorXd(5); // theta1, cos(theta2), sin(theta2), dtheta1, dtheta2
    action = VectorXd(skeleton->getNumDofs());

    reset();
}

void MomentumCtrlEnv::reset()
{
    world->reset();
    VectorXd rands = VectorXd::Random(skeleton->getNumDofs());
    VectorXd zeros = VectorXd::Zero(skeleton->getNumDofs());
    skeleton->setPositions(zeros);
    skeleton->setVelocities(zeros);
    updateState();
}

void MomentumCtrlEnv::step()
{
    //VectorXd ref = (action.array() * scales.array()).matrix();
    VectorXd ref = (skeleton->getPositions() + (action.array() * scales.array()).matrix());
    vector<BodyNode*> bns = skeleton->getBodyNodes();
    fallen = false;
    Vector3d L_prev = bns[2]->getAngularMomentum(); // angular momentum of the dist wrt (0, 0, 0)
    double t_prev = world->getTime();
    for (size_t i = 0; i < forceRate / actionRate; ++i)
    {
        // stable PD
        VectorXd q = skeleton->getPositions();
        VectorXd dq = skeleton->getVelocities();
        MatrixXd invM = (skeleton->getMassMatrix() + mkd * skeleton->getTimeStep()).inverse();
        VectorXd p = -mkp * skeleton->getPositionDifferences(q + dq * skeleton->getTimeStep(), ref);
        VectorXd d = -mkd * dq;
        VectorXd qddot = invM * (-skeleton->getCoriolisAndGravityForces() + p + d + skeleton->getConstraintForces());
        VectorXd force = p + d - mkd * qddot * world->getTimeStep();

        // PD
        /*
        VectorXd q = skeleton->getPositions();
        VectorXd dq = skeleton->getVelocities();
        VectorXd p = -mkp * skeleton->getPositionDifferences(q, ref);
        VectorXd d = -mkd * dq;
        VectorXd force = p + d;
        */

        //double rem = fmod(world->getTime(), 10.0);
        //if (rem >= 4.0 && rem < 4.2)
        //    skeleton->getBodyNode(2)->addExtForce(Vector3d(-500, 0, 0));
        //else if (rem >= 9.0 && rem < 9.2)
        //    skeleton->getBodyNode(2)->addExtForce(Vector3d(500, 0, 0));
        skeleton->setForces(force);
        world->step();
        dart::collision::CollisionResult result = world->getLastCollisionResult();
        for (int j = 1; j < bns.size(); ++j)
            if (result.inCollision(bns[j]))
            {
                fallen = true;
                break;
            }
    }
    updateState();
}

void MomentumCtrlEnv::updateState()
{
    //state << skeleton->getPositions(), skeleton->getVelocities();
    VectorXd q = skeleton->getPositions();
    state << q[0], cos(q[1]), cos(q[1]), skeleton->getVelocities();
    Vector3d c_r = skeleton->getRootBodyNode()->getCOM();
    Vector3d com = skeleton->getCOM();
    //done = abs(c_r.x()) > 0.1 || c_r.y() > 0.2 || fallen;
    done = fallen;
    vector<BodyNode*> bns = skeleton->getBodyNodes();
    //Vector3d com = skeleton->getCOM();
    //Vector3d v_com = skeleton->getCOMLinearVelocity();
    //Vector3d L = Vector3d::Zero(); // angular momentum wrt com
    //for (size_t i = 2; i < bns.size(); ++i)
    //{
    //    Vector3d com_i = bns->getCOM();
    //    Vector3d v_com_i = bns->getCOMLinearVelocity();
    //    Matrix3d I = bns->getInertia();
    //    Matrix3d R = bns->getWorldTransform().linear();
    //    Matrix3d omega = bns->getAngularVelocity();
    //    L += (com_i - com).cross(v_com_i - v_com) + R * I * R.transpose() * omega;
    //}
    Vector3d L = bns[2]->getAngularMomentum(); // angular momentum of the dist wrt (0, 0, 0)
    double t = world->getTime();
    if (t == t_prev)
        t = t_prev + 1;
    Vector3d dL = (L - L_prev) / (t - t_prev);
    Vector3d g = world->getGravity();
    double m = skeleton->getMass();
    //Vector3d com = skeleton->getCOM();
    Vector3d tau = com.cross(m * g);
    //cout << "com: " << com.transpose() << endl;
    //cout << "mg: " << (m * g).transpose() << endl;
    //cout << exp(-10 * abs(c_r.x())) << " " << exp(-10 * abs(com.x() - c_r.x())) << " " << exp(-action.norm()) << endl;
    //reward = exp(-10 * abs(c_r.x())) + exp(-10 * abs(com.x() - c_r.x())) + exp(-action.norm());
    //cout << exp(-skeleton->getPositions().norm()) << endl;
    //reward = exp(-skeleton->getPositions().norm()) + exp(-action.norm());
    //reward = exp(-(dL - tau).norm()) + exp(-action.norm());
    reward = exp(-1 * (dL - tau).norm());
    //cout << "dL: " << dL.z() << endl;
    //cout << "tau: " << tau.z() << endl;
    //cout << exp(-(dL - tau).norm()) << endl;
}

