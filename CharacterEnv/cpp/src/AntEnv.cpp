#include <dart/collision/bullet/BulletCollisionDetector.hpp>
#include <dart/collision/ode/OdeCollisionDetector.hpp>
#include "SimCharacter.h"
#include "AntEnv.h"
#include "IOUtil.h"

#define SIM_MODE 0

using namespace std;
using namespace Eigen;
using namespace dart::dynamics;

AntEnv::AntEnv(const char *cfgFilename)
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
    double floor_width = 1e4;
    double floor_height = 1;
    shared_ptr<BoxShape> box(new BoxShape(Vector3d(floor_width, floor_height, floor_width)));
    //auto shapeNode = body->createShapeNodeWith<CollisionAspect, DynamicsAspect>(box);
    auto shapeNode = body->createShapeNodeWith<CollisionAspect, DynamicsAspect, VisualAspect>(box);
    shapeNode->getDynamicsAspect()->setFrictionCoeff(json["friction_coeff"].get<double>());
    shapeNode->getDynamicsAspect()->setRestitutionCoeff(json["restitution_coeff"].get<double>());
    shapeNode->getVisualAspect()->setColor(Eigen::Vector3d(1.0, 1.0, 1.0));
    Isometry3d tf(Isometry3d::Identity());
    tf.translation() = Vector3d(0.0, -0.500, 0.0);
    body->getParentJoint()->setTransformFromParentBodyNode(tf);
    world->addSkeleton(floor);

    scales = readVectorXdFrom(json["scales"]);
    state = VectorXd(skeleton->getNumDofs() * 2);
    action = VectorXd(skeleton->getNumDofs() - 6);

    skeleton->setPositionLowerLimits(readVectorXdFrom(json["lower_limits"]));
    skeleton->setPositionUpperLimits(readVectorXdFrom(json["upper_limits"]));
    double cfm = dart::constraint::JointLimitConstraint::getConstraintForceMixing();
    double erp = dart::constraint::JointLimitConstraint::getErrorReductionParameter();
    cfm = 0.001;
    erp = 0.2;
    dart::constraint::JointLimitConstraint::setConstraintForceMixing(cfm);
    dart::constraint::JointLimitConstraint::setErrorReductionParameter(erp);
    dart::constraint::ContactConstraint::setConstraintForceMixing(cfm);
    dart::constraint::ContactConstraint::setErrorReductionParameter(erp);
    cout << "CFM: " << cfm << endl;
    cout << "ERP: " << erp << endl;

    //VectorXd limits = 1000 * VectorXd::Ones(skeleton->getNumDofs());
    //skeleton->setVelocityLowerLimits(-limits);
    //skeleton->setVelocityUpperLimits(limits);

    for (Joint *joint: skeleton->getJoints())
    {
        joint->setActuatorType(Joint::FORCE);
        joint->setLimitEnforcement(true);
    }

    skeleton->setSelfCollisionCheck(true);
    skeleton->setAdjacentBodyCheck(false);

    reset();
}

void AntEnv::reset()
{
    world->reset();
    VectorXd zeros = VectorXd::Zero(skeleton->getNumDofs());
    VectorXd initPos(skeleton->getNumDofs());
    initPos << 0, 0, 0, 0, -0.25, 0,
            0, -0.0, 0.0,
            0, 0.0, -0.0,
            0, -0.0, 0.0,
            0, 0.0, -0.0;
    //skeleton->setPositions(zeros);
    skeleton->setPositions(initPos);
    skeleton->setVelocities(zeros);
    prev_com = skeleton->getRootBodyNode()->getCOM();
    updateState();
}

void AntEnv::step()
{
    fallen = false;
    prev_com = skeleton->getRootBodyNode()->getCOM();
    VectorXd force = VectorXd::Zero(skeleton->getNumDofs());
    force.tail(action.size()) = action.array() * scales.array();
    //force.setZero();
    //cout << force.transpose() << endl;
    double h = 1.0 / forceRate;

#if SIM_MODE == 1
    VectorXd q = skeleton->getPositions();
    VectorXd dq = skeleton->getVelocities();
    MatrixXd aS = MatrixXd::Zero(action.size(), action.size());
    aS.diagonal() = scales;
    MatrixXd sS = MatrixXd::Zero(state.size(), state.size());
    sS.diagonal() = (normalizerStd.array() + 1e-8).matrix().cwiseInverse();
    MatrixXd dsdq = MatrixXd::Zero(36, 18);
    for (size_t i = 0; i < 18; ++i)
        dsdq(i, i) = 1;
    MatrixXd J = MatrixXd::Zero(18, 18);
    J.bottomRows(12) = aS * policyJacobian * sS * dsdq;
#endif

    for (size_t i = 0; i < forceRate / actionRate; ++i)
    {
#if SIM_MODE == 0
        VectorXd q = skeleton->getPositions();
        VectorXd dq = skeleton->getVelocities();
        VectorXd ddq = skeleton->getAccelerations();
        MatrixXd M = skeleton->getMassMatrix();
        VectorXd C = skeleton->getCoriolisAndGravityForces();
        //cout << "M\n" << M << endl;
        //cout << "C\n" << C << endl;
        MatrixXd aS = MatrixXd::Zero(action.size(), action.size());
        aS.diagonal() = scales;
        MatrixXd sS = MatrixXd::Zero(state.size(), state.size());
        sS.diagonal() = (normalizerStd.array() + 1e-8).matrix().cwiseInverse();
        //cout << "sS\n" << sS << endl;
        MatrixXd dsdq = MatrixXd::Zero(36, 18);
        for (size_t i = 0; i < 18; ++i)
            dsdq(i, i) = 1;
        MatrixXd J = MatrixXd::Zero(18, 18);
        J.bottomRows(12) = aS * policyJacobian * sS * dsdq;
        //cout << "pJ\n" << policyJacobian << endl;
        //cout << "J\n" << J << endl;
        //MatrixXd K(2, 2);
        //MatrixXd D(2, 2);
        //K << 500, 0,
        //     0, 500;
        //D << 0.0, 0.0,
        //     0.0, 0.0;
        //force = -K * q - D * dq;
        //MatrixXd A = (M + h * h * K);
        MatrixXd A = (M - h * h * J);
        VectorXd b = M * dq + h * (force - C);
        //MatrixXd A = M;
        //VectorXd b = (M - h * h * J) * dq + h * (force - C);
        //cout << "A\n" << A << endl;
        //cout << "b\n" << b << endl;
        //VectorXd dq_n = A.inverse() * b;
        VectorXd dq_n = A.fullPivLu().solve(b);
        //VectorXd q_n = q + h * dq_n;
        //q_n = q_n.cwiseMax(lowerLimits);
        //q_n = q_n.cwiseMin(upperLimits);
        //dq_n = (q_n - q) / h;
        //skeleton->setPositions(q_n);
        //if (dq_n.norm() == 0)
        //{
        //    cout << "policyJacobian:\n" << policyJacobian << endl;
        //    cout << "aS:\n" << aS << endl;
        //    cout << "sS:\n" << sS << endl;
        //    cout << "A:\n" << A << endl;
        //    cout << "M:\n" << M << endl;
        //    cout << "J:\n" << J << endl;
        //    cout << "b:\n" << b << endl;
        //    cout << "dq:\n" << endl;
        //    cout << "force:\n" << force << endl;
        //    cout << "C:\n" << C << endl;
        //    exit(0);
        //}
        skeleton->setVelocities(dq_n);
        //cout << "dq_n: " << dq_n.transpose() << endl;
        world->getConstraintSolver()->solve();
        if (skeleton->isImpulseApplied())
        {
          skeleton->computeImpulseForwardDynamics();
          skeleton->setImpulseApplied(false);
        }
        skeleton->integratePositions(h);
        //cout << "v: " << skeleton->getVelocities().transpose() << endl;
        //cout << "p: " << skeleton->getPositions().transpose() << endl;
        if (skeleton->getPositions().array().isNaN().any() || skeleton->getVelocities().array().isNaN().any())
        {
            done = true;
            cout << "done" << endl;
            skeleton->setPositions(q);
            skeleton->setVelocities(dq);
            break;
        }
        world->setTime(world->getTime() + h);
#elif SIM_MODE == 1
        VectorXd f = force + i * h * J * dq;
        skeleton->setForces(f);
        //cout << "f: " << f.transpose() << endl;
        world->step();
        if (skeleton->getPositions().array().isNaN().any() || skeleton->getVelocities().array().isNaN().any())
        {
            done = true;
            cout << "done" << endl;
            skeleton->setPositions(q);
            skeleton->setVelocities(dq);
            break;
        }
#else
        skeleton->setForces(force);
        world->step();
#endif
        dart::collision::CollisionResult result = world->getLastCollisionResult();
        if (result.inCollision(skeleton->getBodyNodes()[0]))
        {
            fallen = true;
            break;
        }
    }
    updateState();
    //cout << skeleton->getPositions().transpose() << endl;
}

void AntEnv::updateState()
{
    state << skeleton->getPositions(), skeleton->getVelocities();
    Eigen::Vector3d curr_com = skeleton->getRootBodyNode()->getCOM();
    done = curr_com.y() < 0.20 || curr_com.y() > 1.20 || fallen;
    reward = 1 * ((curr_com.x() - prev_com.x()) * actionRate - 0.5) + exp(-action.norm()) + (done ? 0 : 1);
    //cout << curr_com.transpose() << endl;
}
