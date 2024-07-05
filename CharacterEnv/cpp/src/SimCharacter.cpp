#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <Eigen/Core>
#include <dart/dart.hpp>
#include <nlohmann/json.hpp>
#include "SimCharacter.h"

using namespace dart::dynamics;

SimCharacter::SimCharacter(std::string filename)
{
    std::ifstream input(filename);
    if (input.fail())
	throw std::ios_base::failure("cannot open " + filename);
    nlohmann::json json;
    input >> json;
    skeleton = Skeleton::create(filename);
    createJoint(json, nullptr);
    input.close();
}

void SimCharacter::createJoint(nlohmann::json json, dart::dynamics::BodyNodePtr parent)
{
    if (json.is_object())
    {
	std::string name = json["name"];
	std::vector<double> v = json["pos"].get<std::vector<double>>();
	Eigen::Vector3d pos(v[0], v[1], v[2]);
	std::string type = json["type"];
	BodyNodePtr bn;
	if (type == "free")
	{
	    FreeJoint::Properties properties;
	    properties.mName = name;
	    properties.mT_ParentBodyToJoint.translation() = pos;
	    bn = skeleton->createJointAndBodyNodePair<FreeJoint>(parent, properties, BodyNode::AspectProperties(name)).second;
	}
	else if (type == "ball")
	{
	    BallJoint::Properties properties;
	    properties.mName = name;
	    properties.mT_ParentBodyToJoint.translation() = pos;
	    bn = skeleton->createJointAndBodyNodePair<BallJoint>(parent, properties, BodyNode::AspectProperties(name)).second;
	}
	else if (type == "planar")
	{
	    PlanarJoint::Properties properties;
	    properties.mName = name;
	    properties.mT_ParentBodyToJoint.translation() = pos;
	    std::string plane = json["plane"];
	    if (plane == "xy")
		properties.setXYPlane();
	    else if (plane == "yz")
		properties.setYZPlane();
	    else if (plane == "zx")
		properties.setZXPlane();
	    else
		std::cerr << "unknown plane: " + plane << std::endl;
	    bn = skeleton->createJointAndBodyNodePair<PlanarJoint>(parent, properties, BodyNode::AspectProperties(name)).second;
	}
	else if (type == "revolute")
	{
	    RevoluteJoint::Properties properties;
	    properties.mName = name;
	    properties.mT_ParentBodyToJoint.translation() = pos;
	    std::string axis = json["axis"];
	    if (axis == "x")
		properties.mAxis = Eigen::Vector3d::UnitX();
	    else if (axis == "y")
		properties.mAxis = Eigen::Vector3d::UnitY();
	    else if (axis == "z")
		properties.mAxis = Eigen::Vector3d::UnitZ();
	    else
		std::cerr << "unknown axis: " + axis << std::endl;
	    bn = skeleton->createJointAndBodyNodePair<RevoluteJoint>(parent, properties, BodyNode::AspectProperties(name)).second;
	}
        else if (type == "weld")
        {
            WeldJoint::Properties properties;
	    properties.mName = name;
	    properties.mT_ParentBodyToJoint.translation() = pos;
	    bn = skeleton->createJointAndBodyNodePair<WeldJoint>(parent, properties, BodyNode::AspectProperties(name)).second;
        }
        else if (type == "prismatic")
        {
	    PrismaticJoint::Properties properties;
	    properties.mName = name;
	    properties.mT_ParentBodyToJoint.translation() = pos;
	    std::string axis = json["axis"];
	    if (axis == "x")
		properties.mAxis = Eigen::Vector3d::UnitX();
	    else if (axis == "y")
		properties.mAxis = Eigen::Vector3d::UnitY();
	    else if (axis == "z")
		properties.mAxis = Eigen::Vector3d::UnitZ();
	    else
		std::cerr << "unknown axis: " + axis << std::endl;
	    bn = skeleton->createJointAndBodyNodePair<PrismaticJoint>(parent, properties, BodyNode::AspectProperties(name)).second;
        }
        else
            std::cerr << "unknown joint type: " << type << std::endl;
	if (json.contains("mass"))
	    bn->setMass(json["mass"]);
	if (json.contains("COM"))
	{
	    std::vector<double> com = json["COM"].get<std::vector<double>>();
	    bn->setLocalCOM(Eigen::Vector3d(com[0], com[1], com[2]));
	}
	if (json.contains("MOI"))
	{
	    std::vector<double> moi = json["MOI"].get<std::vector<double>>();
	    bn->setMomentOfInertia(moi[0], moi[1], moi[2], moi[3], moi[4], moi[5]);
	}
	if (json.contains("shape"))
	{
	    for (auto &shape: json["shape"])
	    {
		std::shared_ptr<Shape> dShape;
		Eigen::Isometry3d tf = Eigen::Isometry3d::Identity();
		if (shape["type"] == "box")
		{
		    std::vector<double> size = shape["size"].get<std::vector<double>>();
		    std::vector<double> pos = shape["pos"].get<std::vector<double>>();
		    tf.translation() = Eigen::Vector3d(pos[0], pos[1], pos[2]);
		    dShape = std::make_shared<BoxShape>(Eigen::Vector3d(size[0], size[1], size[2]));
		}
		else if (shape["type"] == "mesh")
		{
		    const auto retriever = std::make_shared<dart::common::LocalResourceRetriever>();
		    dShape = std::make_shared<MeshShape>(Eigen::Vector3d::Ones(), MeshShape::loadMesh("file:" + std::string(shape["path"]), retriever));
		}
		else
		    std::cerr << "unknown shape: " << shape["type"] << std::endl;
		auto shapeNode = bn->createShapeNodeWith<CollisionAspect, DynamicsAspect, VisualAspect>(dShape);
		//auto shapeNode = bn->createShapeNodeWith<CollisionAspect, DynamicsAspect>(dShape);
		shapeNode->setRelativeTransform(tf);
                if (shape.contains("color"))
                {
                    std::vector<double> color = shape["color"].get<std::vector<double>>();
                    shapeNode->getVisualAspect()->setColor(Eigen::Vector3d(color[0], color[1], color[2]));
                }
                else 
                    shapeNode->getVisualAspect()->setColor(Eigen::Vector3d(1.0, 1.0, 0.0));
	    }
	}

	if (json.contains("children"))
	{
	    for (auto &child: json["children"])
		createJoint(child, bn);
	}
    }
}
