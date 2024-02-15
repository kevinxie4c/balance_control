#ifndef CUSTOM_EVENT_HANDLER_H
#define CUSTOM_EVENT_HANDLER_H

#include <dart/gui/osg/osg.hpp>
#include "CharacterEnv.h"

class CustomEventHandler: public osgGA::GUIEventHandler
{
    public:
        CustomEventHandler(CharacterEnv *env);
        virtual bool handle(const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& aa) override;

        CharacterEnv *env = nullptr;
};

#endif
