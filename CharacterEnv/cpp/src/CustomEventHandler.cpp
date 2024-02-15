#include "CustomEventHandler.h"

CustomEventHandler::CustomEventHandler(CharacterEnv *env): env(env) {}

bool CustomEventHandler::handle(const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& aa)
{
    if (ea.getEventType() == osgGA::GUIEventAdapter::KEYDOWN)
    {
        if (ea.getKey() == osgGA::GUIEventAdapter::KEY_Space)
        {
            env->playing = !env->playing;
            return true;
        }
        if (ea.getKey() == osgGA::GUIEventAdapter::KEY_R)
        {
            env->reset();
            return true;
        }
    }
    return false;
}
