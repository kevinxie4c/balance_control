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
        else if (ea.getKey() == osgGA::GUIEventAdapter::KEY_R)
        {
            env->reset();
            return true;
        }
        else if (ea.getKey() == osgGA::GUIEventAdapter::KEY_V)
        {
            if (env->viewer->isRecording())
                env->viewer->pauseRecording();
            else
                env->viewer->record("img");
        }
        else if (ea.getKey() ==osgGA::GUIEventAdapter::KEY_S && !env->playing)
        {
            env->reqStep = true;
        }
    }
    return false;
}
