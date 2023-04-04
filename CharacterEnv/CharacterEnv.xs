#define PERL_NO_GET_CONTEXT

// Perl header files define some macros that conflict with the C++ header files. So put the header here.
#include "CharacterEnv.h"


#ifdef __cplusplus
extern "C" {
#endif
#include "EXTERN.h"
#include "perl.h"
#include "XSUB.h"

#include "ppport.h"
#ifdef __cplusplus
}
#endif

typedef double doubleArray;

doubleArray * doubleArrayPtr(int num)
{
    return new doubleArray[num];
}

MODULE = CharacterEnv		PACKAGE = CharacterEnv		

CharacterEnv *
CharacterEnv::new(const char *characterFilename, const char *poseFilename)
CODE:
    RETVAL = new CharacterEnv(characterFilename, poseFilename);
OUTPUT:
    RETVAL


MODULE = CharacterEnv		PACKAGE = CharacterEnvPtr

void
CharacterEnv::reset()
CODE:
    THIS->reset();


void
CharacterEnv::step()
CODE:
    THIS->step();


double
CharacterEnv::get_time()
CODE:
    RETVAL = THIS->getTime();
OUTPUT:
    RETVAL


double
CharacterEnv::get_time_step()
CODE:
    RETVAL = THIS->getTimeStep();
OUTPUT:
    RETVAL


#void
#CharacterEnv::set_time_step(double h)
#CODE:
#    THIS->setTimeStep(h);


void
CharacterEnv::set_action_list(doubleArray * array, ...)
CODE:
    if (ix_array == THIS->action.size())
    {
	memcpy(THIS->action.data(), array, ix_array * sizeof(double));
    }
    else
	croak("CharacterEnv::set_action(...) -- incorrect number of arguments");
CLEANUP:
    free(array);
    

doubleArray *
CharacterEnv::get_state_list()
PREINIT:
    doubleArray* state;
    U32 size_RETVAL;
CODE:
    size_RETVAL = THIS->state.size();
    RETVAL = THIS->state.data();
OUTPUT:
    RETVAL
CLEANUP:
    XSRETURN(size_RETVAL);


void
CharacterEnv::set_positions_list(doubleArray * array, ...)
CODE:
    if (ix_array == THIS->skeleton->getNumDofs())
    {
	Eigen::VectorXd v(ix_array);
	memcpy(v.data(), array, ix_array * sizeof(double));
	THIS->skeleton->setPositions(v);
    }
    else
	croak("CharacterEnv::set_positions(...) -- incorrect number of arguments");
CLEANUP:
    free(array);
    

doubleArray *
CharacterEnv::get_positions_list()
PREINIT:
    U32 size_RETVAL;
CODE:
    Eigen::VectorXd v = THIS->skeleton->getPositions();
    size_RETVAL = v.size();
    RETVAL = v.data();
OUTPUT:
    RETVAL
CLEANUP:
    XSRETURN(size_RETVAL);


double
CharacterEnv::get_reward()
CODE:
    RETVAL = THIS->reward;
OUTPUT:
    RETVAL
