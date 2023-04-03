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
CharacterEnv::new(const char *filename)
CODE:
    RETVAL = new CharacterEnv(filename);
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


void
CharacterEnv::set_action(doubleArray * array, ...)
CODE:
    if (ix_array == THIS->action.size())
    {
	for (size_t i = 0; i < ix_array; ++i)
	    THIS->action[i] = array[i];
    }
    else
	croak("CharacterEnv::set_action(...) -- incorrect number of arguments");
CLEANUP:
    free(array);
    

doubleArray *
CharacterEnv::get_state()
PREINIT:
    doubleArray* state;
    U32 size_RETVAL;
CODE:
    state = new double[THIS->state.size()];
    size_RETVAL = THIS->state.size();
    for (size_t i = 0; i < size_RETVAL; ++i)
	state[i] = THIS->state[i];
    RETVAL = state;
OUTPUT:
    RETVAL
CLEANUP:
    free(state);
    XSRETURN(size_RETVAL);


double
CharacterEnv::get_reward()
CODE:
    RETVAL = THIS->reward;
OUTPUT:
    RETVAL
