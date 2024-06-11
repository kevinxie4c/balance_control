#define PERL_NO_GET_CONTEXT

// Perl header files define some macros that conflict with the C++ header files. So put the header here.
#include "CharacterEnv.h"
#include "ParallelEnv.h"


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

typedef CharacterEnv* CharacterEnvPtrArray;


MODULE = CharacterEnv		PACKAGE = CharacterEnv

CharacterEnv *
CharacterEnv::new(const char *cfgFilename)
CODE:
    RETVAL = CharacterEnv::makeEnv(cfgFilename);
OUTPUT:
    RETVAL


MODULE = CharacterEnv		PACKAGE = CharacterEnvPtr

#void
#CharacterEnv::DESTROY()


void
CharacterEnv::reset()


void
CharacterEnv::step()


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
CharacterEnv::enable_RSI(int enable)
CODE:
    THIS->enableRSI = (bool)enable;


void
CharacterEnv::print_info()


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


int
CharacterEnv::get_done()
CODE:
    RETVAL = (int)THIS->done;
OUTPUT:
    RETVAL


double
CharacterEnv::get_period()
CODE:
    RETVAL = THIS->period;
OUTPUT:
    RETVAL


double
CharacterEnv::get_phase()
CODE:
    RETVAL = THIS->phase;
OUTPUT:
    RETVAL


size_t
CharacterEnv::get_state_size()
CODE:
    RETVAL = THIS->state.size();
OUTPUT:
    RETVAL


size_t
CharacterEnv::get_action_size()
CODE:
    RETVAL = THIS->action.size();
OUTPUT:
    RETVAL


void
CharacterEnv::run_viewer()


void
CharacterEnv::render_viewer()


bool
CharacterEnv::viewer_done()


bool
CharacterEnv::is_playing()
CODE:
    RETVAL = THIS->playing;
OUTPUT:
    RETVAL


void
CharacterEnv::set_policy_jacobian_row(size_t row_num, doubleArray * array, ...)
CODE:
    if (ix_array == THIS->policyJacobian.cols())
    {
        for (size_t i = 0; i < (size_t)ix_array; ++i)
            THIS->policyJacobian(row_num, i) = array[i];
    }
    else
        croak("CharacterEnv::set_jacobian_row(row_num, ...) -- incorrect number of arguments");
CLEANUP:
    free(array);


void
CharacterEnv::set_normalizer_mean(doubleArray * array, ...)
CODE:
    if (ix_array == THIS->normalizerMean.size())
    {
	memcpy(THIS->normalizerMean.data(), array, ix_array * sizeof(double));
    }
    else
	croak("CharacterEnv::set_normalizer_mean(...) -- incorrect number of arguments");
CLEANUP:
    free(array);


void
CharacterEnv::set_normalizer_std(doubleArray * array, ...)
CODE:
    if (ix_array == THIS->normalizerStd.size())
    {
	memcpy(THIS->normalizerStd.data(), array, ix_array * sizeof(double));
    }
    else
	croak("CharacterEnv::set_normalizer_std(...) -- incorrect number of arguments");
CLEANUP:
    free(array);
    

MODULE = CharacterEnv		PACKAGE = ParallelEnv

ParallelEnv *
ParallelEnv::new(const char *cfgFilename, size_t num_threads)
CODE:
    RETVAL = new ParallelEnv(cfgFilename, num_threads);
OUTPUT:
    RETVAL


MODULE = CharacterEnv		PACKAGE = ParallelEnvPtr

void
ParallelEnv::DESTROY()


CharacterEnvPtrArray *
ParallelEnv::get_env_list()
PREINIT:
    U32 size_RETVAL;
CODE:
    size_RETVAL = THIS->envs.size();
    RETVAL = THIS->envs.data();
OUTPUT:
    RETVAL
CLEANUP:
    XSRETURN(size_RETVAL);


void
ParallelEnv::reset()


size_t
ParallelEnv::get_task_done_id()


void
ParallelEnv::step(size_t id)


void
ParallelEnv::print_task_done()
