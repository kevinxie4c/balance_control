#define PERL_NO_GET_CONTEXT

// Perl header files define some macros that conflict with the C++ header files. So put the header here.
#include "CharacterEnv.h"
#include "ParallelEnv.h"
#include "OMPEnv.h"


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

typedef Eigen::MatrixXf Eigen__MatrixXf;


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
	croak("CharacterEnv::set_positions_list(...) -- incorrect number of arguments");
CLEANUP:
    free(array);


void
CharacterEnv::set_velocities_list(doubleArray * array, ...)
CODE:
    if (ix_array == THIS->skeleton->getNumDofs())
    {
	Eigen::VectorXd v(ix_array);
	memcpy(v.data(), array, ix_array * sizeof(double));
	THIS->skeleton->setVelocities(v);
    }
    else
	croak("CharacterEnv::set_velocities_list(...) -- incorrect number of arguments");
CLEANUP:
    free(array);


void
CharacterEnv::set_positions_matrix(Eigen::MatrixXf* m)
CODE:
    //std::cout << m->transpose() << std::endl;
    THIS->skeleton->setPositions(m->cast<double>());
    

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


doubleArray *
CharacterEnv::get_velocities_list()
PREINIT:
    U32 size_RETVAL;
CODE:
    Eigen::VectorXd v = THIS->skeleton->getVelocities();
    size_RETVAL = v.size();
    RETVAL = v.data();
OUTPUT:
    RETVAL
CLEANUP:
    XSRETURN(size_RETVAL);


Eigen::MatrixXf *
CharacterEnv::get_positions_matrix()
CODE:
    Eigen::MatrixXf *v = new Eigen::MatrixXf(THIS->skeleton->getPositions().cast<float>());
    RETVAL = v;
OUTPUT:
    RETVAL


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


bool
CharacterEnv::req_step()
CODE:
    RETVAL = THIS->reqStep;
    THIS->reqStep = false;
OUTPUT:
    RETVAL


MODULE = CharacterEnv           PACKAGE = Eigen::MatrixXf

Eigen::MatrixXf *
Eigen::MatrixXf::new(size_t rows, size_t cols)
CODE:
    RETVAL = new Eigen::MatrixXf(rows, cols);
    //std::cout << std::hex << RETVAL->data() << std::endl;
OUTPUT:
    RETVAL


MODULE = CharacterEnv           PACKAGE = Eigen::MatrixXfPtr

void
Eigen::MatrixXf::DESTROY()

void *
Eigen::MatrixXf::data()
CODE:
    RETVAL = THIS->data();
    //std::cout << *THIS << std::endl;
    //std::cout << std::hex << RETVAL << std::endl;
OUTPUT:
    RETVAL

size_t
Eigen::MatrixXf::rows()
CODE:
    RETVAL = THIS->rows();
OUTPUT:
    RETVAL

size_t
Eigen::MatrixXf::cols()
CODE:
    RETVAL = THIS->cols();
OUTPUT:
    RETVAL
    

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


MODULE = CharacterEnv		PACKAGE = OMPEnv

OMPEnv *
OMPEnv::new(const char *cfgFilename, size_t num_threads)
CODE:
    RETVAL = new OMPEnv(cfgFilename, num_threads);
OUTPUT:
    RETVAL


MODULE = CharacterEnv		PACKAGE = OMPEnvPtr

void
OMPEnv::DESTROY()


CharacterEnvPtrArray *
OMPEnv::get_env_list()
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
OMPEnv::reset()


void
OMPEnv::step()


void
OMPEnv::set_means_matrix(Eigen::MatrixXf* m)
CODE:
    THIS->means = m->cast<double>();


void
OMPEnv::set_stds_matrix(Eigen::MatrixXf* m)
CODE:
    THIS->stds = m->cast<double>();


void
OMPEnv::set_values_matrix(Eigen::MatrixXf* m)
CODE:
    THIS->values = m->cast<double>();


Eigen::MatrixXf *
OMPEnv::get_observations_matrix()
CODE:
    Eigen::MatrixXf *m = new Eigen::MatrixXf(THIS->observations.cast<float>());
    RETVAL = m;
OUTPUT:
    RETVAL


Eigen::MatrixXf *
OMPEnv::get_obs_buffer_matrix()
CODE:
    Eigen::MatrixXf *m = new Eigen::MatrixXf(THIS->obs_buffer.cast<float>());
    RETVAL = m;
OUTPUT:
    RETVAL


Eigen::MatrixXf *
OMPEnv::get_act_buffer_matrix()
CODE:
    Eigen::MatrixXf *m = new Eigen::MatrixXf(THIS->act_buffer.cast<float>());
    RETVAL = m;
OUTPUT:
    RETVAL


Eigen::MatrixXf *
OMPEnv::get_adv_buffer_matrix()
CODE:
    Eigen::MatrixXf *m = new Eigen::MatrixXf(THIS->adv_buffer.cast<float>());
    RETVAL = m;
OUTPUT:
    RETVAL


Eigen::MatrixXf *
OMPEnv::get_ret_buffer_matrix()
CODE:
    Eigen::MatrixXf *m = new Eigen::MatrixXf(THIS->ret_buffer.cast<float>());
    RETVAL = m;
OUTPUT:
    RETVAL


Eigen::MatrixXf *
OMPEnv::get_logp_buffer_matrix()
CODE:
    Eigen::MatrixXf *m = new Eigen::MatrixXf(THIS->logp_buffer.cast<float>());
    RETVAL = m;
OUTPUT:
    RETVAL


void
OMPEnv::trace_back()
