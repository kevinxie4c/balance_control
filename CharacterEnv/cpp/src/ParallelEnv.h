#ifndef PARALLEL_ENV_H
#define PARALLEL_ENV_H

#include <vector>
#include <deque>
#include <memory>
#include "CharacterEnv.h"

struct ParaArg;

class ParallelEnv
{
    public:
	int num_threads;
	//std::vector<std::shared_ptr<CharacterEnv>> envs;
	std::vector<CharacterEnv*> envs;

	ParallelEnv(const char *cfgFilename, size_t num_threads);
	void reset();
	size_t get_task_done_id();
	void step(size_t id);
	void print_task_done();
	~ParallelEnv();

    private:
	std::vector<pthread_t> threads;
	std::vector<bool> task_todo;
	std::deque<size_t> task_done;
	bool stop_all;

	pthread_mutex_t done_lock;
	std::vector<pthread_cond_t> work_conds;
	pthread_mutex_t work_lock;
	pthread_cond_t done_cond;
	std::vector<size_t> ids;
	std::vector<ParaArg> args;

	static void *env_step(void *arg);
};

struct ParaArg
{
    ParallelEnv *env;
    size_t id;
};

#endif
