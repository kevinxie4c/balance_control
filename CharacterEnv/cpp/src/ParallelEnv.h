#ifndef PARALLEL_ENV_H
#define PARALLEL_ENV_H

#include <vector>
#include <queue>
#include <shared_ptr>
#include "CharacterEnv.h"

class ParallelEnv
{
    public:
	int num_threads;
	std::vector<std::shared_ptr<CharacterEnv>> envs;

	ParallelEnv(const char *cfgFilename, size_t num_threads);
	size_t ParallelEnv::get_task_done_id();
	~ParallelEnv();

    private:
	std::vector<pthread_t> threads;
	std::queue<size_t> task_todo;
	std::queue<size_t> task_done;
	bool stop_all;

	pthread_mutex_t todo_lock, done_lock;
	std::vector<pthread_cond_t> work_conds;
	std::vector<pthread_mutex_t> work_locks;
	pthread_cond_t done_cond;
	std::vector<size_t> ids;

	void *env_step(void *arg);
};

#endif
