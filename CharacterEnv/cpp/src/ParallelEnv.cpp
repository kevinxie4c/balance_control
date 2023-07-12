#include <iostream>
#include <pthread.h>
#include <cassert>
#include "ParallelEnv.h"

using namespace std;

ParallelEnv::ParallelEnv(const char *cfgFilename, size_t num_threads): stop_all(false), num_threads(num_threads)
{
    cout << "ParallelEnv init" << endl;
    pthread_mutex_init(&todo_lock, NULL);
    pthread_mutex_init(&done_lock, NULL);
    pthread_cond_init(&done_cond, NULL);
    threads = vector<pthread_t>(num_threads);
    work_conds = vector<pthread_cond_t>(num_threads);
    work_locks = vector<pthread_mutex_t>(num_threads);
    for (size_t i = 0; i < num_threads; ++i)
    {
	pthread_cond_init(&work_conds[i], NULL);
	pthread_mutex_init(&work_locks[i], NULL);
	//envs.push_back(make_shared<CharacterEnv>(cfgFilename));
	envs.push_back(new CharacterEnv(cfgFilename));
	args.push_back({ this, i });
	task_done.push(i);
	pthread_create(&threads[i], NULL, env_step, &args[i]);
	pthread_detach(threads[i]);
    }
}

size_t ParallelEnv::get_task_done_id()
{
    size_t id;
    pthread_mutex_lock(&done_lock);
    if (task_done.size() == 0)
	pthread_cond_wait(&done_cond, &done_lock);
    id = task_done.front();
    task_done.pop();
    pthread_mutex_unlock(&done_lock);
    return id;
}

void ParallelEnv::step(size_t id)
{
    pthread_mutex_lock(&todo_lock);
    task_todo.push(id);
    pthread_mutex_unlock(&todo_lock);
    pthread_cond_signal(&work_conds[id]);
}

void *ParallelEnv::env_step(void *arg)
{
    ParallelEnv *env = ((ParaArg*)arg)->env;
    size_t id = ((ParaArg*)arg)->id;
    cout << "thread #" << id << " starts" << endl;

    while (1)
    {
	pthread_mutex_lock(&env->work_locks[id]);
	if (env->task_todo.empty() || env->task_todo.front() != id)
	    pthread_cond_wait(&env->work_conds[id], &env->work_locks[id]);
	pthread_mutex_unlock(&env->work_locks[id]);

	if (env->stop_all)
	    break;

	pthread_mutex_lock(&env->todo_lock);
	assert(env->task_todo.front() == id);
	env->task_todo.pop();
	pthread_mutex_unlock(&env->todo_lock);

	env->envs[id]->step();

	pthread_mutex_lock(&env->done_lock);
	env->task_done.push(id);
	pthread_mutex_unlock(&env->done_lock);
	pthread_cond_signal(&env->done_cond);
    }

    cout << "thread #" << id << " ends" << endl;
    pthread_exit(NULL);
}

ParallelEnv::~ParallelEnv()
{
    stop_all = true;
    for (size_t i = 0; i < num_threads; ++i)
	pthread_cond_signal(&work_conds[i]);
    pthread_mutex_destroy(&todo_lock);
    pthread_mutex_destroy(&done_lock);
    pthread_cond_destroy(&done_cond);
    for (size_t i = 0; i < num_threads; ++i)
    {
	pthread_cond_destroy(&work_conds[i]);
	pthread_mutex_destroy(&work_locks[i]);
    }
    for (CharacterEnv* ptr: envs)
	delete ptr;
    cout << "ParallelEnv destory" << endl;
}
