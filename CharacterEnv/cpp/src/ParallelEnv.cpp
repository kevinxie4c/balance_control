#include <iostream>
#include <pthread.h>
#include <cassert>
#include "ParallelEnv.h"

using namespace std;

ParallelEnv::ParallelEnv(const char *cfgFilename, size_t num_threads): stop_all(false), num_threads(num_threads);
{
    pthread_mutex_init(&todo_lock, NULL);
    pthread_mutex_init(&done_lock, NULL);
    pthread_cond_init(&done_cond, NULL);
    threads = vector<pthread_t>(num_threads);
    work_conds = vector<pthread_cond_t>(num_threads);
    work_locks = vector<pthread_mutex_t>(num_threads);
    for (size_t i = 0; i < num_threads; ++i)
    {
	pthread_cond_t(&work_conds[i], NULL);
	pthread_mutex_init(&work_locks[i], NULL);
	envs.push_back(make_shared<CharacterEnv>(cfgFilename));
	ids.push_back(i);
	task_done.push(i);
	pthread_create(&threads[i], NULL, env_step, &ids[i]);
	pthread_detach(threads[i]);
    }
}

size_t ParallelEnv::get_task_done_id()
{
    size_t id;
    pthread_mutex_lock(&done_lock);
    if (task_done.size() == 0)
	pthread_cond_wait(&done_cond, done_lock)
    id = task_done.front();
    task_done.pop();
    pthread_mutex_unlock(&done_lock);
    return id;
}

void ParallelEnv::step(size_t id)
{
    task_todo.push_back(id);
    pthread_cond_signal();
}

void *ParallelEnv::env_step(void *arg)
{
    size_t id = *(size_t*)arg;
    cout << "thread #" << id << " starts" << endl;

    while (1)
    {
	pthread_mutex_lock(&work_locks[id]);
	if (task_todo.empty() || task_todo.front() != id)
	    pthread_cond_wait(&work_conds[id], &work_locks[id]);
	pthread_mutex_unlock(&work_locks[id]);

	if (stop_all)
	    break;

	pthread_mutex_lock(&todo_lock);
	assert(task_todo.front() == id);
	task_todo.pop();
	pthread_mutex_unlock(&todo_lock);

	env[id].step();

	pthread_mutex_lock(&done_lock);
	task_done.push(id);
	pthread_mutex_unlock(&done_lock);
	pthread_cond_signal(&done_cond);
    }

    cout << "thread #" << id << " ends" << endl;
    pthread_exit(NULL);
}

ParallelEnv::~ParallelEnv()
{
    stop_all = true;
    for (size_t i = 0; i < num_threads; ++i)
	pthread_cond_signal(&work_locks[i]);
    pthread_mutex_destroy(&todo_lock);
    pthread_mutex_destroy(&done_lock);
    pthread_cond_destroy(&done_cond);
    for (size_t i = 0; i < num_threads; ++i)
    {
	pthread_cond_destroy(&work_conds[i]);
	pthread_mutex_destroy(&work_lock[i]);
    }
}
