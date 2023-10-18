#!/usr/bin/env perl
use FindBin;
use Time::HiRes qw(time);
use blib "$FindBin::Bin/CharacterEnv/blib";
use CharacterEnv;

my $tm_total = time;
my $step_time;

print "CharacterEnv test\n";
my $env = CharacterEnv->new('data/env_config.json');
print "state_size: ", $env->get_state_size, "\n";
print "action_size: ", $env->get_action_size, "\n";
$env->set_action_list((0) x 14);
open my $fout, '>', 'positions.txt';
print "period: ", $env->get_period, "\n";
print "reward: ", $env->get_reward, "\n";
for (1 .. 10) {
    print "time: ", $env->get_time, "\n";
    #print "phase: ", $env->get_phase, "\n";
    #print join(' ', $env->get_state_list), "\n";
    #print $fout join(' ', $env->get_positions_list), "\n";
    #print "reward: ", $env->get_reward, "\n";
    my $tm = time;
    $env->step;
    $step_time += (time - $tm);
}

my $total_time = time - $tm_total;
print "total_time: $total_time\n";
print "step_time: $step_time\n";
print "step_time / total_time: ", $step_time / $total_time, "\n";

print "ParallelEnv test\n";
my $num_threads = 4;
$env = ParallelEnv->new('data/env_config.json', $num_threads);
print $env, "\n";
my @envs = $env->get_env_list;
print "# of envs: ", scalar(@envs), "\n";
print "envs[0] $envs[0]\n";
#print "envs[1]->time ", $envs[1]->get_time, "\n";

for (1 .. 400) {
    my $id = $env->get_task_done_id;
    print "$id\n";
    $env->step($id);
}
my $all_done = 0;
my @dones = (0) x $num_threads;
while (!$all_done) {
    my $id = $env->get_task_done_id;
    $dones[$id] = 1;
    $all_done = 1;
    for (@dones) {
	$all_done = $all_done && $_;
    }
}
for my $id (0 .. $#envs) {
    print "envs[$id]->time ", $envs[$id]->get_time, "\n";
}
