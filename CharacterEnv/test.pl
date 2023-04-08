#/usr/bin/env perl
use FindBin;
use Time::HiRes qw(time);
use blib "$FindBin::Bin/blib";
use CharacterEnv;

my $tm_total = time;
my $step_time;

my $env = CharacterEnv->new('data/env_config.json');
print "state_size: ", $env->get_state_size, "\n";
print "action_size: ", $env->get_action_size, "\n";
$env->set_action_list((0) x 69);
open my $fout, '>', 'positions.txt';
print "period: ", $env->get_period, "\n";
print "reward: ", $env->get_reward, "\n";
for (1 .. 100) {
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
