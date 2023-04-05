#/usr/bin/env perl
use FindBin;
use blib "$FindBin::Bin/blib";
use CharacterEnv;

my $env = CharacterEnv->new('data/env_config.json');
$env->set_action_list((0) x 69);
open my $fout, '>', 'positions.txt';
print "period: ", $env->get_period, "\n";
print "reward: ", $env->get_reward, "\n";
for (1 .. 100) {
    print "time: ", $env->get_time, "\n";
    print "phase: ", $env->get_phase, "\n";
    #print join(' ', $env->get_state_list), "\n";
    #print $fout join(' ', $env->get_positions_list), "\n";
    print "reward: ", $env->get_reward, "\n";
    $env->step;
}
