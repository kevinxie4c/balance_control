#/usr/bin/env perl
use FindBin;
use blib "$FindBin::Bin/blib";
use CharacterEnv;

my $env = CharacterEnv->new('../data/humanoid.json', '../data/positions.txt');
open my $fout, '>', 'positions.txt';
for (1 .. 10) {
    print $env->get_time, "\n";
    print join(' ', $env->get_state_list), "\n";
    print $fout join(' ', $env->get_positions_list), "\n";
    $env->step;
}
