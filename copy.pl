#!/usr/bin/env perl
use File::Copy;

die "usage: $0 suffix [path]" unless @ARGV == 1 or @ARGV == 2;
my $suffix = shift @ARGV;
my $path = 'model';
if (@ARGV) {
    $path = shift @ARGV;
}
copy "$path/actor-$suffix.par", "$path/actor.par";
copy "$path/critic-$suffix.par", "$path/critic.par";
copy "$path/state_normalizer-$suffix.nd", "$path/state_normalizer.nd";
