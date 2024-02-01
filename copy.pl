#!/usr/bin/env perl
use File::Copy;

my $suffix = shift @ARGV;
copy "model/actor-$suffix.par", "model/actor.par";
copy "model/critic-$suffix.par", "model/critic.par";
copy "model/state_normalizer-$suffix.nd", "state_normalizer.nd";
