#/usr/bin/env perl
use FindBin;
use blib "$FindBin::Bin/blib";
use CharacterEnv;

my $env = CharacterEnv->new('../data/humanoid.json');
