#!/usr/bin/env perl

($m, $lx, $ly, $lz) = @ARGV;
$m = 1 unless $m;
$lx = 1 unless $lx;
$ly = $lx unless $ly;
$lz = $lx unless $lz;

$sx = $lx ** 2;
$sy = $ly ** 2;
$sz = $lz ** 2;

$a = $m / 12;

printf "%f, %f, %f\n", $a * ($sy + $sz), $a * ($sx + $sz), $a * ($sx + $sy);
