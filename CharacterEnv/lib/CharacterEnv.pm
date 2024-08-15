package CharacterEnv;

use 5.014000;
use strict;
use warnings;

require Exporter;

our @ISA = qw(Exporter);

# Items to export into callers namespace by default. Note: do not export
# names by default without a very good reason. Use EXPORT_OK instead.
# Do not simply export all your public functions/methods/constants.

# This allows declaration	use CharacterEnv ':all';
# If you do not need this, moving things directly into @EXPORT or @EXPORT_OK
# will save memory.
our %EXPORT_TAGS = ( 'all' => [ qw(
	
) ] );

our @EXPORT_OK = ( @{ $EXPORT_TAGS{'all'} } );

our @EXPORT = qw(
	
);

our $VERSION = '0.01';

require XSLoader;
XSLoader::load('CharacterEnv', $VERSION);

# Preloaded methods go here.

use AI::MXNet qw(mx);
use AI::MXNet::Base;

sub _mxar2matrix {
    my $a = shift;
    my $m = shift;
    check_call(AI::MXNetCAPI::NDArraySyncCopyToCPU($a->handle, ${$m->data}, $a->size));
}

sub _matrix2mxar {
    my $m = shift;
    my $a = shift;
    check_call(AI::MXNetCAPI::NDArraySyncCopyFromCPU($a->handle, ${$m->data}, $a->size));
}

package CharacterEnvPtr;

sub set_positions {
    my $self = shift;
    my $a = shift;
    my $m = Eigen::MatrixXf->new($a->shape->[0], 1);
    CharacterEnv::_mxar2matrix($a, $m);
    $self->set_positions_matrix($m);
}

sub get_positions {
    my $self = shift;
    my $m = $self->get_positions_matrix;
    my $a = mx->nd->empty([$m->rows]);
    CharacterEnv::_matrix2mxar($m, $a);
    return $a;
}

package OMPEnvPtr;

sub get_observations {
    my $self = shift;
    my $m = $self->get_observations_matrix;
    my $a = mx->nd->empty([$m->cols, $m->rows]);
    CharacterEnv::_matrix2mxar($m, $a);
    return $a;
}

sub set_means {
    my $self = shift;
    my $a = shift;
    my $m = Eigen::MatrixXf->new($a->shape->[1], $a->shape->[0]);
    CharacterEnv::_mxar2matrix($a, $m);
    $self->set_means_matrix($m);
}

sub set_stds {
    my $self = shift;
    my $a = shift;
    my $m = Eigen::MatrixXf->new($a->shape->[1], $a->shape->[0]);
    CharacterEnv::_mxar2matrix($a, $m);
    $self->set_stds_matrix($m);
}

1;
__END__
# Below is stub documentation for your module. You'd better edit it!

=head1 NAME

CharacterEnv - Perl extension for blah blah blah

=head1 SYNOPSIS

  use CharacterEnv;
  blah blah blah

=head1 DESCRIPTION

Stub documentation for CharacterEnv, created by h2xs. It looks like the
author of the extension was negligent enough to leave the stub
unedited.

Blah blah blah.

=head2 EXPORT

None by default.



=head1 SEE ALSO

Mention other useful documentation such as the documentation of
related modules or operating system documentation (such as man pages
in UNIX), or any relevant external documentation such as RFCs or
standards.

If you have a mailing list set up for your module, mention it here.

If you have a web site set up for your module, mention it here.

=head1 AUTHOR

kevin, E<lt>kevin@(none)E<gt>

=head1 COPYRIGHT AND LICENSE

Copyright (C) 2023 by kevin

This library is free software; you can redistribute it and/or modify
it under the same terms as Perl itself, either Perl version 5.36.0 or,
at your option, any later version of Perl 5 you may have available.


=cut
