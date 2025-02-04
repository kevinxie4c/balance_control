#!/usr/bin/env perl
use AI::MXNet qw(mx);
use AI::MXNet::Gluon::NN qw(nn);
use AI::MXNet::Gluon qw(gluon);
use AI::MXNet::AutoGrad qw(autograd);
use Getopt::Long qw(:config no_ignore_case);
use Time::HiRes qw(time);
use List::Util qw(shuffle);
use File::Slurp;
use JSON;
use Data::Dumper;
use strict;
use warnings;

my $use_gpu = undef;
my $load_model = undef;
my $save_model = 'model';
my $play_policy = 0;
my $reinit_logstd = 0;
my $save_interval = 200;
my $g_sigma = 0;
my $outdir = "output";
my $imgdir = "img";
my $num_threads = 4;
my $sigma_begin = 0.1;
my $sigma_end = 0.01;
my $decay_factor = 0.001;
my $steps_per_itr = 1024;
my $num_itrs = 5000;
my $mini_batch_size = 256;
my $a_scale = 1;
my $parameters = undef;
my $config_file = undef;
my $traj_file = undef;
my $print_help = 0;

GetOptions(
    'G|gpu:i'              => \$use_gpu,
    'l|load_model=s'       => \$load_model,
    's|save_model=s'       => \$save_model,
    'p|play_policy'        => \$play_policy,
    't|traj_file=s'        => \$traj_file,
    'r|reinit_logstd'      => \$reinit_logstd,
    'i|save_interval=i'    => \$save_interval,
    'n|num_threads=i'      => \$num_threads,
    'm|mini_batch_size=i'  => \$mini_batch_size,
    'd|decay_factor=f'     => \$decay_factor,
    'B|sigma_begin=f'      => \$sigma_begin,
    'E|sigma_end=f'        => \$sigma_end,
    'N|num_itrs=i'         => \$num_itrs,
    'K|steps_per_itr=i'    => \$steps_per_itr,
    'a|a_scale=f'          => \$a_scale,
    'P|parameters=s'       => \$parameters,
    'h|help'               => \$print_help,
);

if (@ARGV != 1 || $print_help) {
    die <<"HELP_MSG";
usage: $0 [options] config_file
options:
    -G --gpu=INT
    -l --load_model=STRING
    -s --save_model=STRING
    -p --play_policy
    -i --save_interval=INT
    -n --num_threads=INT
    -m --mini_batch_size=INT
    -N --num_itrs=INT
    -K --steps_per_itr=INT
    -a --a_scale=FLOAT
    -P --parameters=STRING
    -h --help
HELP_MSG
}

$config_file = shift @ARGV;

my $num_train_itrs = $steps_per_itr / $mini_batch_size;
if (int($num_train_itrs) != $num_train_itrs) {
    warn "please choose a better mini_batch_size";
    $num_train_itrs = int($num_train_itrs);
}

my $current_ctx = defined($use_gpu) ? mx->gpu($use_gpu) : mx->cpu;
AI::MXNet::Context->set_current($current_ctx);

my $test_tensor = mx->nd->zeros([2, 2]);
print "test tensor: $test_tensor\n";

$num_threads = 1 if $play_policy;

if (defined($save_model)) {
    mkdir $save_model unless -e $save_model;
} else {
    die "undefined save_model";
}

sub discounted_cumulative_sums {
    my ($x, $discount) = @_;
    my $n = $x->size;
    my $y = mx->nd->zeros([$n]);
    my $r = 0;
    for my $i (reverse(0 .. $n - 1)) {
        $r = $x->slice($i) + $discount * $r;
        $y->slice($i) .= $r;
    }
    $y;
}

sub nd_std {
    my $x = shift;
    $x = $x - $x->mean;
    $x = $x->abs ** 2;
    $x = $x->mean->sqrt;
}

sub reorder {
    my ($nd, $indices) = @_;
    return mx->nd->array($nd->aspdl->dice_axis(-1, $indices));
}

# discounted_cumulative_sums test
#my $x = mx->nd->ones([5]);
#my $y = discounted_cumulative_sums($x, 0.5);
#print $y->aspdl;

package Buffer {
    sub new {
        my ($class, $ob_dim, $ac_dim, $size, $gamma, $lam) = @_;
        my $self = bless {
            observation_buffer     => mx->nd->zeros([$size, $ob_dim]),
            action_buffer          => mx->nd->zeros([$size, $ac_dim]),
            advantage_buffer       => mx->nd->zeros([$size]),
            reward_buffer          => mx->nd->zeros([$size]),
            return_buffer          => mx->nd->zeros([$size]),
            value_buffer           => mx->nd->zeros([$size]),
            logprobability_buffer  => mx->nd->zeros([$size]),
            gamma => $gamma,
            lam => $lam,
            pointer => 0,
            trajectory_start_index => 0,
        }, $class;
    }

    sub store {
        my ($self, $observation, $action, $reward, $value, $logprobability) = @_;
        for ($observation, $action, $reward, $value, $logprobability) {
            my $nan = 0;
            if (ref eq '') {
                $nan = 0 if /NaN/;
            } elsif ($_->aspdl =~ /NaN/) {
                $nan = 0;
            }
            if ($nan) {
                print "store: NaN\n";
            }
        }
        my $pointer = $self->{pointer};
        $self->{observation_buffer}->slice($pointer) .= $observation;
        $self->{action_buffer}->slice($pointer) .= $action;
        $self->{reward_buffer}->slice($pointer) .= $reward;
        $self->{value_buffer}->slice($pointer) .= $value;
        $self->{logprobability_buffer}->slice($pointer) .= $logprobability;
        $self->{pointer} = $pointer + 1;
    }

    sub finish_trajectory {
        my ($self, $last_value) = @_;
        $last_value = 0 unless defined $last_value;
        my ($a, $b) = ($self->{trajectory_start_index}, $self->{pointer} - 1);
        my $rewards = mx->nd->concatenate([$self->{reward_buffer}->slice([$a, $b]), mx->nd->array([$last_value])]);
        my $values = mx->nd->concatenate([$self->{value_buffer}->slice([$a, $b]), mx->nd->array([$last_value])]);
        my $deltas = $rewards->slice([0,-2]) + $self->{gamma} * $values->slice([1,-1]) - $values->slice([0,-2]);

        $self->{advantage_buffer}->slice([$a, $b]) .= main::discounted_cumulative_sums($deltas, $self->{gamma} * $self->{lam});
        $self->{return_buffer}->slice([$a, $b]) .= main::discounted_cumulative_sums($rewards, $self->{gamma})->slice([0,-2]);

        $self->{trajectory_start_index} = $self->{pointer};
    }

    sub get {
        my $self = shift;
        my $b = $self->{pointer} - 1;
        ($self->{pointer}, $self->{trajectory_start_index}) = (0, 0);
        my $advantage_mean = $self->{advantage_buffer}->mean->aspdl->at(0);
        my $advantage_std = main::nd_std($self->{advantage_buffer})->aspdl->at(0);
        $self->{advantage_buffer} = ($self->{advantage_buffer} - $advantage_mean) / $advantage_std;
        return ($self->{observation_buffer}->slice([0, $b]), $self->{action_buffer}->slice([0, $b]), $self->{advantage_buffer}->slice([0, $b]), $self->{return_buffer}->slice([0, $b]), $self->{logprobability_buffer}->slice([0, $b]));
    }
}

use Math::Trig ();
use File::Slurp;
use PDL;
use FindBin;
use blib "$FindBin::Bin/CharacterEnv/blib";
use CharacterEnv;

my $omp_env = QOMPEnv->new($config_file, $num_threads);
my @envs = $omp_env->get_env_list;
#my $unit_size = 512;
my $state_size = $envs[0]->get_state_size;
my $action_size = $envs[0]->get_action_size;
print "state_size: $state_size\n";
print "action_size: $action_size\n";
$envs[0]->print_info;

package CriticModel {
    use AI::MXNet::Gluon::Mouse;
    use AI::MXNet::Function::Parameters; # must include this for function parameters
    extends 'AI::MXNet::Gluon::Block';

    sub BUILD {
        my $self = shift;
        my $attrs = shift;
        my $state_layers = $attrs->{state_layers};
        my $action_layers = $attrs->{action_layers};
        my $base_layers = $attrs->{base_layers};
        my $activation = $attrs->{activation};
        unless (defined($state_layers) && defined($action_layers) && defined($base_layers) && defined($activation)) {
            die "usage: CriticModel->new(state_layers => [size_1, size_2, ...], action_layers => [size_1, size_2, ...], base_layers => [size_1, size2, ...], activation => activation_type)";
        }
        $activation = 'tanh' unless defined $activation;
        $self->name_scope(sub {
                my $prev_size;
                my $b_input_size = 0;

                my $s_net = nn->Sequential;
                $prev_size = $state_size;
                $s_net->name_scope(sub {
                        for my $size (@$state_layers) {
                            $s_net->add(nn->Dense($size, in_units => $prev_size, activation => $activation));
                            $prev_size = $size;
                        }
                    });
                $self->s_net($s_net);
                $b_input_size += $prev_size;

                my $a_net = nn->Sequential;
                $prev_size = $action_size;
                $a_net->name_scope(sub {
                        for my $size (@$action_layers) {
                            $a_net->add(nn->Dense($size, in_units => $prev_size, activation => $activation));
                            $prev_size = $size;
                        }
                    });
                $self->a_net($a_net);
                $b_input_size += $prev_size;

                my $b_net = nn->Sequential;
                $prev_size = $b_input_size;
                $b_net->name_scope(sub {
                        for my $size (@$base_layers) {
                            $b_net->add(nn->Dense($size, in_units => $prev_size, activation => $activation));
                            $prev_size = $size;
                        }
                        $b_net->add(nn->Dense(1, in_units => $prev_size)); # linear activation for the last layer
                    });
                $self->b_net($b_net);
            });
    }

    method forward($s, $a) {
        my $s_out = $self->s_net->($s); 
        my $a_out = $self->a_net->($a);
        my $b_input = mx->nd->concat($s_out, $a_out, dim => 1);
        return $self->b_net->($b_input);
    }
}

package ActorModel {
    use Math::Trig;
    use AI::MXNet::Gluon::Mouse;
    use AI::MXNet::Function::Parameters; # must include this for function parameters
    extends 'AI::MXNet::Gluon::Block';

    sub BUILD {
        my $self = shift;
        my $attrs = shift;
        my $sizes = $attrs->{sizes};
        my $activation = $attrs->{activation};
        unless (defined($sizes) && defined($activation)) {
            die "usage: ActorModel->new(sizes => [size_1, size_2, ...], activation => activation_type)";
        }
        $activation = 'tanh' unless defined $activation;
        $self->name_scope(sub {
                my $net = nn->Sequential;
                my $prev_size = $state_size;
                $net->name_scope(sub {
                        for my $size (@$sizes) {
                            $net->add(nn->Dense($size, in_units => $prev_size, activation => $activation));
                            $prev_size = $size;
                        }
                        $net->add(nn->Dense($action_size, in_units => $prev_size));
                    });
                $self->dense_base($net);
            });
    }

    method forward($x) {
        my $y = $self->dense_base->($x);
        return $y;
    }

}

package RunningMeanStd {
    sub new {
        my ($class, $shape) = @_;
        my $self = bless {
            n => mx->nd->zeros([1]),
            mean => mx->nd->zeros($shape),
            nvar => mx->nd->zeros($shape),
            std => mx->nd->zeros($shape),
        }, $class;
    }

    sub update {
        my ($self, $x) = @_;
        ++$self->{n};
        my $m = $self->{mean}->copy;
        $self->{mean} = $m + ($x - $m) / $self->{n};
        $self->{nvar} = $self->{nvar} + ($x - $m) * ($x - $self->{mean});
        $self->{std} = sqrt($self->{nvar} / $self->{n});
    }
}

# See
#   https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
#   https://github.com/openai/baselines/blob/master/baselines/common/running_mean_std.py
package RunningMeanStdMulti {
    sub new {
        my ($class, $shape) = @_;
        my $self = bless {
            n => mx->nd->zeros([1]),
            mean => mx->nd->zeros($shape),
            nvar => mx->nd->zeros($shape),
            std => mx->nd->zeros($shape),
        }, $class;
    }

    sub update {
        my ($self, $x) = @_;
        my $nx = $x->shape->[0];
        my $n = $self->{n}->copy;
        $self->{n} = $self->{n} + $nx;
        my $m = $self->{mean}->copy;
        my $m_batch = $x->mean(axis => 0);
        my $nvar_batch = (($x - $m_batch) ** 2)->sum(axis => 0);
        my $delta = $m_batch - $m;
        $self->{mean} = $m + $delta * $nx / $self->{n};
        $self->{nvar} = $self->{nvar} + $nvar_batch + ($delta ** 2) * $n * $nx / $self->{n};
        $self->{std} = sqrt($self->{nvar} / $self->{n});
    }
}

package Normalizer {
    sub new {
        my ($class, $shape) = @_;
        my $self = bless {
            #ms => RunningMeanStd->new($shape),
            ms => RunningMeanStdMulti->new($shape),
        }, $class;
    }

    sub normalize {
        my ($self, $x, $update) = @_;
        $update = 1 unless defined $update;
        if ($update) {
            $self->{ms}->update($x);
        }
        return ($x - $self->{ms}{mean}) / ($self->{ms}{std} + 1e-8);
    }

    sub save {
        my ($self, $suffix) = @_;
        if (defined $suffix) {
            $suffix = "-$suffix";
        } else {
            $suffix = '';
        }
        mx->nd->save("$save_model/state_normalizer$suffix.nd", {
                mean => $self->{ms}{mean},
                nvar => $self->{ms}{nvar},
                std => $self->{ms}{std},
                n => $self->{ms}{n},
            });
    }

    sub load {
        my ($self) = @_;
        my $h = mx->nd->load("$load_model/state_normalizer.nd");
        $self->{ms}{mean} = $h->{mean};
        $self->{ms}{nvar} = $h->{nvar};
        $self->{ms}{std} = $h->{std};
        $self->{ms}{n} = $h->{n};
    }
}


my $gamma = 0.99;
my $clip_ratio = 0.2;
my $actor_num_epochs = 3;
my $critic_num_epochs = 3;
my $lam = 0.97;
my $target_kl = 0.01;
my $policy_learning_rate = 1e-3;
my $value_function_learning_rate = 1e-2;
my $actor_layers = [64, 64];
my $critic_state_layers = [32];
my $critic_action_layers = [32];
my $critic_base_layers = [64];

if (defined($parameters)) {
    die "Cannot find the parameter file!" unless -f $parameters;
    my $para = decode_json(read_file($parameters));
    $policy_learning_rate = $para->{policy_learning_rate} if defined $para->{policy_learning_rate};
    $value_function_learning_rate = $para->{value_function_learning_rate} if defined $para->{value_function_learning_rate};
    $actor_layers = $para->{actor_layers} if defined $para->{actor_layers};
    $critic_state_layers = $para->{critic_state_layers} if defined $para->{critic_state_layers};
    $critic_action_layers = $para->{critic_action_layers} if defined $para->{critic_action_layers};
    $critic_base_layers = $para->{critic_base_layers} if defined $para->{critic_base_layers};
    $actor_num_epochs = $para->{actor_num_epochs} if defined $para->{actor_num_epochs};
    $critic_num_epochs = $para->{critic_num_epochs} if defined $para->{critic_num_epochs};
}

my $actor_net = ActorModel->new(sizes => $actor_layers,  activation => 'relu');
#print $actor_net;
my $critic_net = CriticModel->new(state_layers => $critic_state_layers, action_layers => $critic_action_layers, base_layers => $critic_base_layers, activation => 'relu');
#print $critic_net;
if (defined($load_model)) {
    die "Cannot find the model files!" unless -d $load_model and -f "$load_model/actor.par" and -f "$load_model/critic.par";
    print "load actor from $load_model/actor.par\n";
    $actor_net->load_parameters("$load_model/actor.par");
    print "load critic from $load_model/critic.par\n";
    $critic_net->load_parameters("$load_model/critic.par");
    if ($reinit_logstd) {
        $actor_net->logstd->initialize(init => mx->init->Zero, force_reinit => 1);
    }
} else {
    $actor_net->dense_base->initialize(mx->init->Xavier());
    $critic_net->s_net->initialize(mx->init->Xavier());
    $critic_net->a_net->initialize(mx->init->Xavier());
    $critic_net->b_net->initialize(mx->init->Xavier());
}

my $state_normalizer = Normalizer->new([1, $state_size]);
if (defined($load_model)) {
    $state_normalizer->load;
}

my $policy_optimizer = gluon->Trainer(
    $actor_net->collect_params(), 'adam',
    { learning_rate => $policy_learning_rate });
my $value_optimizer = gluon->Trainer(
    $critic_net->collect_params(), 'adam',
    { learning_rate => $value_function_learning_rate });

sub train_policy {
    my ($observation_buffer) = @_;
    my $policy_loss;
    autograd->record(sub {
            my $mu = $actor_net->($observation_buffer);
            my $critic_value = $critic_net->($observation_buffer, $mu);
            $policy_loss = -mx->nd->mean($critic_value);
        });
    $policy_loss->backward;
    #print($policy_loss->aspdl);
    $policy_optimizer->step($observation_buffer->shape->[0]);

    return $policy_loss;
}

sub train_value_function {
    my ($observation_buffer, $action_buffer, $return_buffer) = @_;
    my $value_loss;
    autograd->record(sub {
            $value_loss = mx->nd->mean(($return_buffer - $critic_net->($observation_buffer, $action_buffer)) ** 2);
        });
    $value_loss->backward;
    #print($value_loss->aspdl);
    $value_optimizer->step($return_buffer->shape->[0]);
    return $value_loss;
}

for my $env (@envs) {
    $env->reset;
}

my $i_img = 0;

my $interrupt = 0;

$SIG{INT} = sub {
    print "Receive an INT signal. Wait for the current iteration.\n";
    $interrupt = 1;
};

if ($play_policy) {
    my $env = $envs[0];
    #$env->run_viewer;
    #exit;
    my ($pos, $i);
    if (defined($traj_file)) {
        my $h = mx->nd->load($traj_file);
        $pos = $h->{pos};
        $i = 0;
    }
    mkdir $outdir unless -e $outdir;
    mkdir $imgdir unless -e $imgdir;
    open my $fout, '>', "$outdir/positions.txt";
    open my $f_action, '>', "$outdir/actions.txt";
    open my $f_reward, '>', "$outdir/rewards.txt";
    my $acc_gamma = 1;
    my $test_return = 0;
    $env->reset;
    #$env->set_positions(mx->nd->array([0.5, 0]));
    until ($env->viewer_done) {
        if ($env->is_playing || $env->req_step) {
            if (defined($pos)) {
                if ($i >= $pos->shape->[0]) {
                    $i = 0;
                }
                $env->set_positions($pos->slice($i)->squeeze);
                ++$i;
            } else {
                #print($env->get_positions->aspdl, "\n");
                #print(join(' ', $env->get_positions_list), "\n");
                my $observation = mx->nd->array([[$env->get_state_list]]);
                #$observation = $state_normalizer->normalize($observation, 0);
                print $fout join(' ', $env->get_positions_list), "\n";
                #print "state: ", $observation->aspdl, "\n";
                my $mu = $actor_net->($observation);
                #my $action = $actor_net->choose_action($observation);
                my $action = $mu;   # deterministic
                #$action = $action->clip(-1, 1);
                print $f_action join(' ', $action->aspdl->list), "\n";
                #print "action: ", $action->aspdl, "\n";
                #$a_scale = 0;
                $env->set_action_list(($action * $a_scale)->aspdl->list);
                #$env->set_action_list((0) x $action_size);
                $env->step;
                my $reward = $env->get_reward;
                my $done = $env->get_done;
                $test_return += $acc_gamma * $reward;
                $acc_gamma *= $gamma;
                print $f_reward "$reward $test_return $done\n";
                #print "(", ($env->get_state_list)[0], ") ";
                #print "$reward ";
                #my $done = $reward < 10;
                #last if $done;
            }
        }
        $env->render_viewer;
    }
    exit();
}

#open my $f_action, '>', "$outdir/action.txt";
#open my $f_pos, '>', "$outdir/position.txt";
open my $f_ret, '>', "$save_model/return_length.txt";

my $best_return = '-inf';
my $total_num_samples = 0;

for my $itr (1 .. $num_itrs) {
    $g_sigma = $sigma_begin * ($num_itrs - $itr + 1) / ($num_itrs - 1) + $sigma_end * ($itr - 1) / ($num_itrs - 1);
    $omp_env->reset;
    my ($sum_return, $sum_length, $num_episodes) = (0, 0, 0);
    my ($max_return, $max_length) = ('-inf', '-inf');
    my $best_traj;
    my (@observation_buffers, @action_buffers, @return_buffers);
    my $buffer_size = 0;

    my $prev_time = time;
    #for my $i (1 .. 100) {
    my $i = 1;
    while (1) {
        #print "i: $i\n";
        ++$i;
        my $observations = $omp_env->get_observations;
        #print "o:\n", $observations->aspdl, "\n";
        #print "o->shape: ", join('x', @{$observations->shape}), "\n";
        #last if $observations->shape->[0] == 0 or $i > 500;
        if ($observations->shape->[0] == 0) {
            $omp_env->trace_back;
            $sum_return += $omp_env->get_avg_ret;
            $sum_length += $omp_env->get_max_len;
            $num_episodes += 1;
            #$max_return = $omp_env->get_max_ret if $omp_env->get_max_ret > $max_return;
            if ($omp_env->get_max_ret > $max_return) {
                $max_return = $omp_env->get_max_ret;
                $best_traj = $omp_env->get_best_traj;
            }
            $max_length = $omp_env->get_max_len if $omp_env->get_max_len > $max_length;
            #print $omp_env->get_ret_buffer->aspdl, "\n";
            #exit;
            push @observation_buffers, $omp_env->get_obs_buffer;
            push @action_buffers, $omp_env->get_act_buffer;
            push @return_buffers, $omp_env->get_ret_buffer;
            $buffer_size += $omp_env->get_buffer_size;
            $omp_env->reset;
            $i = 1;
            next;
        }
        if ($buffer_size + $omp_env->get_buffer_size >= $steps_per_itr) {
            $omp_env->trace_back;
            $sum_return += $omp_env->get_avg_ret;
            $sum_length += $omp_env->get_max_len;
            $num_episodes += 1;
            #$max_return = $omp_env->get_max_ret if $omp_env->get_max_ret > $max_return;
            if ($omp_env->get_max_ret > $max_return) {
                $max_return = $omp_env->get_max_ret;
                $best_traj = $omp_env->get_best_traj;
            }
            $max_length = $omp_env->get_max_len if $omp_env->get_max_len > $max_length;
            #print $omp_env->get_ret_buffer->aspdl, "\n";
            #exit;
            push @observation_buffers, $omp_env->get_obs_buffer;
            push @action_buffers, $omp_env->get_act_buffer;
            push @return_buffers, $omp_env->get_ret_buffer;
            $buffer_size += $omp_env->get_buffer_size;
            last;
        }
        #$observations = $state_normalizer->normalize($observations);
        my $means = $actor_net->($observations);
        my $stds = mx->nd->ones($means->shape) * $g_sigma;
        #print "mean:\n", $means->aspdl, "\n";
        #print "std:\n", $stds->aspdl, "\n";
        $omp_env->set_means($means);
        $omp_env->set_stds($stds);
        my $values = $critic_net->($observations, $means);
        $omp_env->set_values($values);
        $omp_env->step;
    }
    #$omp_env->trace_back;
    #my $avg_ret = $omp_env->get_avg_ret;
    #my $max_ret = $omp_env->get_max_ret;

    my $sim_time = time - $prev_time;
    $prev_time = time;

    #print $omp_env->get_obs_buffer->aspdl;
    #print $omp_env->get_act_buffer->aspdl;
    #print $omp_env->get_adv_buffer->aspdl;
    #print $omp_env->get_ret_buffer->aspdl;
    #print $omp_env->get_logp_buffer->aspdl;
    #my $all_observation_buffer = $omp_env->get_obs_buffer;
    #my $all_action_buffer = $omp_env->get_act_buffer;
    #my $all_return_buffer = $omp_env->get_ret_buffer;
    my $all_observation_buffer = mx->nd->concat(@observation_buffers, dim => 0);
    my $all_action_buffer = mx->nd->concat(@action_buffers, dim => 0);
    my $all_return_buffer = mx->nd->concat(@return_buffers, dim => 0);
    my $batch_size = $all_return_buffer->shape->[0];
    print "batch_size: $batch_size\n";
    #print "obs buffer size: ", $all_observation_buffer->shape->[0], ". ";
    #print "act buffer size: ", $all_action_buffer->shape->[0], ". ";
    #print "ret buffer size: ", $all_return_buffer->shape->[0], "\n";
    $total_num_samples += $batch_size;
    $num_train_itrs = int($batch_size / $mini_batch_size);

    my $indices = pdl(shuffle(0 .. $batch_size-1));
    $all_observation_buffer = reorder($all_observation_buffer, $indices);
    $all_action_buffer = reorder($all_action_buffer, $indices);
    $all_return_buffer = reorder($all_return_buffer, $indices);

    my $decay_frac = 1.0 / (1 + $decay_factor * ($itr - 1));
    $policy_optimizer->set_learning_rate($policy_learning_rate * $decay_frac);
    $value_optimizer->set_learning_rate($value_function_learning_rate * $decay_frac);
    #print $policy_optimizer->_optimizer->lr, " ", $value_optimizer->_optimizer->lr, "\n";

    my $loss_sum = 0;
    my $itrs_sum = 0;

    my $value_loss;
    for my $epoch (1 .. $critic_num_epochs) {
        for my $i (0 .. $num_train_itrs - 1) {
            my $a = $i * $mini_batch_size;
            my $b = $a + $mini_batch_size - 1;
            $b = $batch_size - 1 if $b >= $batch_size;
            #print "mini batch $a $b\n";
            $value_loss = train_value_function($all_observation_buffer->slice([$a, $b]), $all_action_buffer->slice([$a, $b]), $all_return_buffer->slice([$a, $b]));
            $loss_sum += $value_loss;
            ++$itrs_sum;
        }
    }
    $value_loss = $loss_sum / $itrs_sum;

    $loss_sum = 0;
    $itrs_sum = 0;

    my $policy_loss;
    for my $epoch (1 .. $actor_num_epochs) {
        for my $i (0 .. $num_train_itrs - 1) {
            my $a = $i * $mini_batch_size;
            my $b = $a + $mini_batch_size - 1;
            $b = $batch_size - 1 if $b >= $batch_size;
            #print "mini batch $a $b\n";
            $policy_loss = train_policy($all_observation_buffer->slice([$a, $b]));
            $loss_sum += $policy_loss;
            ++$itrs_sum;
        }
    }
    $policy_loss = $loss_sum / $itrs_sum;

    my $train_time = time - $prev_time;
    $prev_time = time;

    my $test_return = 0;
    my $test_length = 0;
    my $acc_gamma = 1;
    my $env = $envs[0];
    $env->reset;
    my $observation = mx->nd->array([[$env->get_state_list]]);
    #$observation = $state_normalizer->normalize($observation, 0);
    until ($env->get_done || $test_length >= $steps_per_itr) {
        my $mu = $actor_net->($observation);
        my $action = $mu;   # deterministic
        #$action = $action->clip(-1, 1);
        $env->set_action_list(($action * $a_scale)->aspdl->list);
        $env->step;
        my $reward = $env->get_reward;
        $test_return += $acc_gamma * $reward;
        $acc_gamma *= $gamma;
        ++$test_length;
        $observation = mx->nd->array([[$env->get_state_list]]);
        #$observation = $state_normalizer->normalize($observation, 0);
    }
    my $test_time = time - $prev_time;

    print "$max_return $max_length\n";
    print "$sum_return $num_episodes\n";
    #print "Itr: $itr. Sigma: $g_sigma. Mean Return: $avg_ret. Max Return: $max_ret. Test Return: $test_return. Test Length: $test_length. Policy Loss: ", $policy_loss->aspdl->sclr, ". Value Loss: ", $value_loss->aspdl->sclr, ". Time: $sim_time, $train_time\n";
    print "Itr: $itr. Sigma: $g_sigma. Mean Return: ", $sum_return / $num_episodes, ". Mean Length: ", $sum_length / $num_episodes, ". Test Return: $test_return. Test Length: $test_length. Policy Loss: ", $policy_loss->aspdl->sclr, ". Value Loss: ", $value_loss->aspdl->sclr, ". Time: $sim_time, $train_time, $test_time\n";
    #print $f_ret $avg_ret, " ", $max_ret, " $test_return $test_length\n";
    print $f_ret $total_num_samples, " ", $sum_return / $num_episodes, " ", $sum_length / $num_episodes, " $test_return $test_length $max_return $max_length\n";

    mx->nd->save(sprintf("$save_model/traj-%06d.nd", $itr), { pos => $best_traj });

    if ($itr % $save_interval == 0) {
        $actor_net->save_parameters(sprintf("$save_model/actor-%06d.par", $itr));
        $critic_net->save_parameters(sprintf("$save_model/critic-%06d.par", $itr));
        $state_normalizer->save(sprintf("%06d", $itr));
    }

    if ($best_return < $test_return) {
        $best_return = $test_return;
        $actor_net->save_parameters("$save_model/actor-best.par");
        $critic_net->save_parameters("$save_model/critic-best.par");
        $state_normalizer->save("best");
    }

    last if $interrupt;
}

$actor_net->save_parameters("$save_model/actor.par");
$critic_net->save_parameters("$save_model/critic.par");
$state_normalizer->save;
