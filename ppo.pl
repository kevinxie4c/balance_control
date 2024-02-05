#!/usr/bin/env perl
use AI::MXNet qw(mx);
use AI::MXNet::Gluon::NN qw(nn);
use AI::MXNet::Gluon qw(gluon);
use AI::MXNet::AutoGrad qw(autograd);
use Getopt::Long qw(:config no_ignore_case);
use Time::HiRes qw(time);
use List::Util qw(shuffle);
use strict;
use warnings;

my $use_gpu = undef;
my $load_model = undef;
my $save_model = 'model';
my $play_policy = 0;
my $save_interval = 200;
my $g_sigma = 0;
my $outdir = "output";
my $num_threads = 4;
my $sigma_begin = 0.1;
my $sigma_end = 0.01;
my $steps_per_itr = 1024;
my $num_itrs = 5000;
my $mini_batch_size = 256;
my $a_scale = 1;
my $config_file = undef;

GetOptions(
    'G|gpu:i'              => \$use_gpu,
    'l|load_model=s'       => \$load_model,
    's|save_model=s'       => \$save_model,
    'p|play_policy'        => \$play_policy,
    'i|save_interval=i'    => \$save_interval,
    'n|num_threads=i'      => \$num_threads,
    'm|mini_batch_size=i'  => \$mini_batch_size,
    'B|sigma_begin=f'      => \$sigma_begin,
    'E|sigma_end=f'        => \$sigma_end,
    'N|num_itrs=i'         => \$num_itrs,
    'K|steps_per_itr=i'    => \$steps_per_itr,
    'a|a_scale=f'          => \$a_scale,
);

unless (@ARGV == 1) {
    die "usage: $0 [options] config_file\n";
}

$config_file = shift @ARGV;

my $num_train_itrs = $steps_per_itr / $mini_batch_size;
if (int($num_train_itrs) != $num_train_itrs) {
    warn "please choose a better mini_batch_size";
    $num_train_itrs = int($num_train_itrs);
}

mkdir $outdir unless -e $outdir;
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

    #sub concat {
    #    my $class = shift;
    #    my $buffer = bless {
    #        observation_buffer    => nd->concat(map({ $_->{observation_buffer} }    @_), dim => 0),
    #        action_buffer         => nd->concat(map({ $_->{action_buffer} }         @_), dim => 0),
    #        advantage_buffer      => nd->concat(map({ $_->{advantage_buffer} }      @_), dim => 0),
    #        reward_buffer         => nd->concat(map({ $_->{reward_buffer} }         @_), dim => 0),
    #        return_buffer         => nd->concat(map({ $_->{return_buffer} }         @_), dim => 0),
    #        value_buffer          => nd->concat(map({ $_->{value_buffer} }          @_), dim => 0),
    #        logprobability_buffer => nd->concat(map({ $_->{logprobability_buffer} } @_), dim => 0),
    #    }, $class;
    #}
}

use Math::Trig ();
use GD;
use File::Slurp;
use PDL;
use FindBin;
use blib "$FindBin::Bin/CharacterEnv/blib";
use CharacterEnv;

my $para_env = ParallelEnv->new($config_file, $num_threads);
my @envs = $para_env->get_env_list;
#my $unit_size = 512;
my $state_size = $envs[0]->get_state_size;
my $action_size = $envs[0]->get_action_size;
print "state_size: $state_size\n";
print "action_size: $action_size\n";
$envs[0]->print_info;

sub mlp {
    my ($sizes, $activation) = @_;
    $activation = 'tanh' unless defined $activation;
    my $net = nn->Sequential;
    $net->name_scope(sub {
            my $i = 0;
            for my $size (@$sizes) {
                if ($i < $#$sizes) {
                    $net->add(nn->Dense($size, activation => $activation));
                } else {
                    $net->add(nn->Dense($size)); # linear activation for the last layer
                }
                ++$i;
            }
        });
    return $net;
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
                    });
                $self->dense_base($net);
                #$self->dense_mu(nn->Dense($action_size, in_units => $prev_size, activation => 'tanh'));
                $self->dense_mu(nn->Dense($action_size, in_units => $prev_size));
                #$self->dense_sigma(nn->Dense($action_size, in_units => $prev_size, activation => 'softrelu'));
            });
    }

    method forward($x) {
        my $y = $self->dense_base->($x);
        #my ($mu, $sigma) = ($self->dense_mu->($y), $self->dense_sigma->($y));
        my $mu = $self->dense_mu->($y);
        my $sigma = mx->nd->ones($mu->shape) * $g_sigma;
        return ($mu, $sigma);
    }

    method choose_action($x) {
        my ($mu, $sigma) = $self->($x);
        return $self->sample($mu, $sigma);
    }

    method sample($mu, $sigma) {
        my $eps = mx->nd->random_normal(0, 1, $mu->shape, ctx => $current_ctx);
        return $mu + $sigma * $eps;
    }

    method log_prob($x, $mu, $sigma) {
        return mx->nd->sum(-0.5 * log(2.0 * pi) - $sigma->add(1e-8)->log() - ($x - $mu) ** 2 / (2 * $sigma ** 2 + 1e-8), axis => 1);
    }
}

package RunningMeanStd {
    sub new {
        my ($class, $shape) = @_;
        my $self = bless {
            n => 0,
            mean => mx->nd->zeros($shape),
            nvar => mx->nd->zeros($shape),
            std => mx->nd->zeros($shape),
        }, $class;
    }

    sub update {
        my ($self, $x) = @_;
        ++$self->{n};
        my $m = $self->{mean}->copy;
        $self->{mean} =  $m + ($x - $m) / $self->{n};
        $self->{nvar} = $self->{nvar} + ($x - $m) * ($x - $self->{mean});
        $self->{std} = sqrt($self->{nvar} / $self->{n});
    }
}

package Normalizer {
    sub new {
        my ($class, $shape) = @_;
        my $self = bless {
            ms => RunningMeanStd->new($shape),
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
            });
    }

    sub load {
        my ($self) = @_;
        my $h = mx->nd->load("$load_model/state_normalizer.nd");
        $self->{ms}{mean} = $h->{mean};
        $self->{ms}{nvar} = $h->{nvar};
        $self->{ms}{std} = $h->{std};
    }
}


my $gamma = 0.99;
my $clip_ratio = 0.2;
my $num_epochs = 3;
my $lam = 0.97;
my $target_kl = 0.01;
my $policy_learning_rate = 5e-4;
my $value_function_learning_rate = 5e-3;
my $decay_factor = 0.001;
my $actor_net = ActorModel->new(sizes => [64, 64],  activation => 'relu');
#print $actor_net;
my $critic_net = mlp([64, 64, 1], 'relu');
#print $critic_net;
if (defined($load_model)) {
    die "Canno find the model files!" unless -d $load_model and -f "$load_model/actor.par" and -f "$load_model/critic.par";
    print "load actor from $load_model/actor.par\n";
    $actor_net->load_parameters("$load_model/actor.par");
    print "load critic from $load_model/critic.par\n";
    $critic_net->load_parameters("$load_model/critic.par");
} else {
    $actor_net->dense_base->initialize(mx->init->Xavier());
    $actor_net->dense_mu->initialize(mx->init->Normal(0.01));
    #$actor_net->dense_sigma->initialize(mx->init->Zero);
    $critic_net->initialize(mx->init->Xavier());
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
    my ($observation_buffer, $action_buffer, $logprobability_buffer, $advantage_buffer) = @_;
    my $policy_loss;
    my ($mu, $sigma);
    autograd->record(sub {
            ($mu, $sigma) = $actor_net->($observation_buffer);
            my $ratio = mx->nd->exp($actor_net->log_prob($action_buffer, $mu, $sigma) - $logprobability_buffer);
            my $min_advantage = mx->nd->where($advantage_buffer > 0,
                (1 + $clip_ratio) * $advantage_buffer,
                (1 - $clip_ratio) * $advantage_buffer,
            );
            $policy_loss = -mx->nd->mean(mx->nd->broadcast_minimum($ratio * $advantage_buffer, $min_advantage));
        });
    $policy_loss->backward;
    #print($policy_loss->aspdl);
    $policy_optimizer->step($observation_buffer->shape->[0]);

    my $kl = mx->nd->mean($logprobability_buffer - $actor_net->log_prob($action_buffer, $mu, $sigma));
    #$kl = nd->sum($kl);	# Do we need this?
    return ($policy_loss, $kl);
}

sub train_value_function {
    my ($observation_buffer, $return_buffer) = @_;
    my $value_loss;
    autograd->record(sub {
            $value_loss = mx->nd->mean(($return_buffer - $critic_net->($observation_buffer)) ** 2);
        });
    $value_loss->backward;
    #print($value_loss->aspdl);
    $value_optimizer->step($return_buffer->shape->[0]);
    return $value_loss;
}

#my $buffer = Buffer->new($state_size, $action_size, $steps_per_itr);
my @buffers;
for (1 .. $num_threads) {
    push(@buffers, Buffer->new($state_size, $action_size, $steps_per_itr, $gamma, $lam));
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
    open my $fout, '>', "$outdir/positions.txt";
    open my $f_action, '>', "$outdir/actions.txt";
    open my $f_reward, '>', "$outdir/rewards.txt";
    my $acc_gamma = 1;
    my $test_return = 0;
    $env->reset;
    my $observation = mx->nd->array([[$env->get_state_list]]);
    $observation = $state_normalizer->normalize($observation, 0);
    until ($env->viewer_done) {
        print $fout join(' ', $env->get_positions_list), "\n";
        #print "state: ", $observation->aspdl, "\n";
        my ($mu, $sigma) = $actor_net->($observation);
        #my $action = $actor_net->choose_action($observation);
        my $action = $mu;   # deterministic
        $action = $action->clip(-1, 1);
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
        $observation = mx->nd->array([[$env->get_state_list]]);
        $observation = $state_normalizer->normalize($observation, 0);
        $env->render_viewer;
    }
    exit();
}

#open my $f_action, '>', "$outdir/action.txt";
#open my $f_pos, '>', "$outdir/position.txt";
open my $f_ret, '>', "$outdir/return_length.txt";

my $best_return = '-inf';

for my $itr (1 .. $num_itrs) {
    $g_sigma = $sigma_begin * ($num_itrs - $itr + 1) / ($num_itrs - 1) + $sigma_end * ($itr - 1) / ($num_itrs - 1);
    $para_env->reset;
    my ($sum_return, $sum_length, $num_episodes) = (0, 0, 0);
    #my ($episode_return, $episode_length) = (0, 0);
    #my $observation = mx->nd->array([[$env->get_state_list]]);
    #$observation = $state_normalizer->normalize($observation, 0);

    #for my $t (1 .. $steps_per_itr) {
    #    my ($mu, $sigma) = $actor_net->($observation);
    #    my $action = $actor_net->choose_action($observation);
    #    #$action = mx->nd->array([[0, 0]]); # debug
    #    #print $f_action join(' ', $action->aspdl->list), "\n";	# debug
    #    #print $f_pos join(' ', $env->get_state_list), "\n";	# debug
    #    #render;    # debug
    #    $action = $action->clip(-1, 1);
    #    $env->set_action_list(($action * $a_scale)->aspdl->list);
    #    $env->step;
    #    my $reward = $env->get_reward;
    #    #print(($env->get_state_list)[4], "\t", $reward, "\n");  # debug
    #    #print(join(" ", $env->get_state_list), "\n");   # debug
    #    my $done = $reward < 10;
    #    #$done = 0; # debug
    #    my $observation_new = mx->nd->array([[$env->get_state_list]]);
    #    $episode_return += $reward;
    #    $episode_length += 1;

    #    my $value_t = $critic_net->($observation)->aspdl->at(0, 0);
    #    my $logprobability_t = $actor_net->log_prob($action, $mu, $sigma)->aspdl->at(0, 0);

    #    $buffer->store($observation, $action, $reward, $value_t, $logprobability_t);
    #    $observation = $observation_new;

    #    if ($done || $t == $steps_per_itr) {
    #        #exit;  # debug
    #        my $last_value = $value_t;
    #        $buffer->finish_trajectory($last_value);
    #        $sum_return += $episode_return;
    #        $sum_length += $episode_length;
    #        $num_episodes += 1;
    #        $env->reset;
    #        ($episode_return, $episode_length) = (0, 0);
    #        $observation = mx->nd->array([[$env->get_state_list]]);
    #    }
    #}

    my @finished = (1) x $num_threads;
    my (@observation, @action, @logprobability_t);
    my $total_steps = 0;
    my @steps = (0) x $num_threads;
    my @acc_gamma_list = (1) x $num_threads;
    my $prev_time = time;
    while ($total_steps < $steps_per_itr) {
        my $id = $para_env->get_task_done_id;
        my $env = $envs[$id];

        if (defined($observation[$id])) {
            my $reward = $env->get_reward;
            my $done = $env->get_done;
            if (abs($reward) < 1e-10) {
                $reward = 0;
            }
            #print "$reward\n";
            if ($reward eq 'NaN') {
                print "reward: NaN\n";
                $reward = -100;
                $done = 1;
            }
            $sum_return += $acc_gamma_list[$id] * $reward;
            $acc_gamma_list[$id] *= $gamma;
            $sum_length += 1;

            my $value_t = $critic_net->($observation[$id])->aspdl->at(0, 0);

            $buffers[$id]->store($observation[$id], $action[$id], $reward, $value_t, $logprobability_t[$id]);

            if ($done) {
                my $last_value = $value_t;
                $buffers[$id]->finish_trajectory($last_value);
                $num_episodes += 1;
                $env->reset;
                $observation[$id] = undef;
                $acc_gamma_list[$id] = 1;
            }
        }

        my @state_list = $env->get_state_list;
        if (grep($_ eq 'NaN', @state_list) > 0) {
            print "state: NaN\n";
            @state_list = (-1) x @state_list;
        }
        $observation[$id] = mx->nd->array([[@state_list]]);
        $observation[$id] = $state_normalizer->normalize($observation[$id]);
        my ($mu, $sigma) = $actor_net->($observation[$id]);
        $action[$id] = $actor_net->sample($mu, $sigma);
        if ($action[$id]->aspdl =~ /NaN/) {
            print "action: NaN\n";
        }
        $logprobability_t[$id] = $actor_net->log_prob($action[$id], $mu, $sigma)->aspdl->at(0, 0);
        #$action[$id] = $action[$id]->clip(-1, 1);
        $env->set_action_list(($action[$id]->clip(-1, 1) * $a_scale)->aspdl->list);
        $para_env->step($id);
        $finished[$id] = 0;
        ++$total_steps;
        ++$steps[$id];
    }
    print "steps: @steps\n";

    my $all_finished = 0;
    while (!$all_finished) {
        my $id = $para_env->get_task_done_id;
        $finished[$id] = 1;
        my $env = $envs[$id];

        if (defined($observation[$id])) {
            my $reward = $env->get_reward;
            if (abs($reward) < 1e-10) {
                $reward = 0;
            }
            $reward = -10 if $reward eq 'NaN';
            $sum_return += $acc_gamma_list[$id] * $reward;
            $sum_length += 1;

            my $value_t = $critic_net->($observation[$id])->aspdl->at(0, 0);

            $buffers[$id]->store($observation[$id], $action[$id], $reward, $value_t, $logprobability_t[$id]);

            my $last_value = $value_t;
            $buffers[$id]->finish_trajectory($last_value);
            $num_episodes += 1;
            $env->reset;
            $observation[$id] = undef;
            $acc_gamma_list[$id] = 1;
        }

        $all_finished = 1;
        for (@finished) {
            $all_finished = $all_finished && $_;
        }
    }

    my $sim_time = time - $prev_time;
    $prev_time = time;
    my (@observation_buffers, @action_buffers, @advantage_buffers, @return_buffers, @logprobability_buffers);
    for my $id (0 .. $num_threads - 1) {
        my ($observation_buffer, $action_buffer, $advantage_buffer, $return_buffer, $logprobability_buffer) = $buffers[$id]->get;
        push @observation_buffers, $observation_buffer;
        push @action_buffers, $action_buffer;
        push @advantage_buffers, $advantage_buffer;
        push @return_buffers, $return_buffer;
        push @logprobability_buffers, $logprobability_buffer;
    }
    my $all_observation_buffer = mx->nd->concat(@observation_buffers, dim => 0);
    my $all_action_buffer = mx->nd->concat(@action_buffers, dim => 0);
    my $all_advantage_buffer = mx->nd->concat(@advantage_buffers, dim => 0);
    my $all_return_buffer = mx->nd->concat(@return_buffers, dim => 0);
    my $all_logprobability_buffer = mx->nd->concat(@logprobability_buffers, dim => 0);
    die "wrong size" if $all_observation_buffer->shape->[0] != $steps_per_itr;

    my $indices = pdl(shuffle(0 .. $steps_per_itr-1));
    $all_observation_buffer = reorder($all_observation_buffer, $indices);
    $all_action_buffer = reorder($all_action_buffer, $indices);
    $all_advantage_buffer = reorder($all_advantage_buffer, $indices);
    $all_return_buffer = reorder($all_return_buffer, $indices);
    $all_logprobability_buffer = reorder($all_logprobability_buffer, $indices);

    $policy_optimizer->set_learning_rate($policy_learning_rate / (1 + $decay_factor * ($itr - 1)));
    $value_optimizer->set_learning_rate($value_function_learning_rate / (1 + $decay_factor * ($itr - 1)));
    #print $policy_optimizer->_optimizer->lr, " ", $value_optimizer->_optimizer->lr, "\n";

    my $loss_sum = 0;
    my $itrs_sum = 0;
    my ($policy_loss, $kl);
POLICY_LOOP:
    for my $epoch (1 .. $num_epochs) {
        for my $i (0 .. $num_train_itrs - 1) {
            my $a = $i * $mini_batch_size;
            my $b = $a + $mini_batch_size - 1;
            $b = $steps_per_itr - 1 if $b >= $steps_per_itr;
            ($policy_loss, $kl) = train_policy($all_observation_buffer->slice([$a, $b]), $all_action_buffer->slice([$a, $b]), $all_logprobability_buffer->slice([$a, $b]), $all_advantage_buffer->slice([$a, $b]));
            my ($m, $s) = $actor_net->(mx->nd->ones([1, $state_size]));
            if ($m =~ /NaN/ || $s =~ /NaN/) {
                print "NaN\n";
            }
            $loss_sum += $policy_loss;
            ++$itrs_sum;
            if ($kl->aspdl->sclr > 1.5 * $target_kl) {
                last POLICY_LOOP;
            }
        }
    }
    $policy_loss = $loss_sum / $itrs_sum;

    $loss_sum = 0;
    $itrs_sum = 0;
    my $value_loss;
    for my $epoch (1 .. $num_epochs) {
        for my $i (0 .. $num_train_itrs - 1) {
            my $a = $i * $mini_batch_size;
            my $b = $a + $mini_batch_size - 1;
            $b = $steps_per_itr - 1 if $b >= $steps_per_itr;
            $value_loss = train_value_function($all_observation_buffer->slice([$a, $b]), $all_return_buffer->slice([$a, $b]));
            $loss_sum += $value_loss;
            ++$itrs_sum;
        }
    }
    $value_loss = $loss_sum / $itrs_sum;
    my $train_time = time - $prev_time;

    my $test_return = 0;
    my $test_length = 0;
    my $acc_gamma = 1;
    my $env = $envs[0];
    $env->reset;
    my $observation = mx->nd->array([[$env->get_state_list]]);
    $observation = $state_normalizer->normalize($observation, 0);
    until ($env->get_done || $test_length >= $steps_per_itr) {
        my ($mu, $sigma) = $actor_net->($observation);
        my $action = $mu;   # deterministic
        $action = $action->clip(-1, 1);
        $env->set_action_list(($action * $a_scale)->aspdl->list);
        $env->step;
        my $reward = $env->get_reward;
        $test_return += $acc_gamma * $reward;
        $acc_gamma *= $gamma;
        ++$test_length;
        $observation = mx->nd->array([[$env->get_state_list]]);
        $observation = $state_normalizer->normalize($observation, 0);
    }

    print "Itr: $itr. Sigma: $g_sigma. Mean Return: ", $sum_return / $num_episodes, ". Mean Length: ", $sum_length / $num_episodes, ". Test Return: $test_return. Test Length: $test_length. Policy Loss: ", $policy_loss->aspdl->sclr, ". Value Loss: ", $value_loss->aspdl->sclr, ". Time: $sim_time, $train_time\n";
    print $f_ret $sum_return / $num_episodes, " ", $sum_length / $num_episodes, " $test_return $test_length\n";

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
