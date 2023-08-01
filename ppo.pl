#!/usr/bin/env perl
use AI::MXNet qw(mx);
use AI::MXNet::Gluon::NN qw(nn);
use AI::MXNet::Gluon qw(gluon);
use AI::MXNet::AutoGrad qw(autograd);
use Getopt::Long qw(:config no_ignore_case);
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
my $steps_per_epoch = 200;
my $num_epochs = 5000;
my $a_scale = 1;

GetOptions(
    'G|gpu:i'             => \$use_gpu,
    'l|load_model=s'      => \$load_model,
    's|save_model=s'      => \$save_model,
    'p|play_policy'       => \$play_policy,
    'i|save_interval=i'   => \$save_interval,
    'n|num_threads=i'     => \$num_threads,
    'B|sigma_begin=f'     => \$sigma_begin,
    'E|sigma_end=f'       => \$sigma_end,
    'N|num_epochs=i'      => \$num_epochs,
    'K|steps_per_epoch=i' => \$steps_per_epoch,
    'a|a_scale=f'         => \$a_scale,
);

mkdir $outdir unless -e $outdir;
my $current_ctx = defined($use_gpu) ? mx->gpu($use_gpu) : mx->cpu;
AI::MXNet::Context->set_current($current_ctx);

my $test_tensor = nd->zeros([2, 2]);
print "test tensor: $test_tensor\n";

$num_threads = 1 if $play_policy;

sub discounted_cumulative_sums {
    my ($x, $discount) = @_;
    my $n = $x->size;
    my $y = nd->zeros([$n]);
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

# discounted_cumulative_sums test
#my $x = nd->ones([5]);
#my $y = discounted_cumulative_sums($x, 0.5);
#print $y->aspdl;

package Buffer {
    sub new {
	my ($class, $ob_dim, $ac_dim, $size, $gamma, $lam) = @_;
	$gamma = 0.99 unless defined($gamma);
	$lam = 0.95 unless defined($lam);
	my $self = bless {
	    observation_buffer     => nd->zeros([$size, $ob_dim]),
	    action_buffer          => nd->zeros([$size, $ac_dim]),
	    advantage_buffer       => nd->zeros([$size]),
	    reward_buffer          => nd->zeros([$size]),
	    return_buffer          => nd->zeros([$size]),
	    value_buffer           => nd->zeros([$size]),
	    logprobability_buffer  => nd->zeros([$size]),
	    gamma => $gamma,
	    lam => $lam,
	    pointer => 0,
	    trajectory_start_index => 0,
	}, $class;
    }

    sub store {
	my ($self, $observation, $action, $reward, $value, $logprobability) = @_;
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
	my $rewards = nd->concatenate([$self->{reward_buffer}->slice([$a, $b]), nd->array([$last_value])]);
	my $values = nd->concatenate([$self->{value_buffer}->slice([$a, $b]), nd->array([$last_value])]);
	my $deltas = $rewards->slice([0,-2]) + $self->{gamma} * $values->slice([1,-1]) - $values->slice([0,-2]);

	$self->{advantage_buffer}->slice([$a, $b]) .= main::discounted_cumulative_sums($deltas, $self->{gamma} * $self->{lam});
	$self->{return_buffer}->slice([$a, $b]) .= main::discounted_cumulative_sums($rewards, $self->{gamma})->slice([0,-2]);

	$self->{trajectory_start_index} = $self->{pointer};
    }

    sub get {
	my $self = shift;
	($self->{pointer}, $self->{trajectory_start_index}) = (0, 0);
	my $advantage_mean = $self->{advantage_buffer}->mean->aspdl->at(0);
	my $advantage_std = main::nd_std($self->{advantage_buffer})->aspdl->at(0);
	$self->{advantage_buffer} = ($self->{advantage_buffer} - $advantage_mean) / $advantage_std;
	return ($self->{observation_buffer}, $self->{action_buffer}, $self->{advantage_buffer}, $self->{return_buffer}, $self->{logprobability_buffer});
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

my $para_env = ParallelEnv->new('data/env_config.json', $num_threads);
my @envs = $para_env->get_env_list;
#my $unit_size = 512;
my $state_size = $envs[0]->get_state_size;
my $action_size = $envs[0]->get_action_size;

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
		$self->dense_mu(nn->Dense($action_size, in_units => $prev_size, activation => 'tanh'));
		#$self->dense_sigma(nn->Dense($action_size, in_units => $prev_size, activation => 'softrelu'));
	    });
    }

    method forward($x) {
	my $y = $self->dense_base->($x);
	#my ($mu, $sigma) = ($self->dense_mu->($y), $self->dense_sigma->($y));
	my $mu = $self->dense_mu->($y);
	my $sigma = nd->ones($mu->shape) * $g_sigma;
	return ($mu, $sigma);
    }

    method choose_action($x) {
	my ($mu, $sigma) = $self->($x);
	return $self->sample($mu, $sigma);
    }

    method sample($mu, $sigma) {
	my $eps = nd->random_normal(0, 1, $mu->shape, ctx => $current_ctx);
	return $mu + $sigma * $eps;
    }
    
    method log_prob($x, $mu, $sigma) {
	return nd->sum(-0.5 * log(2.0 * pi) - $sigma->add(1e-8)->log() - ($x - $mu) ** 2 / (2 * $sigma ** 2 + 1e-8), axis => 1);
    }
}


my $gamma = 0.99;
my $clip_ratio = 0.2;
my $train_policy_iterations = 80;
my $train_value_iterations = 80;
my $lam = 0.97;
my $target_kl = 0.01;
my $policy_learning_rate = 3e-4;
my $value_function_learning_rate = 1e-3;
my $actor_net = ActorModel->new(sizes => [1024, 512],  activation => 'relu');
#print $actor_net;
my $critic_net = mlp([1024, 512, 1], 'relu');
#print $critic_net;
if (defined($load_model)) {
    die "Canno find the model files!" unless -d $load_model and -f "$load_model/actor.par" and -f "$load_model/critic.par";
    print "load actor from $load_model/actor.par\n";
    $actor_net->load_parameters("$load_model/actor.par");
    print "load critic from $load_model/critic.par\n";
    $critic_net->load_parameters("$load_model/critic.par");
} else {
    $actor_net->initialize(mx->init->Xavier());
    $critic_net->initialize(mx->init->Xavier());
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
	    my $ratio = nd->exp($actor_net->log_prob($action_buffer, $mu, $sigma) - $logprobability_buffer);
	    my $min_advantage = nd->where($advantage_buffer > 0,
	        (1 + $clip_ratio) * $advantage_buffer,
	        (1 - $clip_ratio) * $advantage_buffer,
	    );
	    $policy_loss = -nd->mean(nd->broadcast_minimum($ratio * $advantage_buffer, $min_advantage));
	});
    $policy_loss->backward;
    #print($policy_loss->aspdl);
    $policy_optimizer->step($observation_buffer->shape->[0]);
    
    my $kl = nd->mean($logprobability_buffer - $actor_net->log_prob($action_buffer, $mu, $sigma));
    #$kl = nd->sum($kl);	# Do we need this?
    return $kl;
}

sub train_value_function {
    my ($observation_buffer, $return_buffer) = @_;
    my $value_loss;
    autograd->record(sub {
	    $value_loss = nd->mean(($return_buffer - $critic_net->($observation_buffer)) ** 2);
	});
    $value_loss->backward;
    #print($value_loss->aspdl);
    $value_optimizer->step($return_buffer->shape->[0]);
}

#my $buffer = Buffer->new($state_size, $action_size, $steps_per_epoch);
my @buffers;
for (1 .. $num_threads) {
    push(@buffers, Buffer->new($state_size, $action_size, $steps_per_epoch));
}

for my $env (@envs) {
    $env->reset;
}

my $i_img = 0;

if (defined($save_model)) {
    mkdir $save_model unless -e $save_model;
} else {
    die "undefined save_model";
}

my $interrupt = 0;

$SIG{INT} = sub {
    print "Receive an INT signal. Wait for the current iteration.\n";
    $interrupt = 1;
};

if ($play_policy) {
    my $env = $envs[0];
    open my $fout, '>', "$outdir/positions.txt";
    for (1 .. 1) {
    $env->reset;
    my $observation = nd->array([[$env->get_state_list]]);
    #for my $t (1 .. $steps_per_epoch) {
    for my $t (1 .. 30) {
	print $fout join(' ', $env->get_positions_list), "\n";
	#print "state: ", $observation->aspdl, "\n";
	my ($mu, $sigma) = $actor_net->($observation);
	#my $action = $actor_net->choose_action($observation);
	my $action = $mu;   # deterministic
	$action = $action->clip(-1, 1);
	#print "action: ", $action->aspdl, "\n";
	$env->set_action_list(($action * $a_scale)->aspdl->list);
	#$env->set_action_list((0) x $action_size);
	$env->step;
	my $reward = $env->get_reward;
	#print "(", ($env->get_state_list)[0], ") ";
	print "$reward ";
	#my $done = $reward < 10;
	#last if $done;
	$observation= nd->array([[$env->get_state_list]]);
    }
    print "end\n";
}
    exit;
}

#open my $f_action, '>', "$outdir/action.txt";
#open my $f_pos, '>', "$outdir/position.txt";
open my $f_ret, '>', "$outdir/return_length.txt";

for my $epoch (1 .. $num_epochs) {
    $g_sigma = $sigma_begin * ($num_epochs - $epoch + 1) / ($num_epochs - 1) + $sigma_end * ($epoch - 1) / ($num_epochs - 1);
    $para_env->reset;
    my ($sum_return, $sum_length, $num_episodes) = (0, 0, 0);
    #my ($episode_return, $episode_length) = (0, 0);
    #my $observation = nd->array([[$env->get_state_list]]);

    #for my $t (1 .. $steps_per_epoch) {
    #    my ($mu, $sigma) = $actor_net->($observation);
    #    my $action = $actor_net->choose_action($observation);
    #    #$action = nd->array([[0, 0]]); # debug
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
    #    my $observation_new = nd->array([[$env->get_state_list]]);
    #    $episode_return += $reward;
    #    $episode_length += 1;

    #    my $value_t = $critic_net->($observation)->aspdl->at(0, 0);
    #    my $logprobability_t = $actor_net->log_prob($action, $mu, $sigma)->aspdl->at(0, 0);

    #    $buffer->store($observation, $action, $reward, $value_t, $logprobability_t);
    #    $observation = $observation_new;

    #    if ($done || $t == $steps_per_epoch) {
    #        #exit;  # debug
    #        my $last_value = $value_t;
    #        $buffer->finish_trajectory($last_value);
    #        $sum_return += $episode_return;
    #        $sum_length += $episode_length;
    #        $num_episodes += 1;
    #        $env->reset;
    #        ($episode_return, $episode_length) = (0, 0);
    #        $observation = nd->array([[$env->get_state_list]]);
    #    }
    #}

    my @finished = (1) x $num_threads;
    my @steps = (0) x $num_threads;
    my (@observation, @action, @mu, @sigma);
    my $total_steps = 0;
    while ($total_steps < $steps_per_epoch * $num_threads) {
	my $id = $para_env->get_task_done_id;
	my $env = $envs[$id];

	if (defined($observation[$id])) {
	    my $reward = $env->get_reward;
	    my $done = $reward < 10;
	    $sum_return += $reward;
	    $sum_length += 1;

	    my $value_t = $critic_net->($observation[$id])->aspdl->at(0, 0);
	    my $logprobability_t = $actor_net->log_prob($action[$id], $mu[$id], $sigma[$id])->aspdl->at(0, 0);

	    $buffers[$id]->store($observation[$id], $action[$id], $reward, $value_t, $logprobability_t);
	    
	    if ($done) {
		my $last_value = $value_t;
		$buffers[$id]->finish_trajectory($last_value);
		$num_episodes += 1;
		$env->reset;
		$observation[$id] = undef;
	    }
	}

	if ($steps[$id] < $steps_per_epoch) {
	    $observation[$id] = nd->array([[$env->get_state_list]]);
	    ($mu[$id], $sigma[$id]) = $actor_net->($observation[$id]);
	    $action[$id] = $actor_net->choose_action($observation[$id]); # maybe choose action using mu, sigma?
	    $action[$id] = $action[$id]->clip(-1, 1);
	    $env->set_action_list(($action[$id] * $a_scale)->aspdl->list);
	    $para_env->step($id);
	    $finished[$id] = 0;
	    ++$steps[$id];
	    ++$total_steps;
	} else {
	    $finished[$id] = 1;
	}
    }

    my $all_finished = 0;
    while (!$all_finished) {
	my $id = $para_env->get_task_done_id;
	$finished[$id] = 1;
	my $env = $envs[$id];

	if (defined($observation[$id])) {
	    my $reward = $env->get_reward;
	    $sum_return += $reward;
	    $sum_length += 1;

	    my $value_t = $critic_net->($observation[$id])->aspdl->at(0, 0);
	    my $logprobability_t = $actor_net->log_prob($action[$id], $mu[$id], $sigma[$id])->aspdl->at(0, 0);

	    $buffers[$id]->store($observation[$id], $action[$id], $reward, $value_t, $logprobability_t);
	    
	    my $last_value = $value_t;
	    $buffers[$id]->finish_trajectory($last_value);
	    $num_episodes += 1;
	    $env->reset;
	    $observation[$id] = undef;
	}

	$all_finished = 1;
	for (@finished) {
	    $all_finished = $all_finished && $_;
	}
    }

    my (@observation_buffers, @action_buffers, @advantage_buffers, @return_buffers, @logprobability_buffers);
    for my $id (0 .. $num_threads - 1) {
	my ($observation_buffer, $action_buffer, $advantage_buffer, $return_buffer, $logprobability_buffer) = $buffers[$id]->get;
	push @observation_buffers, $observation_buffer;
	push @action_buffers, $action_buffer;
	push @advantage_buffers, $advantage_buffer;
	push @return_buffers, $return_buffer;
	push @logprobability_buffers, $logprobability_buffer;
    }
    my $all_observation_buffer = nd->concat(@observation_buffers, dim => 0);
    my $all_action_buffer = nd->concat(@action_buffers, dim => 0);
    my $all_advantage_buffer = nd->concat(@advantage_buffers, dim => 0);
    my $all_return_buffer = nd->concat(@return_buffers, dim => 0);
    my $all_logprobability_buffer = nd->concat(@logprobability_buffers, dim => 0);

    for (1 .. $train_policy_iterations) {
	my $kl = train_policy($all_observation_buffer, $all_action_buffer, $all_logprobability_buffer, $all_advantage_buffer);
	if ($kl > 1.5 * $target_kl) {
	    last;
	}
    }

    for (1 .. $train_value_iterations) {
	train_value_function($all_observation_buffer, $all_return_buffer);
    }

    print "Epoch: $epoch. Sigma: $g_sigma. Mean Return: ", $sum_return / $num_episodes, ". Mean Length: ", $sum_length / $num_episodes, "\n";
    print $f_ret $sum_return / $num_episodes, " ", $sum_length / $num_episodes, "\n";

    if ($epoch % $save_interval == 0) {
	$actor_net->save_parameters(sprintf("$save_model/actor-%06d.par", $epoch));
	$critic_net->save_parameters(sprintf("$save_model/critic-%06d.par", $epoch));
    }

    last if $interrupt;
}

$actor_net->save_parameters("$save_model/actor.par");
$critic_net->save_parameters("$save_model/critic.par");
