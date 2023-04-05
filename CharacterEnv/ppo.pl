#!/usr/bin/env perl
use AI::MXNet qw(mx);
use AI::MXNet::Gluon::NN qw(nn);
use AI::MXNet::Gluon qw(gluon);
use AI::MXNet::AutoGrad qw(autograd);
use Getopt::Long qw(:config no_ignore_case);
use strict;
use warnings;

my ($use_gpu, $load_model, $save_model, $play_policy) = (0, undef, "model", 0);
my $outdir = "output";
GetOptions(
    'G|gpu' => \$use_gpu,
    'l|load_model=s'	=> \$load_model,
    's|save_model=s'	=> \$save_model,
    'p|play_policy'	=> \$play_policy,
);

mkdir $outdir unless -e $outdir;
my $ctx = $use_gpu ? mx->gpu : mx->cpu;

sub discounted_cumulative_sums {
    my ($x, $discount) = @_;
    my $n = $x->size;
    my $y = nd->zeros([$n], ctx => $ctx);
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
#my $x = nd->ones([5], 'ctx' => $ctx);
#my $y = discounted_cumulative_sums($x, 0.5);
#print $y->aspdl;

package Buffer {
    sub new {
	my ($class, $ob_dim, $ac_dim, $size, $gamma, $lam) = @_;
	$gamma = 0.99 unless defined($gamma);
	$lam = 0.95 unless defined($lam);
	my $self = bless {
	    observation_buffer => nd->zeros([$size, $ob_dim], ctx => $ctx),
	    action_buffer => nd->zeros([$size, $ac_dim], ctx => $ctx),
	    advantage_buffer => nd->zeros([$size], ctx => $ctx),
	    reward_buffer => nd->zeros([$size], ctx => $ctx),
	    return_buffer => nd->zeros([$size], ctx => $ctx),
	    value_buffer => nd->zeros([$size], ctx => $ctx),
	    logprobability_buffer => nd->zeros([$size], ctx => $ctx),
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
	my $rewards = nd->concatenate([$self->{reward_buffer}->slice([$a, $b]), nd->array([$last_value], ctx => $ctx)]);
	my $values = nd->concatenate([$self->{value_buffer}->slice([$a, $b]), nd->array([$last_value], ctx => $ctx)]);
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
}

use Math::Trig ();
use GD;
use File::Slurp;
use PDL;
use FindBin;
use blib "$FindBin::Bin/blib";
use CharacterEnv;

my $env = CharacterEnv->new('data/env_config.json');
my $unit_size = 512;
my $state_size = $env->get_state_size;
my $action_size = $env->get_action_size;

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
		$net->name_scope(sub {
			for my $size (@$sizes) {
			    $net->add(nn->Dense($size, activation => $activation));
			}
		    });
		$self->dense_base($net);
		$self->dense_mu(nn->Dense($action_size, in_units => $unit_size, activation => 'tanh'));
		$self->dense_sigma(nn->Dense($action_size, in_units => $unit_size, activation => 'softrelu'));
	    });
    }

    method forward($x) {
	my $y = $self->dense_base->($x);
	my ($mu, $sigma) = ($self->dense_mu->($y), $self->dense_sigma->($y));
	return ($mu, $sigma);
    }

    method choose_action($x) {
	my ($mu, $sigma) = $self->($x);
	return $self->sample($mu, $sigma);
    }

    method sample($mu, $sigma) {
	my $eps = nd->random_normal(0, 1, $mu->shape, ctx => $ctx);
	return $mu + $sigma * $eps;
    }
    
    method log_prob($x, $mu, $sigma) {
	return nd->sum(-0.5 * log(2.0 * pi) - $sigma->add(1e-8)->log() - ($x - $mu) ** 2 / (2 * $sigma ** 2 + 1e-8), axis => 1);
    }
}


my $steps_per_epoch = 500;
my $epochs = 5000;
my $gamma = 0.99;
my $clip_ratio = 0.2;
my $train_policy_iterations = 80;
my $train_value_iterations = 80;
my $lam = 0.97;
my $target_kl = 0.01;
my $policy_learning_rate = 3e-4;
my $value_function_learning_rate = 1e-3;
my $actor_net = ActorModel->new(sizes => [$unit_size, $unit_size],  activation => 'relu');
#print $actor_net;
my $critic_net = mlp([$unit_size, $unit_size, 1], 'relu');
#print $critic_net;
if (defined($load_model)) {
    die "Canno find the model files!" unless -d $load_model and -f "$load_model/actor.par" and -f "$load_model/critic.par";
    $actor_net->load_parameters("$load_model/actor.par", ctx => $ctx);
    $critic_net->load_parameters("$load_model/critic.par", ctx => $ctx);
} else {
    $actor_net->initialize(mx->init->Xavier(), ctx => $ctx);
    $critic_net->initialize(mx->init->Xavier(), ctx => $ctx);
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

my $buffer = Buffer->new($state_size, $action_size, $steps_per_epoch);
$env->reset;
#my $a_scale = Math::Trig::pi;
my $a_scale = 1;

my $i_img = 0;

if (defined($save_model)) {
    mkdir $save_model unless -e $save_model;
}

my $interrupt = 0;

$SIG{INT} = sub {
    $interrupt = 1;
};

if ($play_policy) {
    open my $fout, '>', "$outdir/positions.txt";
    my $observation = nd->array([[$env->get_state_list]], ctx => $ctx);
    #for my $t (1 .. $steps_per_epoch) {
    for my $t (1 .. 100) {
	print $fout join(' ', $env->get_positions_list), "\n";
	my ($mu, $sigma) = $actor_net->($observation);
	#my $action = $actor_net->choose_action($observation);
	my $action = $mu;   # deterministic
	$action = $action->clip(-1, 1);
	$env->set_action_list(($action * $a_scale)->aspdl->list);
	$env->step;
	my $reward = $env->get_reward;
	#my $done = $reward < 10;
	#last if $done;
	$observation= nd->array([[$env->get_state_list]], ctx => $ctx);
    }
    exit;
}

#open my $f_action, '>', "$outdir/action.txt";
#open my $f_pos, '>', "$outdir/position.txt";
open my $f_ret, '>', "$outdir/return_length.txt";

for my $epoch (1 .. $epochs) {
    my ($sum_return, $sum_length, $num_episodes) = (0, 0, 0);
    my ($episode_return, $episode_length) = (0, 0);
    my $observation = nd->array([[$env->get_state_list]], ctx => $ctx);

    for my $t (1 .. $steps_per_epoch) {
	my ($mu, $sigma) = $actor_net->($observation);
	my $action = $actor_net->choose_action($observation);
	#$action = nd->array([[0, 0]], ctx => $ctx); # debug
	#print $f_action join(' ', $action->aspdl->list), "\n";	# debug
	#print $f_pos join(' ', $env->get_state_list), "\n";	# debug
	#render;    # debug
	$action = $action->clip(-1, 1);
	$env->set_action_list(($action * $a_scale)->aspdl->list);
	$env->step;
	my $reward = $env->get_reward;
	#print(($env->get_state_list)[4], "\t", $reward, "\n");  # debug
	#print(join(" ", $env->get_state_list), "\n");   # debug
	my $done = $reward < 10;
	#$done = 0; # debug
	my $observation_new = nd->array([[$env->get_state_list]], ctx => $ctx);
	$episode_return += $reward;
	$episode_length += 1;

	my $value_t = $critic_net->($observation)->aspdl->at(0, 0);
	my $logprobability_t = $actor_net->log_prob($action, $mu, $sigma)->aspdl->at(0, 0);

	$buffer->store($observation, $action, $reward, $value_t, $logprobability_t);
	$observation = $observation_new;

	if ($done || $t == $steps_per_epoch) {
	    #exit;  # debug
	    my $last_value = $value_t;
	    $buffer->finish_trajectory($last_value);
	    $sum_return += $episode_return;
	    $sum_length += $episode_length;
	    $num_episodes += 1;
	    $env->reset;
	    ($episode_return, $episode_length) = (0, 0);
	    $observation = nd->array([[$env->get_state_list]], ctx => $ctx);
	}
    }

    my ($observation_buffer, $action_buffer, $advantage_buffer, $return_buffer, $logprobability_buffer) = $buffer->get;

    for (1 .. $train_policy_iterations) {
	my $kl = train_policy($observation_buffer, $action_buffer, $logprobability_buffer, $advantage_buffer);
	if ($kl > 1.5 * $target_kl) {
	    last;
	}
    }

    for (1 .. $train_value_iterations) {
	train_value_function($observation_buffer, $return_buffer);
    }

    print "Epoch: $epoch. Mean Return: ", $sum_return / $num_episodes, " Mean Length: ", $sum_length / $num_episodes, "\n";
    print $f_ret $sum_return / $num_episodes, " ", $sum_length / $num_episodes, "\n";

    last if $interrupt;
}

if (defined($save_model)) {
    $actor_net->save_parameters("$save_model/actor.par");
    $critic_net->save_parameters("$save_model/critic.par");
}
