import torch
from torch.nn import functional as F


def _clip_actions(algo, actions):
    epsilon = 1e-6
    lower = torch.from_numpy(algo._env_spec.action_space.low).to(algo.device) + epsilon
    upper = torch.from_numpy(algo._env_spec.action_space.high).to(algo.device) - epsilon

    clip_up = (actions > upper).float()
    clip_down = (actions < lower).float()
    with torch.no_grad():
        clip = ((upper - actions) * clip_up + (lower - actions) * clip_down)

    return actions + clip

def update_loss_qf(
        algo, tensors, v,
        obs,
        actions,
        next_obs,
        dones,
        rewards,
        policy,
        use_discrete_sac: bool = False,
        turn_off_dones: bool = False,
):
    with torch.no_grad():
        alpha = algo.log_alpha.param.exp()

    if use_discrete_sac:
        action_ids = action_ids = torch.argmax(actions.long(),dim=-1)
        q1_pred = algo.qf1(obs).gather(1, action_ids.view(-1, 1)).squeeze()
        q2_pred = algo.qf2(obs).gather(1, action_ids.view(-1, 1)).squeeze()
    else:
        q1_pred = algo.qf1(obs, actions).flatten()
        q2_pred = algo.qf2(obs, actions).flatten()

    next_action_dists, *_ = policy(next_obs)
    if use_discrete_sac:
        act_probs = next_action_dists.probs
        logits = next_action_dists.logits
        log_probs = torch.log_softmax(logits, dim=-1)

        target_q_values = torch.min(
            algo.target_qf1(next_obs),
            algo.target_qf2(next_obs),
        )
        target_q_values = target_q_values - alpha * log_probs
        target_q_values = (act_probs * target_q_values).sum(dim=-1)
        target_q_values = target_q_values * algo.discount
    else:
        if hasattr(next_action_dists, 'rsample_with_pre_tanh_value'):
            new_next_actions_pre_tanh, new_next_actions = next_action_dists.rsample_with_pre_tanh_value()
            new_next_action_log_probs = next_action_dists.log_prob(new_next_actions, pre_tanh_value=new_next_actions_pre_tanh)
        else:
            new_next_actions = next_action_dists.rsample()
            new_next_actions = _clip_actions(algo, new_next_actions)
            new_next_action_log_probs = next_action_dists.log_prob(new_next_actions)

        target_q_values = torch.min(
            algo.target_qf1(next_obs, new_next_actions).flatten(),
            algo.target_qf2(next_obs, new_next_actions).flatten(),
        )

        target_q_values = target_q_values - alpha * new_next_action_log_probs
        target_q_values = target_q_values * algo.discount

    with torch.no_grad():
        if turn_off_dones:
            dones[...] = 0
        q_target = rewards + target_q_values * (1. - dones)

    # critic loss weight: 0.5
    loss_qf1 = F.mse_loss(q1_pred, q_target) * 0.5
    loss_qf2 = F.mse_loss(q2_pred, q_target) * 0.5

    tensors.update({
        'QTargetsMean': q_target.mean(),
        'QTdErrsMean': ((q_target - q1_pred).mean() + (q_target - q2_pred).mean()) / 2,
        'LossQf1': loss_qf1,
        'LossQf2': loss_qf2,
    })

def update_loss_sacp(
        algo, tensors, v,
        obs,
        policy,
        use_discrete_sac: bool = False,
):
    with torch.no_grad():
        alpha = algo.log_alpha.param.exp()

    
    action_dists, *_ = policy(obs)
    act_probs = None
    if use_discrete_sac:
        act_probs = action_dists.probs
        logits = action_dists.logits
        new_action_log_probs = torch.log_softmax(logits, dim=-1)

        min_q_values = torch.min(
            algo.qf1(obs),
            algo.qf2(obs),
        )
        loss_sacp = (act_probs * (alpha * new_action_log_probs - min_q_values)).sum(dim=-1).mean()
        
    else:
        if hasattr(action_dists, 'rsample_with_pre_tanh_value'):
            new_actions_pre_tanh, new_actions = action_dists.rsample_with_pre_tanh_value()
            new_action_log_probs = action_dists.log_prob(new_actions, pre_tanh_value=new_actions_pre_tanh)
        else:
            new_actions = action_dists.rsample()
            new_actions = _clip_actions(algo, new_actions)
            new_action_log_probs = action_dists.log_prob(new_actions)

        min_q_values = torch.min(
            algo.qf1(obs, new_actions).flatten(),
            algo.qf2(obs, new_actions).flatten(),
        )

        loss_sacp = (alpha * new_action_log_probs - min_q_values).mean()

    tensors.update({
        'SacpNewActionLogProbMean': new_action_log_probs.mean(),
        'LossSacp': loss_sacp,
    })

    v.update({
        'new_action_log_probs': new_action_log_probs,
        'act_probs': act_probs
    })


def update_loss_alpha(
        algo, tensors, v, use_discrete_sac: bool = False,
):
    if use_discrete_sac:
        loss_alpha = (v['act_probs'].detach() * (-algo.log_alpha.param * (
                v['new_action_log_probs'].detach() + algo._target_entropy
        ))).sum(dim=-1).mean()
    else:
        loss_alpha = (-algo.log_alpha.param * (
            v['new_action_log_probs'].detach() + algo._target_entropy
        )).mean()

    tensors.update({
        'Alpha': algo.log_alpha.param.exp(),
        'LossAlpha': loss_alpha,
    })


def update_targets(algo):
    """Update parameters in the target q-functions."""
    target_qfs = [algo.target_qf1, algo.target_qf2]
    qfs = [algo.qf1, algo.qf2]
    for target_qf, qf in zip(target_qfs, qfs):
        for t_param, param in zip(target_qf.parameters(), qf.parameters()):
            t_param.data.copy_(t_param.data * (1.0 - algo.tau) +
                               param.data * algo.tau)
