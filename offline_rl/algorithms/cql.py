# -*- coding: utf-8 -*-
'''
@File    :   cql.py
@Time    :   2024/06/05 20:03:11
@Author  :   Mingjiang Liu 
@Version :   1.0
@Desc    :   CQL algorithm
'''


from copy import deepcopy
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
TensorBatch = List[torch.Tensor]

from offline_rl.utils import soft_update


class Scalar(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self) -> nn.Parameter:
        return self.constant


class HybridCQL():
    def __init__(
        self,
        critic_1,
        critic_1_optimizer,
        critic_2,
        critic_2_optimizer,
        actor,
        actor_optimizer,
        target_entropy_d: float,
        target_entropy_c: float,
        discount: float = 0.99,
        alpha_multiplier: float = 1.0,
        use_automatic_entropy_tuning: bool = True,
        backup_entropy: bool = True,
        policy_lr: float = 1e-4,
        qf_lr: float = 3e-4,
        sac_alpha_lr:  float = 1e-4,
        cql_alpha_lr:  float = 3e-4,
        soft_target_update_rate: float = 5e-3,
        bc_steps: int=100000,
        target_update_period: int = 1,
        cql_n_actions: int = 10,
        cql_importance_sample: bool = True,
        cql_lagrange: bool = False,
        cql_target_action_gap: float = -1.0,
        cql_temp: float = 1.0,
        cql_alpha: float = 1.0,
        cql_init_log_alpha: float = -2.0,
        cql_max_target_backup: bool = False,
        cql_clip_diff_min: float = -np.inf,
        cql_clip_diff_max: float = np.inf,
        device: str = "cuda",
    ):
        super().__init__()

        self.discount = discount
        self.target_entropy_d = target_entropy_d
        self.target_entropy_c = target_entropy_c
        self.alpha_multiplier = alpha_multiplier
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        self.backup_entropy = backup_entropy
        self.policy_lr = policy_lr
        self.qf_lr = qf_lr
        self.sac_alpha_lr = sac_alpha_lr
        self.cql_alhpa_lr = cql_alpha_lr
        self.soft_target_update_rate = soft_target_update_rate
        self.bc_steps = bc_steps
        self.target_update_period = target_update_period
        self.cql_n_actions = cql_n_actions
        self.cql_importance_sample = cql_importance_sample
        self.cql_lagrange = cql_lagrange
        self.cql_target_action_gap = cql_target_action_gap
        self.cql_temp = cql_temp
        self.cql_alpha = cql_alpha
        self.cql_init_log_alpha = cql_init_log_alpha
        self.cql_max_target_backup = cql_max_target_backup
        self.cql_clip_diff_min = cql_clip_diff_min
        self.cql_clip_diff_max = cql_clip_diff_max
        self._device = device

        self.critic_1 = critic_1
        self.critic_2 = critic_2

        self.target_critic_1 = deepcopy(self.critic_1).to(device)
        self.target_critic_2 = deepcopy(self.critic_2).to(device)

        self.actor = actor

        self.actor_optimizer = actor_optimizer
        self.critic_1_optimizer = critic_1_optimizer
        self.critic_2_optimizer = critic_2_optimizer

        if self.use_automatic_entropy_tuning:
            self.log_alpha_d = Scalar(0.0)
            self.alpha_optimizer_d = torch.optim.Adam(
                self.log_alpha_d.parameters(),
                lr=self.sac_alpha_lr,
            )
            self.log_alpha_c = Scalar(0.0)
            self.alpha_optimizer_c = torch.optim.Adam(
                self.log_alpha_c.parameters(),
                lr=self.sac_alpha_lr,
            )
        else:
            self.log_alpha_d = None
            self.alpha_optimizer_c = None

        self.log_alpha_prime = Scalar(cql_init_log_alpha) # CQL log alpha
        self.alpha_prime_optimizer = torch.optim.Adam(
            self.log_alpha_prime.parameters(),
            lr=self.cql_alhpa_lr,
        )

        self.total_it = 0

    def update_target_network(self, soft_target_update_rate: float):
        soft_update(self.target_critic_1, self.critic_1, soft_target_update_rate)
        soft_update(self.target_critic_2, self.critic_2, soft_target_update_rate)

    def _alpha_and_alpha_loss(self, log_alpha: Scalar, observations:torch.tensor, shot_probs:torch.tensor, log_probs:torch.tensor, target_entropy:torch.tensor, log_probs_2:torch.tensor=torch.empty(0)):
        if self.use_automatic_entropy_tuning:
            if log_probs_2.numel() != 0:
                # continuous action alpha
                alpha_loss = -(log_alpha() * (shot_probs * (shot_probs * (log_probs + log_probs_2) + target_entropy)).detach()).sum(-1).mean()
            else:
                # discrete action alpha
                alpha_loss = -(log_alpha() * (shot_probs * (log_probs + target_entropy)).detach()).sum(-1).mean()
            alpha = log_alpha().exp().detach() * self.alpha_multiplier
        else:
            alpha_loss = observations.new_tensor(0.0)
            alpha = observations.new_tensor(self.alpha_multiplier)
        return alpha, alpha_loss

    def _policy_loss(
        self,
        last_time_shot_type: torch.Tensor, 
        hit_xy: torch.Tensor, 
        player_location_xy: torch.Tensor, 
        opponent_location_xy: torch.Tensor,
        shot_type: torch.tensor,
        landing_xy: torch.tensor,
        move_xy: torch.tensor,
        shot_sample: torch.Tensor,
        landing_sample: torch.tensor,
        move_sample: torch.tensor,
        alpha_d: torch.Tensor,
        alpha_c: torch.Tensor,
        shot_probs: torch.tensor,
        log_shot_probs: torch.Tensor,
        landing_log_prob: torch.tensor,
        move_log_prob: torch.tensor,
    ) -> torch.Tensor:
        if self.total_it <= self.bc_steps:
            # loss for bc
            data_log_shot_probs, data_landing_log_prob, data_move_log_prob = self.actor.log_prob(last_time_shot_type, hit_xy, player_location_xy, opponent_location_xy, shot_type, landing_xy, move_xy)
            policy_loss = ((alpha_d * shot_probs * data_log_shot_probs + alpha_c * shot_probs * shot_probs * (data_landing_log_prob + data_move_log_prob)).sum(dim=-1, keepdim=True) - data_landing_log_prob - data_move_log_prob - (shot_type * data_log_shot_probs).sum(dim=-1, keepdim=True)).mean()
        else:
            # loss for CQL/SAC
            # q_new_actions: [batch_size, shot_type_dim]
            q_new_actions = torch.min(
                self.critic_1(last_time_shot_type, hit_xy, player_location_xy, opponent_location_xy, shot_sample, landing_sample, move_sample),
                self.critic_2(last_time_shot_type, hit_xy, player_location_xy, opponent_location_xy, shot_sample, landing_sample, move_sample),
            )
            policy_loss_d = (shot_probs * (alpha_d * log_shot_probs - q_new_actions)).sum(dim=-1).mean()
            policy_loss_c = (shot_probs * (alpha_c * shot_probs * (landing_log_prob + move_log_prob) - q_new_actions)).sum(dim=-1).mean()
            policy_loss = policy_loss_d + policy_loss_c
        return policy_loss

    def _q_loss(
        self,
        last_time_shot_type: torch.Tensor, 
        hit_xy: torch.Tensor, 
        player_location_xy: torch.Tensor, 
        opponent_location_xy: torch.Tensor, 
        shot_type: torch.Tensor, 
        landing_xy: torch.Tensor, 
        move_xy: torch.Tensor, 
        next_last_time_shot_type: torch.Tensor, 
        next_hit_xy: torch.Tensor, 
        next_player_location_xy: torch.Tensor, 
        next_opponent_location_xy: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        alpha_d: torch.Tensor,
        alpha_c: torch.Tensor,
        log_dict: Dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = last_time_shot_type.shape[0]
        # current step q(s,a), q(s,a): [batch_size, shot_type_dim]
        q1_predicted = self.critic_1(last_time_shot_type, hit_xy, player_location_xy, opponent_location_xy, shot_type, landing_xy, move_xy)
        q2_predicted = self.critic_2(last_time_shot_type, hit_xy, player_location_xy, opponent_location_xy, shot_type, landing_xy, move_xy)

        if self.cql_max_target_backup:
            # if use max_target_backup, i.g., TD-error = Q(s,a) - (r + gamma * max Q(s',a'))
            # [batch_size, repeat, n]
            next_shot_sample, next_shot_probs, next_log_shot_probs, next_landing_sample, next_landing_log_prob, next_move_sample, next_move_log_prob = self.actor(
                next_last_time_shot_type, 
                next_hit_xy, 
                next_player_location_xy, 
                next_opponent_location_xy,
                repeat=self.cql_n_actions
            )
            # min_target_q: [batch_size, repeat, shot_type_dim]
            min_target_q = torch.min( 
                    self.target_critic_1(next_last_time_shot_type, next_hit_xy, next_player_location_xy, next_opponent_location_xy, next_shot_sample, next_landing_sample, next_move_sample),
                    self.target_critic_2(next_last_time_shot_type, next_hit_xy, next_player_location_xy, next_opponent_location_xy, next_shot_sample, next_landing_sample, next_move_sample),
                )
            # target_q_values: [batch_size, repeat]
            target_q_values, _ = torch.max(
                min_target_q,
                dim=-1,
            )
            # target_q_values, max_target_indices: [batch_size]
            target_q_values, max_target_indices = torch.max(
                target_q_values,
                dim=-1,
            )
            # get max Q
            target_q_values = target_q_values.unsqueeze(dim=-1) # [batch_size, 1]
            
            # [batch_size, shot_type_dim]
            next_shot_probs = torch.gather(
                next_shot_probs, 1, max_target_indices.unsqueeze(1).unsqueeze(-1).expand(-1, -1, next_shot_probs.shape[-1])
            ).squeeze(1)
            next_log_shot_probs = torch.gather(
                next_log_shot_probs, 1, max_target_indices.unsqueeze(1).unsqueeze(-1).expand(-1, -1, next_shot_probs.shape[-1])
            ).squeeze(1)
            # [batch_size, 1]
            next_landing_log_prob = torch.gather(
                next_landing_log_prob, 1, max_target_indices.unsqueeze(1).unsqueeze(-1)
            ).squeeze(1)
            next_move_log_prob = torch.gather(
                next_move_log_prob, 1, max_target_indices.unsqueeze(1).unsqueeze(-1)
            ).squeeze(1)
        else:
            # [batch_size, n]
            next_shot_sample, next_shot_probs, next_log_shot_probs, next_landing_sample, next_landing_log_prob, next_move_sample, next_move_log_prob = self.actor(
                next_last_time_shot_type, 
                next_hit_xy, 
                next_player_location_xy, 
                next_opponent_location_xy,
            )
            target_q_values = torch.min(
                self.target_critic_1(next_last_time_shot_type, next_hit_xy, next_player_location_xy, next_opponent_location_xy, next_shot_sample, next_landing_sample, next_move_sample),
                self.target_critic_2(next_last_time_shot_type, next_hit_xy, next_player_location_xy, next_opponent_location_xy, next_shot_sample, next_landing_sample, next_move_sample),
            )
            # get E[Q]
            target_q_values = (next_shot_probs * target_q_values).sum(dim=-1, keepdim=True) # [batch_size, 1]

        if self.backup_entropy:
            # target_q = Q + entropy
            # [batch_size, 1]
            target_q_values = target_q_values + (next_shot_probs * (- alpha_d * next_log_shot_probs - alpha_c * next_shot_probs * (next_landing_log_prob + next_move_log_prob))).sum(dim=-1, keepdim=True)

        td_target = rewards + (1.0 - dones) * self.discount * target_q_values.detach()
        qf1_loss = F.mse_loss(q1_predicted[shot_type>0], td_target.squeeze(dim=-1).detach())
        qf2_loss = F.mse_loss(q2_predicted[shot_type>0], td_target.squeeze(dim=-1).detach())

        # CQL
        landing_xy_dim = landing_xy.shape[-1]
        move_xy_dim = move_xy.shape[-1]
        # random sampling
        cql_random_landing_xy = landing_xy.new_empty(
            (batch_size, self.cql_n_actions, landing_xy_dim), requires_grad=False
        ).uniform_(0, 1)
        cql_random_move_xy = landing_xy.new_empty(
            (batch_size, self.cql_n_actions, move_xy_dim), requires_grad=False
        ).uniform_(0, 1)
        # cur policy sampling, [batch_size, repeat, n]
        cql_cur_shot_sample, cql_cur_shot_probs, cql_cur_log_shot_probs, cql_cur_landing_sample, cql_cur_landing_log_prob, cql_cur_move_sample, cql_cur_move_log_prob = self.actor(
            last_time_shot_type, 
            hit_xy, 
            player_location_xy, 
            opponent_location_xy,
            repeat=self.cql_n_actions
        )
        cql_next_shot_sample, cql_next_shot_probs, cql_next_log_shot_probs, cql_next_landing_sample, cql_next_landing_log_prob, cql_next_move_sample, cql_next_move_log_prob = self.actor(
            next_last_time_shot_type, 
            next_hit_xy, 
            next_player_location_xy, 
            next_opponent_location_xy,
            repeat=self.cql_n_actions
        )
        cql_cur_shot_sample, cql_cur_shot_probs, cql_cur_log_shot_probs, cql_cur_landing_sample, cql_cur_landing_log_prob, cql_cur_move_sample, cql_cur_move_log_prob = (
            cql_cur_shot_sample.detach(), 
            cql_cur_shot_probs.detach(), 
            cql_cur_log_shot_probs.detach(), 
            cql_cur_landing_sample.detach(), 
            cql_cur_landing_log_prob.detach(), 
            cql_cur_move_sample.detach(), 
            cql_cur_move_log_prob.detach()
        )
        cql_next_shot_sample, cql_next_shot_probs, cql_next_log_shot_probs, cql_next_landing_sample, cql_next_landing_log_prob, cql_next_move_sample, cql_next_move_log_prob = (
            cql_next_shot_sample.detach(), 
            cql_next_shot_probs.detach(), 
            cql_next_log_shot_probs.detach(), 
            cql_next_landing_sample.detach(), 
            cql_next_landing_log_prob.detach(), 
            cql_next_move_sample.detach(), 
            cql_next_move_log_prob.detach()
        )

        # [batch_size, repeat, shot_type_dim]
        cql_q1_rand = self.critic_1(last_time_shot_type, hit_xy, player_location_xy, opponent_location_xy, shot_type, cql_random_landing_xy, cql_random_move_xy)
        cql_q2_rand = self.critic_2(last_time_shot_type, hit_xy, player_location_xy, opponent_location_xy, shot_type, cql_random_landing_xy, cql_random_move_xy)
        cql_q1_current_actions = self.critic_1(last_time_shot_type, hit_xy, player_location_xy, opponent_location_xy, cql_cur_shot_sample, cql_cur_landing_sample, cql_cur_move_sample)
        cql_q2_current_actions = self.critic_2(last_time_shot_type, hit_xy, player_location_xy, opponent_location_xy, cql_cur_shot_sample, cql_cur_landing_sample, cql_cur_move_sample)
        cql_q1_next_actions = self.critic_1(last_time_shot_type, hit_xy, player_location_xy, opponent_location_xy, cql_next_shot_sample, cql_next_landing_sample, cql_next_move_sample)
        cql_q2_next_actions = self.critic_2(last_time_shot_type, hit_xy, player_location_xy, opponent_location_xy, cql_next_shot_sample, cql_next_landing_sample, cql_next_move_sample)

        # [batch_size, repeat*3+1, shot_type_dim]
        cql_cat_q1 = torch.cat(
            [
                cql_q1_rand,
                torch.unsqueeze(q1_predicted, 1),
                cql_q1_next_actions,
                cql_q1_current_actions,
            ],
            dim=1,
        )
        cql_cat_q2 = torch.cat(
            [
                cql_q2_rand,
                torch.unsqueeze(q2_predicted, 1),
                cql_q2_next_actions,
                cql_q2_current_actions,
            ],
            dim=1,
        )
        cql_std_q1 = torch.std(cql_cat_q1, dim=1)
        cql_std_q2 = torch.std(cql_cat_q2, dim=1)

        if self.cql_importance_sample:
            random_density = np.log(0.5**(landing_xy_dim+move_xy_dim))
            # [batch_size, repeat*3, shot_type_dim]
            cql_cat_q1 = torch.cat(
                [
                    cql_q1_rand - random_density,
                    cql_q1_next_actions - cql_next_landing_log_prob.detach() - cql_next_move_log_prob.detach(),
                    cql_q1_current_actions - cql_cur_landing_log_prob.detach() - cql_cur_move_log_prob.detach(),
                ],
                dim=1,
            )
            cql_cat_q2 = torch.cat(
                [
                    cql_q2_rand - random_density,
                    cql_q1_next_actions - cql_next_landing_log_prob.detach() - cql_next_move_log_prob.detach(),
                    cql_q1_current_actions - cql_cur_landing_log_prob.detach() - cql_cur_move_log_prob.detach(),
                ],
                dim=1,
            )

        # [batch_size]
        cql_qf1_ood = torch.log((torch.exp(cql_cat_q1).sum(dim=1) / self.cql_temp).sum(dim=-1)) * self.cql_temp
        cql_qf2_ood = torch.log((torch.exp(cql_cat_q2).sum(dim=1) / self.cql_temp).sum(dim=-1)) * self.cql_temp

        """Subtract the log likelihood of data"""
        # [batch_size]
        cql_qf1_diff = torch.clamp(
            cql_qf1_ood - q1_predicted[shot_type>0],
            self.cql_clip_diff_min,
            self.cql_clip_diff_max,
        ).mean()
        cql_qf2_diff = torch.clamp(
            cql_qf2_ood - q2_predicted[shot_type>0],
            self.cql_clip_diff_min,
            self.cql_clip_diff_max,
        ).mean()

        if self.cql_lagrange:
            # auto CQL alpha
            alpha_prime = torch.clamp(
                torch.exp(self.log_alpha_prime()), min=0.0, max=100.0
            )
            cql_min_qf1_loss = (
                alpha_prime
                * self.cql_alpha
                * (cql_qf1_diff - self.cql_target_action_gap)
            )
            cql_min_qf2_loss = (
                alpha_prime
                * self.cql_alpha
                * (cql_qf2_diff - self.cql_target_action_gap)
            )

            self.alpha_prime_optimizer.zero_grad()
            alpha_prime_loss = (-cql_min_qf1_loss - cql_min_qf2_loss) * 0.5
            alpha_prime_loss.backward(retain_graph=True)
            self.alpha_prime_optimizer.step()
        else:
            cql_min_qf1_loss = cql_qf1_diff * self.cql_alpha
            cql_min_qf2_loss = cql_qf2_diff * self.cql_alpha
            alpha_prime_loss = last_time_shot_type.new_tensor(0.0)
            alpha_prime = last_time_shot_type.new_tensor(0.0)

        qf_loss = qf1_loss + qf2_loss + cql_min_qf1_loss + cql_min_qf2_loss

        log_dict.update(
            {
                "loss/qf1_loss": qf1_loss.item(),
                "loss/qf2_loss": qf2_loss.item(),
                "q_value/mean_q1_value": q1_predicted[shot_type > 0].mean().item(),
                "q_value/mean_q2_value": q2_predicted[shot_type > 0].mean().item(),
                "q_value/mean_target_q": target_q_values.mean().item(),
            }
        )

        log_dict.update(
            {
                "q_value/cql_std_q1": cql_std_q1.mean().item(),
                "q_value/cql_std_q1": cql_std_q2.mean().item(),
                "q_value/cql_q1_rand": cql_q1_rand.mean().item(),
                "q_value/cql_q2_rand": cql_q2_rand.mean().item(),
                "loss/cql_min_qf1_loss": cql_min_qf1_loss.mean().item(),
                "loss/cql_min_qf2_loss": cql_min_qf2_loss.mean().item(),
                "q_value/cql_q1_diff": cql_qf1_diff.mean().item(),
                "q_value/cql_q2_diff": cql_qf2_diff.mean().item(),
                "q_value/cql_q1_current_action": cql_q1_current_actions.mean().item(),
                "q_value/cql_q2_current_action": cql_q2_current_actions.mean().item(),
                "q_value/cql_q1_next_action": cql_q1_next_actions.mean().item(),
                "q_value/cql_q2_next_action":cql_q2_next_actions.mean().item(),
                "loss/cql_alpha_loss": alpha_prime_loss.item(),
                "alpha_value/cql_alpha": alpha_prime.item(),
            }
        )

        return qf_loss, alpha_prime, alpha_prime_loss

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        (
            last_time_shot_type, 
            hit_xy, 
            player_location_xy, 
            opponent_location_xy, 
            shot_type, 
            landing_xy, 
            move_xy, 
            next_last_time_shot_type, 
            next_hit_xy, 
            next_player_location_xy, 
            next_opponent_location_xy, 
            rewards, 
            dones
        ) = batch
        self.total_it += 1

        # [batch_size, n]
        shot_sample, shot_probs, log_shot_probs, landing_sample, landing_log_prob, move_sample, move_log_prob = self.actor(last_time_shot_type, hit_xy, player_location_xy, opponent_location_xy)

        alpha_d, alpha_d_loss = self._alpha_and_alpha_loss(self.log_alpha_d, last_time_shot_type, shot_probs, log_shot_probs, self.target_entropy_d)
        alpha_c, alpha_c_loss = self._alpha_and_alpha_loss(self.log_alpha_c, last_time_shot_type, shot_probs, landing_log_prob, self.target_entropy_c, move_log_prob)


        """ Policy loss """
        policy_loss = self._policy_loss(
            last_time_shot_type, 
            hit_xy, 
            player_location_xy, 
            opponent_location_xy, 
            shot_type, 
            landing_xy, 
            move_xy, 
            shot_sample,
            landing_sample,
            move_sample,
            alpha_d,
            alpha_c,
            shot_probs,
            log_shot_probs,
            landing_log_prob,
            move_log_prob,
        )

        log_dict = {
            "loss/policy_loss": policy_loss.item(),
            "loss/sac_alpha_d_loss": alpha_d_loss.item(),
            "alpha_value/sac_alpha_d": alpha_d.item(),
            "loss/sac_alpha_c_loss": alpha_c_loss.item(),
            "alpha_value/sac_alpha_c": alpha_c.item(),
        }

        """ Q function loss """
        qf_loss, alpha_prime, alpha_prime_loss = self._q_loss(
            last_time_shot_type, 
            hit_xy, 
            player_location_xy, 
            opponent_location_xy, 
            shot_type, 
            landing_xy, 
            move_xy, 
            next_last_time_shot_type, 
            next_hit_xy, 
            next_player_location_xy, 
            next_opponent_location_xy,
            rewards, 
            dones, 
            alpha_d, 
            alpha_c, 
            log_dict
        )

        if self.use_automatic_entropy_tuning:
            self.alpha_optimizer_d.zero_grad()
            alpha_d_loss.backward()
            self.alpha_optimizer_d.step()
            self.alpha_optimizer_c.zero_grad()
            alpha_c_loss.backward()
            self.alpha_optimizer_c.step()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        qf_loss.backward(retain_graph=True)
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        if self.total_it % self.target_update_period == 0:
            self.update_target_network(self.soft_target_update_rate)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic1": self.critic_1.state_dict(),
            "critic2": self.critic_2.state_dict(),
            "critic1_target": self.target_critic_1.state_dict(),
            "critic2_target": self.target_critic_2.state_dict(),
            "critic_1_optimizer": self.critic_1_optimizer.state_dict(),
            "critic_2_optimizer": self.critic_2_optimizer.state_dict(),
            "actor_optim": self.actor_optimizer.state_dict(),
            "sac_log_alpha_d": self.log_alpha_d,
            "sac_log_alpha_d_optim": self.alpha_optimizer_d.state_dict(),
            "sac_log_alpha_c": self.log_alpha_c,
            "sac_log_alpha_c_optim": self.alpha_optimizer_c.state_dict(),
            "cql_log_alpha": self.log_alpha_prime,
            "cql_log_alpha_optim": self.alpha_prime_optimizer.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict=state_dict["actor"])
        self.critic_1.load_state_dict(state_dict=state_dict["critic1"])
        self.critic_2.load_state_dict(state_dict=state_dict["critic2"])

        self.target_critic_1.load_state_dict(state_dict=state_dict["critic1_target"])
        self.target_critic_2.load_state_dict(state_dict=state_dict["critic2_target"])

        self.critic_1_optimizer.load_state_dict(
            state_dict=state_dict["critic_1_optimizer"]
        )
        self.critic_2_optimizer.load_state_dict(
            state_dict=state_dict["critic_2_optimizer"]
        )
        self.actor_optimizer.load_state_dict(state_dict=state_dict["actor_optim"])

        self.log_alpha_d = state_dict["sac_log_alpha_d"]
        self.alpha_optimizer_d.load_state_dict(
            state_dict=state_dict["sac_log_alpha_d_optim"]
        )

        self.log_alpha_c = state_dict["sac_log_alpha_c"]
        self.alpha_optimizer_c.load_state_dict(
            state_dict=state_dict["sac_log_alpha_c_optim"]
        )

        self.log_alpha_prime = state_dict["cql_log_alpha"]
        self.alpha_prime_optimizer.load_state_dict(
            state_dict=state_dict["cql_log_alpha_optim"]
        )
        self.total_it = state_dict["total_it"]