from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np
import torch
from omegaconf import DictConfig

import verl.utils.torch_functional as verl_F
from verl.trainer.config import AlgoConfig
from verl.utils import as_torch_index, group_mean_std
from verl.utils.import_utils import deprecated
from verl.workers.config import ActorConfig
import torch
import random
from verl.trainer.ppo.core_algos import register_adv_est


@register_adv_est("grpo_dualrm")
def compute_grpo_dualrm_advantage(
    token_batch: dict,
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:

    scores = token_level_rewards.sum(dim=-1)

    def rank_aligned_move_distance(judge_logits, gt_logits):
        judge = torch.as_tensor(judge_logits)
        gt = torch.as_tensor(gt_logits)
        n = judge.numel()

        # if torch.any(judge == 0):
        #     return 1 - torch.zeros(len(judge_logits)), False
        
        idx_j = torch.argsort(judge, stable=True)
        idx_g = torch.argsort(gt, stable=True)

        rank_j = torch.argsort(idx_j) 
        target_pos = idx_g[rank_j]
        
        current_pos = torch.arange(n, device=judge.device)
        return 0.5 + 0.5 * (1 - torch.abs(current_pos - target_pos).sum() / (2 * n)), True

    def compute_ndcg(judge_logits, gt_logits, k=None, device=None):
        
        judge = torch.as_tensor(judge_logits, dtype=torch.float32)
        gt = torch.as_tensor(gt_logits, dtype=torch.float32)
        device = judge.device

        if device is not None:
            judge = judge.to(device)
            gt = gt.to(device)
        else:
            device = judge.device

        n = judge.numel()
        if n == 0:
            return torch.tensor(1.0, device=device), False

        if k is None:
            k = n
        else:
            k = min(k, n)

        _, sorted_indices = torch.sort(judge, descending=True, stable=True)  # [N]
        ranked_relevance = gt[sorted_indices]  # [N]

        # Step 3: Compute DCG@k
        topk_rel = ranked_relevance[:k]  # [k]
        pos = torch.arange(1, k + 1, dtype=torch.float32, device=device)  # [k]
        dcg = ((2.0 ** topk_rel - 1.0) / torch.log2(pos + 1.0)).sum()

        # Step 4: Compute IDCG@k (ideal ranking: sort gt descending)
        ideal_rel, _ = torch.sort(gt, descending=True, stable=True)
        ideal_topk = ideal_rel[:k]
        idcg = ((2.0 ** ideal_topk - 1.0) / torch.log2(pos + 1.0)).sum()

        # Step 5: Normalize
        if idcg.item() == 0:
            return torch.tensor(0.0, device=device), True
        else:
            return dcg / idcg, True


    if 'rank_scores' in token_batch:
        rank_scores = token_batch['rank_scores'].sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    id2rankmean = {}
    id2rankstd = {}
    
    id2judge = defaultdict(list)
    id2gt = defaultdict(list)
    id2raw = defaultdict(list)
    id2rank = defaultdict(list)
    sharp = 0
    with torch.no_grad():

        bsz = scores.shape[0]
        for i in range(bsz):
            id2judge[index[i]].append(token_batch['scale_judge'][i])
            id2gt[index[i]].append(token_batch['pred_judge'][i])
            id2raw[index[i]].append(i)
            id2score[index[i]].append(scores[i])
            if 'rank_scores' in token_batch:
                id2rank[index[i]].append(rank_scores[i])
        
        for idx in id2score:
            
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                if 'rank_scores' in token_batch:
                    scores_tensor = torch.stack(id2score[idx])
                    # rank_scores_tensor = torch.sigmoid(0.3 * torch.stack(id2rank[idx]))
                    _scores = torch.stack(id2rank[idx])
                    rank_scores_tensor = (_scores - _scores.min()) / (_scores.max() - _scores.min() + 1e-8)
                    
                    id2mean[idx] = torch.mean(scores_tensor)
                    id2std[idx] = torch.std(scores_tensor)
                    
                    if id2mean[idx] == 1 or id2mean[idx] == 0:
                        sharp += 1
                        norm_scores_tensor = rank_scores_tensor
                    else:
                        norm_scores_tensor = id2mean[idx] * (rank_scores_tensor * (1 - scores_tensor)) + ((1 - id2mean[idx]) * (rank_scores_tensor * scores_tensor) + (id2mean[idx] * scores_tensor))
                    
                    for _j, num in zip(id2raw[idx], norm_scores_tensor):
                        rank_scores[_j] = num
                    
                    id2rankmean[idx] = torch.mean(norm_scores_tensor)
                    id2rankstd[idx] = torch.std(norm_scores_tensor)
                    
                    if random.random() > 0.99:
                        print(f"[debug]: norm_scores_tensor={norm_scores_tensor}, scores_tensor={scores_tensor}, mean={id2mean[idx]}, std={id2std[idx]}")
                else:
                    order_tensor, order_flag = rank_aligned_move_distance(id2judge[idx], id2gt[idx])
                    # order_tensor, order_flag = compute_ndcg(id2judge[idx], id2gt[idx])
                    scores_tensor = torch.stack(id2score[idx]) * order_tensor

                    id2mean[idx] = torch.mean(scores_tensor)
                    id2std[idx] = torch.std(scores_tensor)
                    
                    if order_flag and random.random() > 0.99:
                        print(f"[debug]: id2judge {id2judge[idx]}, id2gt {id2gt[idx]}, norm_scores_tensor={order_tensor}, scores_tensor={scores_tensor}")
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                if 'rank_scores' in token_batch:
                    scores[i] = (rank_scores[i] - id2rankmean[index[i]]) / (id2rankstd[index[i]] + epsilon)
                else:
                    scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
                    # scores[i] = (scores[i] - id2mean[index[i]])
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        
        if 'rank_scores' in token_batch:
            print(f"[debug]==>: {sharp} / {bsz} = {sharp / bsz}")
        scores = scores.unsqueeze(-1) * response_mask
    return scores, scores
