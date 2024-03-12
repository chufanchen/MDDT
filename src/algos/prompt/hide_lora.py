import math
import torch
import torch.nn as nn
from .l2p import Prompt


class HiDeLoRAPool(Prompt):
    def __init__(
        self,
        length=2,
        top_k=1,
        dropout_rate=0.0,
        init_range=0.02,
        init_prompts="zeros",
        num_tasks=None,
        n_head=None,
        n_layer=None,
        rank=4,
        mod_q=True,
        mod_v=True,
        mod_k=False,
        mod_ff=True,
        lora_alpha=None,
        log_mod_stats=False,
        eval_mode=False,
        continual_mode=False,
        **kwargs,
    ):
        super().__init__(dropout_rate=dropout_rate, **kwargs)
        self.log_mod_stats = log_mod_stats
        self.rank = rank
        self.mod_v = mod_v
        self.mod_q = mod_q
        self.mod_k = mod_k
        self.mod_ff = mod_ff
        self.lora_alpha = lora_alpha if lora_alpha is not None else self.rank * 2
        self._scaling = self.lora_alpha / self.rank
        if not mod_q:
            length -= 1
        if not mod_v:
            length -= 1
        if mod_k:
            length += 1
        if mod_ff:
            # mlp is 4 * embed_dim
            length += 4
        self.mod_q = mod_q
        self.mod_v = mod_v
        self.mod_k = mod_k
        self.mod_ff = mod_ff
        self.init_range = init_range
        self.init_prompts = init_prompts
        self.n_layer = n_layer
        self.n_head = n_head
        self.num_tasks = num_tasks
        self.eval_mode = eval_mode
        self.continual_mode = continual_mode
        self._setup_prompt()

    @property
    def scaling(self):
        return self._scaling

    def _setup_prompt(self):
        self.lora_a = nn.Parameter(
            torch.zeros(
                (self.pool_size, self.n_layer, self.length, self.embed_dim, self.rank)
            )
        )
        self.lora_b = nn.Parameter(
            torch.zeros(
                (self.pool_size, self.n_layer, self.length, self.rank, self.embed_dim)
            )
        )
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def extract_prompt(self, idx):
        """
        Args:
            idx: torch.Tensor. Indices to lookup.

        """
        # idx: [batch_size x 1]
        # lora_a_batched: [batch_size x n_layer x length x embed_dim x rank]
        # lora_b_batched: [batch_size x n_layer x length x rank x embed_dim]
        lora_a_batched = self.lora_a[idx].squeeze(1)
        lora_b_batched = self.lora_b[idx].squeeze(1)
        matrices = []
        idx_v, idx_k, idx_ff = (
            int(self.mod_q),
            sum([self.mod_q, self.mod_v]),
            sum([self.mod_q, self.mod_v, self.mod_k]),
        )
        for a, b in zip(
            lora_a_batched.split(dim=1, split_size=1),
            lora_b_batched.split(dim=1, split_size=1),
        ):
            a = a.squeeze(1)
            b = b.squeeze(1)
            matrices.append(
                (
                    (a[:, 0], b[:, 0]) if self.mod_q else None,  # queries
                    (a[:, idx_v], b[:, idx_v]) if self.mod_v else None,  # values
                    (a[:, idx_k], b[:, idx_k]) if self.mod_k else None,  # keys
                    (
                        a[:, idx_ff:].permute(0, 3, 2, 1).flatten(-2).transpose(2, 1),
                        b[:, idx_ff:].transpose(2, 1).flatten(-2),
                    )
                    if self.mod_ff
                    else None,  # ff
                    self.scaling,
                )
            )

        return matrices, {}

    def forward(self, x_embed, task_id=None, **kwargs):
        # Note: depending on whether self.n_layers is set, the dimensionality is different
        batched_prompt, prompt_stats = self.extract_prompt(task_id)
        if self.dropout_rate > 0:
            batched_prompt = self.add_dropout(batched_prompt)
        out = {
            "prompt_idx": task_id,
            "total_prompt_len": self.top_k * self.length,
            **prompt_stats,
        }
        return dict(prompt=batched_prompt, infos=out)

    def add_dropout(self, batched_prompt):
        return batched_prompt

    # Init e_t with e_{t-1}
    def set_task_id(self, task_id):
        super().set_task_id(task_id)
        task_id_val = task_id.cpu().item()
        if task_id_val > 0:
            self.lora_a[task_id].grad.zero_()
            self.lora_b[task_id].grad.zero_()
            self.lora_a[task_id] = self.lora_a[task_id - 1]
            self.lora_b[task_id] = self.lora_b[task_id - 1]
