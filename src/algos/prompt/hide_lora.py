import math
import torch
import torch.nn as nn
from .l2p import Prompt


class HiDeLoRAPool(Prompt):
    def __init__(
        self,
        length=2,
        n_layer=None,
        top_k=1,
        dropout_rate=0.0,
        init_prompts="zeros",
        rank=4,
        mod_q=True,
        mod_v=True,
        mod_k=False,
        mod_ff=True,
        lora_alpha=None,
        log_mod_stats=False,
        **kwargs,
    ):
        self.log_mod_stats = log_mod_stats
        self.rank = rank
        self.mod_v = mod_v
        self.mod_q = mod_q
        self.mod_k = mod_k
        self.mod_ff = mod_ff
        self.lora_alpha = lora_alpha if lora_alpha is not None else self.rank * 2
        self._scaling = self.lora_alpha / self.rank
        self.n_layer = n_layer
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
        super().__init__(
            length=length,
            top_k=top_k,
            dropout_rate=dropout_rate,
            init_prompts=init_prompts,
            **kwargs,
        )

    @property
    def scaling(self):
        return self._scaling

    def _setup_prompt(self):
        self._create_prompt()
        self._reset_prompt()

    def _create_prompt(self):
        attributes = ["k_lora", "v_lora"]
        for attr_name in attributes:
            setattr(
                self,
                attr_name + "_A",
                nn.Parameter(
                    torch.zeros((self.pool_size, self.n_layer, self.dim, self.r))
                ),
            )
            setattr(
                self,
                attr_name + "_B",
                nn.Parameter(
                    torch.zeros((self.pool_size, self.n_layer, self.r, self.dim))
                ),
            )

        self.q_lora_A = torch.zeros((self.pool_size, self.n_layer, self.dim, self.r))
        self.q_lora_B = torch.zeros((self.pool_size, self.n_layer, self.r, self.dim))

        self.ff_lora_A = torch.zeros(
            (self.pool_size, self.n_layer, 4, self.dim, self.r)
        )
        self.ff_lora_B = torch.zeros(
            (self.pool_size, self.n_layer, 4, self.r, self.dim)
        )

    def _reset_prompt(self):
        params = ["k_lora_A", "k_lora_B", "v_lora_A", "v_lora_B"]
        for param_name in params:
            param = getattr(self, param_name)
            if isinstance(param, nn.Parameter):
                if param_name.endswith("_A"):
                    p, d, _, _ = param.shape
                    for i in range(p):
                        for j in range(d):
                            nn.init.kaiming_uniform_(param[i][j], a=math.sqrt(5))
                else:
                    nn.init.zeros_(param)

    def extract_prompt(self, idx):
        """
        Args:
            idx: torch.Tensor. Indices to lookup.

        """
        # idx: [batch_size x 1]
        # lora_a_batched: [batch_size x n_layer x length x rank x embed_dim]
        # lora_b_batched: [batch_size x n_layer x length x embed_dim x rank]
        matrices = dict()
        params = ["k_lora_A", "k_lora_B", "v_lora_A", "v_lora_B"]
        for param_name in params:
            param = getattr(self, param_name)
            if isinstance(param, nn.Parameter):
                matrices[param_name] = param[idx].squeeze(1)
        matrices["scaling"] = self.scaling

        return matrices, {}

    def to_device(self, device):
        params = [
            "q_lora_A",
            "q_lora_B",
            "k_lora_A",
            "k_lora_B",
            "v_lora_A",
            "v_lora_B",
        ]
        for param_name in params:
            if not isinstance(getattr(self, param_name), nn.Parameter):
                setattr(self, param_name, getattr(self, param_name).to(device))

    def forward(
        self,
        x_embed,
        task_id=None,
        depth_id=None,
        train=False,
    ):
        out = dict()
        self.to_device(x_embed.device)
        if train:
            assert isinstance(task_id, int)
            q = self.q_lora_A[task_id][depth_id] @ self.q_lora_B[task_id][depth_id]
            k = self.k_lora_A[task_id][depth_id] @ self.k_lora_B[task_id][depth_id]
            v = self.v_lora_A[task_id][depth_id] @ self.v_lora_B[task_id][depth_id]
            w = (
                torch.cat(
                    [q.to(x_embed.device), k.to(x_embed.device), v.to(x_embed.device)],
                    dim=-1,
                )
                * self.scaling
            )
            out["lora_value"] = torch.einsum("bld,dz->blz", x_embed, w)
            return out

        else:
            assert isinstance(task_id, list) or isinstance(task_id, torch.Tensor)
            q = torch.bmm(
                self.q_lora_A[task_id, depth_id], self.q_lora_B[task_id, depth_id]
            )
            k = torch.bmm(
                self.k_lora_A[task_id, depth_id], self.k_lora_B[task_id, depth_id]
            )
            v = torch.bmm(
                self.v_lora_A[task_id, depth_id], self.v_lora_B[task_id, depth_id]
            )
            w = (
                torch.cat(
                    [q.to(x_embed.device), k.to(x_embed.device), v.to(x_embed.device)],
                    dim=-1,
                )
                * self.scaling
            )
            out["lora_value"] = torch.bmm(x_embed, w)  # B x L x 3dim
        return out

    def add_dropout(self, batched_prompt):
        return batched_prompt
