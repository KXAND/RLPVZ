"""CNN-based Q-Network for DDQN with dual-branch grid processing.

3x3 branch: captures local spatial patterns (adjacent cell interactions).
1x9 branch: captures full-row patterns (entire row defense/offense balance).
Global features (sun + cooldowns) bypass CNN and merge after grid encoding.
"""

import numpy as np
import torch
import torch.nn as nn


class CNNQNetwork(nn.Module):
    def __init__(self, env, learning_rate=1e-3, device="cpu",
                 hidden_sizes=None, n_inputs_override=None,
                 create_optimizer=True):
        super().__init__()
        self.device = device
        self.rows = env.rows
        self.cols = env.cols
        self.num_cards = env.num_cards
        self.n_outputs = env.action_space.n
        self.actions = np.arange(env.action_space.n)
        self.learning_rate = learning_rate

        # ── derived dims ──────────────────────────────────────────
        n_grid_channels = self.num_cards + 1 + 2   # one-hot(11) + plantHP(1) + zombieHP(1)
        n_global = 1 + self.num_cards               # sun(1) + cooldowns(10)
        self._n_grid_channels = n_grid_channels
        self._n_global = n_global
        self._grid_size = self.rows * self.cols

        # ── 3×3 spatial branch ────────────────────────────────────
        self.branch_3x3 = nn.Sequential(
            nn.Conv2d(n_grid_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        _3x3_flat = self.rows * self.cols * 128  # 5×9×128 = 5760

        # ── 1×9 row branch ────────────────────────────────────────
        self.branch_1x9 = nn.Sequential(
            nn.Conv2d(n_grid_channels, 64, kernel_size=(1, 9), padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        _1x9_flat = 3 * 1 * 128  # (3,1,128) = 384

        # ── grid merge ────────────────────────────────────────────
        grid_out = _3x3_flat + _1x9_flat  # 6144
        self.grid_proj = nn.Sequential(
            nn.Linear(grid_out, 896),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        # ── global branch ─────────────────────────────────────────
        self.global_proj = nn.Sequential(
            nn.Linear(n_global, 64),
            nn.ReLU(inplace=True),
        )

        # ── output head ───────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(896 + 64, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.n_outputs),
        )

        if device == "cuda":
            self.cuda()

        self.optimizer = None
        if create_optimizer:
            self.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.learning_rate,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 596) flat observation vector."""
        bsz = x.shape[0]

        # Split: first n_global dims = global, rest = grid
        glob = x[:, :self._n_global]                         # (B, 11)
        grid_flat = x[:, self._n_global:]                     # (B, 585)

        # Reshape grid to (B, rows, cols, channels) -> (B, C, H, W)
        grid = grid_flat.view(bsz, self.rows, self.cols, self._n_grid_channels)
        grid = grid.permute(0, 3, 1, 2).contiguous()          # (B, 13, 5, 9)

        # CNN branches
        feat_3x3 = self.branch_3x3(grid).reshape(bsz, -1)     # (B, 5760)
        feat_1x9 = self.branch_1x9(grid).reshape(bsz, -1)     # (B, 384)

        # Merge grid features
        grid_feat = self.grid_proj(torch.cat([feat_3x3, feat_1x9], dim=1))  # (B, 896)

        # Global features
        glob_feat = self.global_proj(glob)                    # (B, 64)

        # Full merge → Q-values
        return self.head(torch.cat([grid_feat, glob_feat], dim=1))  # (B, 451)

    # ── DDQN interface (same as QNetwork) ──────────────────────────
    def decide_action(self, state, mask, epsilon):
        if np.random.random() < epsilon:
            valid_actions = self.actions[np.asarray(mask, dtype=bool)]
            return np.random.choice(valid_actions)
        return self.get_greedy_action(state, mask)

    def get_greedy_action(self, state, mask):
        qvals = self.get_qvals(state)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=qvals.device)
        qvals = qvals.clone()
        qvals[~mask_t] = qvals.min()
        return torch.max(qvals, dim=-1)[1].item()

    def get_qvals(self, state):
        if isinstance(state, (list, tuple)):
            state = np.array(state)
        if state.ndim == 1:
            state = state[np.newaxis, :]       # add batch dim
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        qvals = self.forward(state_t)
        if qvals.shape[0] == 1:
            qvals = qvals.squeeze(0)
        return qvals
