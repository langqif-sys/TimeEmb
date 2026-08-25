import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ContinuousTimeEncoder(nn.Module):
    """
    Continuous Temporal Feature Encoder (inspired by CFPT TimeSter module).
    [Optimized]: Corrected Conv1d to apply along temporal axis (seq_len) instead of features axis.
    This reduces parameters by 10,000x and acts as a proper temporal local pattern extractor.
    """
    def __init__(self, time_dim, enc_in, seq_len, rda=4, rdb=4, ksize=5):
        super(ContinuousTimeEncoder, self).__init__()
        hidden_a = max(1, enc_in // rda)
        hidden_b = max(1, enc_in // rdb)
        
        self.fc1 = nn.Linear(time_dim, hidden_a)
        self.ln1 = nn.LayerNorm(hidden_a)
        self.relu1 = nn.ReLU()
        
        self.fc2 = nn.Linear(hidden_a, hidden_b)
        self.ln2 = nn.LayerNorm(hidden_b)
        self.relu2 = nn.ReLU()
        
        # Apply Conv1d over temporal axis: Channels = hidden_b, Spatial Length = seq_len
        self.conv = nn.Conv1d(
            in_channels=hidden_b, 
            out_channels=hidden_b, 
            kernel_size=ksize, 
            padding='same'
        )
        
        self.fc3 = nn.Linear(hidden_b, enc_in)

    def forward(self, x_mark_enc):
        # x_mark_enc: [B, seq_len, time_dim]
        out = self.fc1(x_mark_enc)   # [B, seq_len, hidden_a]
        out = self.ln1(out)
        out = self.relu1(out)
        
        out = self.fc2(out)   # [B, seq_len, hidden_b]
        out = self.ln2(out)
        out = self.relu2(out)
        
        # Permute to [B, hidden_b, seq_len] for temporal convolution
        out = out.permute(0, 2, 1)
        out = self.conv(out)
        # Permute back to [B, seq_len, hidden_b]
        out = out.permute(0, 2, 1)
        
        out = self.fc3(out)   # [B, seq_len, enc_in]
        
        # Transform to frequency domain: [B, enc_in, seq_len // 2 + 1]
        time_freq = torch.fft.rfft(out.permute(0, 2, 1), dim=2, norm='ortho')
        return time_freq.real

class InteractionBlock(nn.Module):
    """
    Invertible Neural Network (INN) Coupling Block (inspired by CFPT).
    [Optimized]: Introduced hidden bottleneck (hidden_dim=32) to map high-dimensional frequency spectrum.
    Forces low-rank representations to act as regularizers (noise filters) and saves 95%+ parameters.
    """
    def __init__(self, channels, hidden_dim=32):
        super(InteractionBlock, self).__init__()
        self.channels = channels
        self.channels_half = channels // 2

        # Map half-channels to bottleneck dimension and back
        def create_network():
            return nn.Sequential(
                nn.Linear(self.channels_half, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.channels_half)
            )

        self.s1 = create_network()
        self.t1 = create_network()
        self.s2 = create_network()
        self.t2 = create_network()
        self.scale_1 = nn.Parameter(torch.tensor(0.01))
        self.scale_2 = nn.Parameter(torch.tensor(0.01))

    def forward(self, x):
        # x: [B, enc_in, channels]
        x1, x2 = torch.split(x, self.channels_half, dim=-1)

        s1 = self.scale_1 * self.s1(x2)
        t1 = self.t1(x2)
        y1 = x1 * torch.exp(torch.clamp(s1, -5, 5)) + t1

        s2 = self.scale_2 * self.s2(y1)
        t2 = self.t2(y1)
        y2 = x2 * torch.exp(torch.clamp(s2, -5, 5)) + t2

        return torch.cat([y1, y2], dim=-1)


# =========================================================================
# TFPS Component 1: Patch Attention Temporal Encoder Layer
# =========================================================================

class PatchTSTEncoderLayer(nn.Module):
    """
    Patch-level Multi-Head Self-Attention Encoder Layer (from TFPS & PatchTST).
    """
    def __init__(self, d_model, n_heads, d_ff=256, dropout=0.0, attn_dropout=0.0, activation="gelu"):
        super(PatchTSTEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=attn_dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        
        self.norm1 = nn.BatchNorm1d(d_model)
        self.norm2 = nn.BatchNorm1d(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.act = nn.GELU() if activation == "gelu" else nn.ReLU()

    def forward(self, x):
        # x: [B * C, num_patches, d_model]
        attn_out, _ = self.self_attn(x, x, x)
        x = x + self.dropout1(attn_out)
        
        # Apply BatchNorm across token channels
        x = self.norm1(x.permute(0, 2, 1)).permute(0, 2, 1)
        
        # Feed-forward
        ff_out = self.linear2(self.dropout(self.act(self.linear1(x))))
        x = x + self.dropout2(ff_out)
        x = self.norm2(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


# =========================================================================
# TFPS Component 2: Time-Domain Patch Transformer Expert & Flatten Head
# =========================================================================

class TimePatchExpert(nn.Module):
    """
    Time-Domain Expert using Patching + Learnable Positional Encoding + Multi-Head Self-Attention + Flatten Head.
    Extracted and optimized from TFPS (NeurIPS 2024).
    """
    def __init__(self, seq_len, pred_len, patch_len=16, stride=8, padding_patch='end',
                 d_model=128, n_heads=8, d_ff=256, n_layers=1, dropout=0.0,
                 attn_dropout=0.0, head_dropout=0.0, individual=False, enc_in=7):
        super(TimePatchExpert, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.d_model = d_model
        self.enc_in = enc_in
        self.individual = individual

        # 1. Patching Configuration
        self.patch_num = int((seq_len - patch_len) / stride + 1)
        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            self.patch_num += 1

        # 2. Patch Linear Projection
        self.w_p = nn.Linear(patch_len, d_model)

        # 3. Learnable Positional Encoding
        self.w_pos = nn.Parameter(torch.zeros(1, self.patch_num, d_model))
        nn.init.uniform_(self.w_pos, -0.02, 0.02)
        self.pos_dropout = nn.Dropout(dropout)

        # 4. Multi-layer Transformer Encoder
        self.layers = nn.ModuleList([
            PatchTSTEncoderLayer(d_model=d_model, n_heads=n_heads, d_ff=d_ff,
                                 dropout=dropout, attn_dropout=attn_dropout)
            for _ in range(n_layers)
        ])

        # 5. Flatten Projection Head (Directly maps Patch feature to target horizon)
        self.head_nf = d_model * self.patch_num
        if self.individual:
            self.head = nn.ModuleList([
                nn.Sequential(
                    nn.Flatten(start_dim=-2),
                    nn.Linear(self.head_nf, pred_len),
                    nn.Dropout(head_dropout)
                ) for _ in range(enc_in)
            ])
        else:
            self.head = nn.Sequential(
                nn.Flatten(start_dim=-2),
                nn.Linear(self.head_nf, pred_len),
                nn.Dropout(head_dropout)
            )

    def forward(self, x):
        # x: [B, seq_len, C] (already RevIN normalized)
        B, L, C = x.shape
        x_perm = x.permute(0, 2, 1) # [B, C, seq_len]

        # Patching
        if self.padding_patch == 'end':
            x_perm = self.padding_patch_layer(x_perm)
        
        # [B, C, num_patches, patch_len]
        patches = x_perm.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        
        # Project patches to d_model & add position embedding
        # [B * C, num_patches, d_model]
        u = self.w_p(patches).reshape(B * C, self.patch_num, self.d_model)
        u = self.pos_dropout(u + self.w_pos)

        # Multi-Head Self-Attention layers
        for layer in self.layers:
            u = layer(u)

        # Reshape to [B, C, num_patches, d_model] -> permute to [B, C, d_model, num_patches]
        z = u.reshape(B, C, self.patch_num, self.d_model).permute(0, 1, 3, 2)

        # Flatten Projection Head
        if self.individual:
            out_list = []
            for i in range(C):
                out_i = self.head[i](z[:, i:i+1, :, :]) # [B, 1, pred_len]
                out_list.append(out_i)
            out_time = torch.cat(out_list, dim=1) # [B, C, pred_len]
        else:
            out_time = self.head(z) # [B, C, pred_len]

        return out_time.permute(0, 2, 1) # [B, pred_len, C]


# =========================================================================
# TFPS Component 3: Time-Frequency MoE Router (Adaptive Gated Softmax)
# =========================================================================

class TimeFreqMoERouter(nn.Module):
    """
    Time-Frequency Gated Mixture of Experts Router (inspired by TFPS SparseMoE).
    Dynamically learns adaptive gating routing weights between Time-Domain and Frequency-Domain experts.
    """
    def __init__(self, seq_len, enc_in, d_router=64, noise_scale=0.0):
        super(TimeFreqMoERouter, self).__init__()
        self.seq_len = seq_len
        self.enc_in = enc_in
        self.noise_scale = noise_scale

        # Router network: maps temporal pattern per channel to 2 expert logits [Time, Freq]
        self.router_mlp = nn.Sequential(
            nn.Linear(seq_len, d_router),
            nn.LayerNorm(d_router),
            nn.ReLU(),
            nn.Linear(d_router, 2)
        )
        
        # Optional noisy gating
        if noise_scale > 0:
            self.noise_linear = nn.Linear(seq_len, 2)

        # Balanced initial weights
        nn.init.zeros_(self.router_mlp[-1].weight)
        nn.init.zeros_(self.router_mlp[-1].bias)

    def forward(self, x):
        # x: [B, seq_len, C] -> permute to [B, C, seq_len]
        x_in = x.permute(0, 2, 1)
        logits = self.router_mlp(x_in) # [B, C, 2]

        if self.training and self.noise_scale > 0:
            noise_logits = self.noise_linear(x_in)
            noise = torch.randn_like(logits) * F.softplus(noise_logits) * self.noise_scale
            logits = logits + noise

        # Softmax gating weights across the 2 experts: [B, C, 2]
        gating_weights = F.softmax(logits, dim=-1)
        return gating_weights


# =========================================================================
# Dual-Domain Unified TimeEmb Model with Koopa, CFPT & TFPS Time MoE
# =========================================================================

class Model(nn.Module):
    """
    Upgraded TimeEmb Model incorporating:
    1. TimeEmb Static-Dynamic Decomposition (Embedding Bank + Frequency Filter)
    2. CFPT Continuous Temporal Encoding & Invertible Interaction (INN)
    3. TFPS Time-Domain Patch-Attention Expert & Flatten Head (NeurIPS 2024)
    4. Time-Frequency MoE Dynamic Gated Router
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.use_revin = configs.use_revin
        self.use_day_index = configs.use_day_index
        self.use_hour_index = configs.use_hour_index
        self.scale = 0.02
        self.emb_len_hour = configs.hour_length
        self.emb_len_day = configs.day_length

        # Options for Dual-Domain MoE
        self.use_time_expert = getattr(configs, 'use_time_expert', 1)
        self.use_freq_expert = getattr(configs, 'use_freq_expert', 1)
        self.use_moe_router = getattr(configs, 'use_moe_router', 1)

        # -----------------------------------------------------------------
        # 1. Frequency-Domain Expert Pipeline (TimeEmb + CFPT + Koopa)
        # -----------------------------------------------------------------
        if self.use_freq_expert:
            layers = [
                nn.Linear(self.seq_len, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, self.pred_len)
            ]
            self.model = nn.Sequential(*layers)

            # Legacy Lookup Table Embeddings
            self.emb_hour = nn.Parameter(torch.zeros(self.emb_len_hour, self.enc_in, self.seq_len // 2 + 1), requires_grad=True)
            self.emb_day = nn.Parameter(torch.zeros(self.emb_len_day, self.enc_in, self.seq_len // 2 + 1), requires_grad=True)

            # CFPT Continuous Temporal Feature Encoder
            self.time_dim = getattr(configs, 'time_dim', 4)
            self.continuous_time_enc = ContinuousTimeEncoder(
                time_dim=self.time_dim,
                enc_in=self.enc_in,
                seq_len=self.seq_len,
                rda=4,
                rdb=4
            )

            # CFPT Invertible Frequency Interaction (INN)
            self.cutoff_ratio = getattr(configs, 'cutoff_ratio', 0.2)
            self.cutoff_idx = max(1, int((self.seq_len // 2 + 1) * self.cutoff_ratio))
            
            freq_len = self.seq_len // 2 + 1
            inn_channels = (freq_len * 2) if (freq_len * 2) % 2 == 0 else (freq_len * 2 + 1)
            self.inn_block = InteractionBlock(channels=inn_channels, hidden_dim=32)

            # Warm-Start Residual Scale Parameters
            self.gamma_time = nn.Parameter(torch.full((1,), 0.01), requires_grad=True)
            self.alpha_inn = nn.Parameter(torch.full((1,), 0.01), requires_grad=True)

            self.w_trend = nn.Parameter(self.scale * torch.randn(1, self.seq_len))
            self.w_fluct = nn.Parameter(self.scale * torch.randn(1, self.seq_len))

        # -----------------------------------------------------------------
        # 2. Time-Domain Patch Transformer Expert (TFPS NeurIPS 2024)
        # -----------------------------------------------------------------
        if self.use_time_expert:
            patch_len = getattr(configs, 'patch_len', 16)
            stride = getattr(configs, 'stride', 8)
            padding_patch = getattr(configs, 'padding_patch', 'end')
            time_d_model = getattr(configs, 'time_d_model', 128)
            time_n_heads = getattr(configs, 'time_n_heads', 8)
            time_d_ff = getattr(configs, 'time_d_ff', 256)
            time_e_layers = getattr(configs, 'time_e_layers', 1)
            dropout = getattr(configs, 'dropout', 0.0)
            fc_dropout = getattr(configs, 'fc_dropout', 0.0)
            head_dropout = getattr(configs, 'head_dropout', 0.0)
            individual = getattr(configs, 'individual', 0)

            self.time_expert = TimePatchExpert(
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                patch_len=patch_len,
                stride=stride,
                padding_patch=padding_patch,
                d_model=time_d_model,
                n_heads=time_n_heads,
                d_ff=time_d_ff,
                n_layers=time_e_layers,
                dropout=dropout,
                attn_dropout=fc_dropout,
                head_dropout=head_dropout,
                individual=bool(individual),
                enc_in=self.enc_in
            )

        # -----------------------------------------------------------------
        # 3. Time-Frequency MoE Gated Router
        # -----------------------------------------------------------------
        if self.use_time_expert and self.use_freq_expert and self.use_moe_router:
            router_noise = getattr(configs, 'router_noise', 0.0)
            self.router = TimeFreqMoERouter(
                seq_len=self.seq_len,
                enc_in=self.enc_in,
                d_router=64,
                noise_scale=router_noise
            )

    def _forward_freq(self, x, hour_index=None, day_index=None, x_mark_enc=None):
        # x: [B, seq_len, enc_in] (normalized)
        x_perm = x.permute(0, 2, 1) # [B, enc_in, seq_len]
        x_fft = torch.fft.rfft(x_perm, dim=2, norm='ortho')
        w_trend = torch.fft.rfft(self.w_trend, dim=1, norm='ortho')
        w_fluct = torch.fft.rfft(self.w_fluct, dim=1, norm='ortho')
        x_freq_real = x_fft.real
        x_freq_imag = x_fft.imag

        # Extract Time-Invariant Component
        emb_time_real = 0
        if x_mark_enc is not None:
            if x_mark_enc.shape[-1] != self.time_dim:
                self.time_dim = x_mark_enc.shape[-1]
                self.continuous_time_enc = ContinuousTimeEncoder(
                    time_dim=self.time_dim,
                    enc_in=self.enc_in,
                    seq_len=self.seq_len,
                    rda=4,
                    rdb=4
                ).to(x.device)
            emb_dynamic = self.continuous_time_enc(x_mark_enc)
            emb_time_real = emb_time_real + self.gamma_time * emb_dynamic

        if self.use_hour_index and hour_index is not None:
            emb_hour = self.emb_hour[hour_index % self.emb_len_hour]
            emb_time_real = emb_time_real + emb_hour

        if self.use_day_index and day_index is not None:
            emb_day = self.emb_day[day_index % self.emb_len_day]
            emb_time_real = emb_time_real + emb_day

        x_freq_real = x_freq_real - emb_time_real
        x_dynamic = torch.complex(x_freq_real, x_freq_imag)

        # INN Coupling Interaction with Residual Gate
        freq_concat = torch.cat([x_dynamic.real, x_dynamic.imag], dim=-1)
        freq_interacted = self.inn_block(freq_concat)
        freq_real_inn, freq_imag_inn = torch.chunk(freq_interacted, 2, dim=-1)
        x_dynamic_inn = torch.complex(freq_real_inn, freq_imag_inn)

        # Controlled by warm-started alpha_inn
        x_dynamic = x_dynamic + self.alpha_inn * (x_dynamic_inn - x_dynamic)

        # Build frequency mask for trend vs fluctuation
        mask_trend = torch.zeros_like(x_dynamic)
        mask_trend[:, :, :self.cutoff_idx] = 1.0

        x_local_trend = x_dynamic * mask_trend
        x_local_fluct = x_dynamic * (1 - mask_trend)

        y_local_trend = x_local_trend * w_trend
        y_local_fluct = x_local_fluct * w_fluct

        y_freq_comp = y_local_trend + y_local_fluct
        y_real = y_freq_comp.real + emb_time_real
        y_freq_imag = y_freq_comp.imag

        y_freq_full = torch.complex(y_real, y_freq_imag)
        y_time_rec = torch.fft.irfft(y_freq_full, n=self.seq_len, dim=2, norm="ortho")
        y_freq_out = self.model(y_time_rec).permute(0, 2, 1) # [B, pred_len, enc_in]
        return y_freq_out

    def forward(self, x, hour_index=None, day_index=None, x_mark_enc=None):
        # RevIN Normalization
        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

        # Forward Expert Branches
        if self.use_time_expert and self.use_freq_expert:
            y_time = self.time_expert(x) # [B, pred_len, C]
            y_freq = self._forward_freq(x, hour_index, day_index, x_mark_enc) # [B, pred_len, C]

            if self.use_moe_router:
                # gating_weights: [B, C, 2] -> permute to [B, 1, C, 2] for broadcast
                gating = self.router(x)
                g_time = gating[:, :, 0:1].permute(0, 2, 1) # [B, 1, C]
                g_freq = gating[:, :, 1:2].permute(0, 2, 1) # [B, 1, C]
                y = g_time * y_time + g_freq * y_freq
            else:
                y = 0.5 * y_time + 0.5 * y_freq
        elif self.use_time_expert:
            y = self.time_expert(x)
        else:
            y = self._forward_freq(x, hour_index, day_index, x_mark_enc)

        # RevIN Denormalization
        if self.use_revin:
            y = y * torch.sqrt(seq_var) + seq_mean

        return y