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
        # Using rdb=4 to further reduce feature dimension for large enc_in (e.g. Electricity)
        hidden_a = max(1, enc_in // rda)
        hidden_b = max(1, enc_in // rdb)
        
        self.fc1 = nn.Linear(time_dim, hidden_a)
        self.ln1 = nn.LayerNorm(hidden_a)
        self.relu1 = nn.ReLU()
        
        self.fc2 = nn.Linear(hidden_a, hidden_b)
        self.ln2 = nn.LayerNorm(hidden_b)
        self.relu2 = nn.ReLU()
        
        # Correctly apply Conv1d over temporal axis: Channels = hidden_b, Spatial Length = seq_len
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

class LocalTransientExtractor(nn.Module):
    """
    Multi-Scale Time-Domain Local Transient Extractor.
    Captures localized abrupt variations and impulse residuals with multi-scale depthwise convolutions.
    Features zero-initialization on final temporal projection for seamless warm-start scale alignment.
    """
    def __init__(self, seq_len, pred_len, hidden_dim=32):
        super(LocalTransientExtractor, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        d_ch = max(1, hidden_dim // 4)
        self.conv3 = nn.Conv1d(1, d_ch, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(1, d_ch, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(1, d_ch, kernel_size=7, padding=3)
        self.conv1 = nn.Conv1d(1, d_ch, kernel_size=1)

        self.fuse = nn.Sequential(
            nn.BatchNorm1d(d_ch * 4),
            nn.GELU(),
            nn.Conv1d(d_ch * 4, 1, kernel_size=1)
        )

        self.temporal_proj = nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, pred_len)
        )

        # Zero-Init: Guarantees 0-residual output at initial step (Scale Alignment)
        nn.init.zeros_(self.temporal_proj[-1].weight)
        nn.init.zeros_(self.temporal_proj[-1].bias)

    def forward(self, x_diff_signed):
        # x_diff_signed: [B, enc_in, seq_len]
        B, C, L = x_diff_signed.shape
        x_in = x_diff_signed.reshape(B * C, 1, L)

        c3 = self.conv3(x_in)
        c5 = self.conv5(x_in)
        c7 = self.conv7(x_in)
        c1 = self.conv1(x_in)

        feat = torch.cat([c3, c5, c7, c1], dim=1) # [B*C, d_ch*4, L]
        feat = self.fuse(feat).squeeze(1)          # [B*C, L]

        out = self.temporal_proj(feat)             # [B*C, pred_len]
        return out.reshape(B, C, self.pred_len).permute(0, 2, 1) # [B, pred_len, enc_in]


class TransientAwareGate(nn.Module):
    """
    Transient-Aware Dynamic Confidence Gating.
    Maps local differential fluctuation magnitude into bounded confidence weights in (0, 1).
    """
    def __init__(self, seq_len, pred_len, hidden_dim=32):
        super(TransientAwareGate, self).__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pred_len),
            nn.Sigmoid()
        )
        # Negative bias init: starts gentle (~0.11), zero-weight init
        nn.init.constant_(self.gate_net[-2].bias, -2.0)
        nn.init.zeros_(self.gate_net[-2].weight)

    def forward(self, x_diff_mag):
        # x_diff_mag: [B, enc_in, seq_len]
        gate = self.gate_net(x_diff_mag) # [B, enc_in, pred_len]
        return gate.permute(0, 2, 1)     # [B, pred_len, enc_in]

class Model(nn.Module):
    """
    Upgraded TimeEmb Model incorporating CFPT Continuous Temporal Encoding,
    Invertible Interaction, and Time-Domain Transient-Aware Complementary Gating.
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

        # 1. Base Prediction Backbone
        layers = [
            nn.Linear(self.seq_len, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.pred_len)
        ]
        self.model = nn.Sequential(*layers)

        # 2. Legacy Lookup Table Embeddings
        self.emb_hour = nn.Parameter(torch.zeros(self.emb_len_hour, self.enc_in, self.seq_len // 2 + 1), requires_grad=True)
        self.emb_day = nn.Parameter(torch.zeros(self.emb_len_day, self.enc_in, self.seq_len // 2 + 1), requires_grad=True)

        # 3. CFPT Continuous Temporal Feature Encoder (Optimized)
        self.time_dim = getattr(configs, 'time_dim', 4)
        # Using rdb=4 for better bottlenecking
        self.continuous_time_enc = ContinuousTimeEncoder(
            time_dim=self.time_dim,
            enc_in=self.enc_in,
            seq_len=self.seq_len,
            rda=4,
            rdb=4
        )

        # 4. CFPT Invertible Frequency Interaction (INN) (Optimized)
        # Support adaptive cutoff ratio via configs
        self.cutoff_ratio = getattr(configs, 'cutoff_ratio', 0.2)
        self.cutoff_idx = max(1, int((self.seq_len // 2 + 1) * self.cutoff_ratio))
        
        freq_len = self.seq_len // 2 + 1
        inn_channels = (freq_len * 2) if (freq_len * 2) % 2 == 0 else (freq_len * 2 + 1)
        self.inn_block = InteractionBlock(channels=inn_channels, hidden_dim=32)

        # 5. Warm-Start Residual Scale Parameters (Crucial for learning rate flow)
        # Initialized to 0.01 instead of absolute 0.0 to enable stable early gradient flow
        self.gamma_time = nn.Parameter(torch.full((1,), 0.01), requires_grad=True)
        self.alpha_inn = nn.Parameter(torch.full((1,), 0.01), requires_grad=True)

        self.w_trend = nn.Parameter(self.scale * torch.randn(1, self.seq_len))
        self.w_fluct = nn.Parameter(self.scale * torch.randn(1, self.seq_len))

        # 6. Time-Domain Local Transient & Dynamic Gating Branch (Scale & Temporal Alignment)
        self.use_transient = getattr(configs, 'use_transient', 1)
        if self.use_transient:
            self.transient_extractor = LocalTransientExtractor(
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                hidden_dim=32
            )
            self.transient_gate = TransientAwareGate(
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                hidden_dim=32
            )

    def forward(self, x, hour_index=None, day_index=None, x_mark_enc=None):
        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

        x = x.permute(0, 2, 1) # [B, enc_in, seq_len]
        x_fft = torch.fft.rfft(x, dim=2, norm='ortho')
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
        B, C, F = x_dynamic.shape
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

        y = y_local_trend + y_local_fluct

        y_real = y.real + emb_time_real
        y_freq_imag = y.imag

        y_freq = torch.complex(y_real, y_freq_imag)
        y = torch.fft.irfft(y_freq, n=self.seq_len, dim=2, norm="ortho")
        y_freq_out = self.model(y).permute(0, 2, 1) # [B, pred_len, enc_in]

        # Time-Domain Transient-Aware Complementary Gating (Scale & Direction Aligned)
        if self.use_transient:
            # Construct signed difference (direction) and magnitude (volatility energy)
            # Note: x is already [B, enc_in, seq_len]
            x_diff_signed = torch.zeros_like(x)
            x_diff_signed[:, :, 1:] = x[:, :, 1:] - x[:, :, :-1]
            x_diff_mag = torch.abs(x_diff_signed)

            y_time_res = self.transient_extractor(x_diff_signed)
            gate = self.transient_gate(x_diff_mag)

            y_final = y_freq_out + gate * y_time_res
        else:
            y_final = y_freq_out

        if self.use_revin:
            y_final = y_final * torch.sqrt(seq_var) + seq_mean

        return y_final