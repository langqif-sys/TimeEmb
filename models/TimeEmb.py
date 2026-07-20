import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ContinuousTimeEncoder(nn.Module):
    """
    Continuous Temporal Feature Encoder (inspired by CFPT TimeSter module).
    Encodes continuous time markers (month, day, weekday, hour, minute)
    into dynamic time-invariant spectrum representations.
    """
    def __init__(self, time_dim, enc_in, seq_len, rda=4, rdb=1, ksize=5):
        super(ContinuousTimeEncoder, self).__init__()
        hidden_a = max(1, enc_in // rda)
        hidden_b = max(1, enc_in // rdb)
        self.time_enc = nn.Sequential(
            nn.Linear(time_dim, hidden_a),
            nn.LayerNorm(hidden_a),
            nn.ReLU(),
            nn.Linear(hidden_a, hidden_b),
            nn.LayerNorm(hidden_b),
            nn.ReLU(),
            nn.Conv1d(in_channels=seq_len, out_channels=seq_len, kernel_size=ksize, padding='same'),
            nn.Linear(hidden_b, enc_in)
        )

    def forward(self, x_mark_enc):
        # x_mark_enc: [B, seq_len, time_dim]
        # output: [B, seq_len, enc_in]
        time_embed = self.time_enc(x_mark_enc)
        # Transform to frequency domain: [B, enc_in, seq_len // 2 + 1]
        time_freq = torch.fft.rfft(time_embed.permute(0, 2, 1), dim=2, norm='ortho')
        return time_freq.real

class InteractionBlock(nn.Module):
    """
    Invertible Neural Network (INN) Coupling Block (inspired by CFPT).
    Allows lossless non-linear interaction between spectrum components.
    """
    def __init__(self, channels):
        super(InteractionBlock, self).__init__()
        self.channels = channels
        self.channels_half = channels // 2

        def create_network():
            return nn.Sequential(
                nn.Linear(self.channels_half, self.channels),
                nn.LayerNorm(self.channels),
                nn.ReLU(),
                nn.Linear(self.channels, self.channels_half)
            )

        self.s1 = create_network()
        self.t1 = create_network()
        self.s2 = create_network()
        self.t2 = create_network()
        self.scale_1 = nn.Parameter(torch.tensor(0.1))
        self.scale_2 = nn.Parameter(torch.tensor(0.1))

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

class Model(nn.Module):
    """
    Upgraded TimeEmb Model incorporating CFPT Continuous Temporal Encoding & Invertible Interaction.
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

        # 2. Legacy Lookup Table Embeddings (Backward Compatible)
        self.emb_hour = nn.Parameter(torch.zeros(self.emb_len_hour, self.enc_in, self.seq_len // 2 + 1), requires_grad=True)
        self.emb_day = nn.Parameter(torch.zeros(self.emb_len_day, self.enc_in, self.seq_len // 2 + 1), requires_grad=True)

        # 3. CFPT Continuous Temporal Feature Encoder (Innovation Point 2)
        self.time_dim = getattr(configs, 'time_dim', 4)
        self.continuous_time_enc = ContinuousTimeEncoder(
            time_dim=self.time_dim,
            enc_in=self.enc_in,
            seq_len=self.seq_len
        )

        # 4. CFPT Invertible Frequency Interaction (INN) (Innovation Point 2)
        self.cutoff_idx = max(1, int((self.seq_len // 2 + 1) * 0.2))
        freq_len = self.seq_len // 2 + 1
        inn_channels = (freq_len * 2) if (freq_len * 2) % 2 == 0 else (freq_len * 2 + 1)
        self.inn_block = InteractionBlock(channels=inn_channels)

        self.w_trend = nn.Parameter(self.scale * torch.randn(1, self.seq_len))
        self.w_fluct = nn.Parameter(self.scale * torch.randn(1, self.seq_len))

    def forward(self, x, hour_index=None, day_index=None, x_mark_enc=None):
        # x: (batch_size, seq_len, enc_in)
        # hour_index: (batch_size,), day_index: (batch_size,)
        # x_mark_enc: (batch_size, seq_len, time_dim)

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
            # Dynamically adapt time_dim if input mark shape differs from default
            if x_mark_enc.shape[-1] != self.time_dim:
                self.time_dim = x_mark_enc.shape[-1]
                self.continuous_time_enc = ContinuousTimeEncoder(
                    time_dim=self.time_dim,
                    enc_in=self.enc_in,
                    seq_len=self.seq_len
                ).to(x.device)
            emb_dynamic = self.continuous_time_enc(x_mark_enc) # [B, enc_in, freq_len]
            emb_time_real = emb_time_real + emb_dynamic

        if self.use_hour_index and hour_index is not None:
            emb_hour = self.emb_hour[hour_index % self.emb_len_hour]
            emb_time_real = emb_time_real + emb_hour

        if self.use_day_index and day_index is not None:
            emb_day = self.emb_day[day_index % self.emb_len_day]
            emb_time_real = emb_time_real + emb_day

        x_freq_real = x_freq_real - emb_time_real
        x_dynamic = torch.complex(x_freq_real, x_freq_imag)

        # INN Coupling Interaction for lossless frequency disentanglement
        B, C, F = x_dynamic.shape
        freq_concat = torch.cat([x_dynamic.real, x_dynamic.imag], dim=-1)
        freq_interacted = self.inn_block(freq_concat)
        freq_real, freq_imag = torch.chunk(freq_interacted, 2, dim=-1)
        x_dynamic = torch.complex(freq_real, freq_imag)

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
        y = self.model(y).permute(0, 2, 1)

        if self.use_revin:
            y = y * torch.sqrt(seq_var) + seq_mean

        return y