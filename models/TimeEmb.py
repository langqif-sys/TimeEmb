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

try:
    from torch.nn.utils.parametrizations import weight_norm as wn
except ImportError:
    from torch.nn.utils import weight_norm as wn

class MLP_bottle(nn.Module):
    def __init__(self, input_len, output_len, bottleneck, bias=True):
        super(MLP_bottle, self).__init__()
        bottleneck = max(1, bottleneck)
        self.linear1 = nn.Sequential(
            wn(nn.Linear(input_len, bottleneck, bias=bias)),
            nn.ReLU(),
            wn(nn.Linear(bottleneck, bottleneck, bias=bias))
        )
        self.linear2 = nn.Sequential(
            wn(nn.Linear(bottleneck, bottleneck)),
            nn.ReLU(),
            wn(nn.Linear(bottleneck, output_len))
        )
        self.skip = wn(nn.Linear(input_len, bottleneck, bias=bias))
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.linear1(x) + self.skip(x))
        x = self.linear2(x)
        return x

class channel_AutoCorrelationLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, dropout=0.0):
        super(channel_AutoCorrelationLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.query_projection = wn(nn.Linear(d_model, d_keys * n_heads))
        self.key_projection = wn(nn.Linear(d_model, d_keys * n_heads))
        self.value_projection = wn(nn.Linear(d_model, d_values * n_heads))
        self.out_projection = wn(nn.Linear(d_values * n_heads, d_model))
        self.n_heads = n_heads
        self.scale = d_keys ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        B, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1).permute(0, 2, 1, 3)
        keys = self.key_projection(keys).view(B, S, H, -1).permute(0, 2, 1, 3)
        values = self.value_projection(values).view(B, S, H, -1).permute(0, 2, 1, 3)

        dots = torch.matmul(queries, keys.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, values)
        out = out.permute(0, 2, 1, 3).reshape(B, L, -1)
        return self.out_projection(out), attn

class BCAB(nn.Module):
    """
    Bidirectional Cross-Attention Block (inspired by BasisFormer).
    Enables mutual knowledge exchange between time-domain series and basis prototypes.
    """
    def __init__(self, d_model, heads=4, d_ff=None, dropout=0.1):
        super(BCAB, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.cross_attention_basis = channel_AutoCorrelationLayer(d_model, heads, dropout=dropout)
        self.conv1_basis = wn(nn.Linear(d_model, d_ff))
        self.conv2_basis = wn(nn.Linear(d_ff, d_model))
        self.dropout_basis = nn.Dropout(dropout)
        self.activation_basis = nn.ReLU()

        self.cross_attention_ts = channel_AutoCorrelationLayer(d_model, heads, dropout=dropout)
        self.conv1_ts = wn(nn.Linear(d_model, d_ff))
        self.conv2_ts = wn(nn.Linear(d_ff, d_model))
        self.dropout_ts = nn.Dropout(dropout)
        self.activation_ts = nn.ReLU()

        self.layer_norm11 = nn.LayerNorm(d_model)
        self.layer_norm12 = nn.LayerNorm(d_model)
        self.layer_norm21 = nn.LayerNorm(d_model)
        self.layer_norm22 = nn.LayerNorm(d_model)

    def forward(self, basis, series):
        # 1. Basis attends to series
        basis_add, basis_attn = self.cross_attention_basis(basis, series, series)
        basis_out = self.layer_norm11(basis + self.dropout_basis(basis_add))
        y_basis = self.dropout_basis(self.conv2_basis(self.dropout_basis(self.activation_basis(self.conv1_basis(basis_out)))))
        basis_out = self.layer_norm12(basis_out + y_basis)

        # 2. Series attends to basis
        series_add, series_attn = self.cross_attention_ts(series, basis, basis)
        series_out = self.layer_norm21(series + self.dropout_ts(series_add))
        y_ts = self.dropout_ts(self.conv2_ts(self.dropout_ts(self.activation_ts(self.conv1_ts(series_out)))))
        series_out = self.layer_norm22(series_out + y_ts)

        return basis_out, series_out, basis_attn, series_attn

class last_layer(nn.Module):
    """
    Computes cross-correlation similarity score matrix between series and bases.
    Acts as the feature selection coefficient matrix (Soft-KNN prototype selection).
    """
    def __init__(self, d_model, n_heads, d_keys=None):
        super(last_layer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        self.query_projection = wn(nn.Linear(d_model, d_keys * n_heads))
        self.key_projection = wn(nn.Linear(d_model, d_keys * n_heads))
        self.n_heads = n_heads
        self.scale = d_keys ** -0.5

    def forward(self, queries, keys):
        B, L, _ = queries.shape
        B, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1).permute(0, 2, 1, 3)
        keys = self.key_projection(keys).view(B, S, H, -1).permute(0, 2, 1, 3)
        dots = torch.matmul(queries, keys.transpose(-1, -2)) * self.scale
        return dots

class Coefnet(nn.Module):
    """
    BasisFormer Coefnet: Stacks BCAB layers and extracts dynamic selection coefficients.
    """
    def __init__(self, blocks, d_model, heads):
        super(Coefnet, self).__init__()
        self.layers = nn.ModuleList([BCAB(d_model, heads) for _ in range(blocks)])
        self.last_layer = last_layer(d_model, heads)

    def forward(self, basis, series):
        for layer in self.layers:
            basis, series, _, _ = layer(basis, series)
        coef = self.last_layer(series, basis)  # [B, heads, C, N]
        return coef

class Model(nn.Module):
    """
    Upgraded TimeEmb Model incorporating CFPT, Koopa, and BasisFormer Time-Domain Basis Selection.
    Features dual-stream frequency-time extraction, dynamic basis selection, and warm-start residual gating.
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

        # 6. BasisFormer Time-Domain Basis Selection Branch
        self.use_basis = getattr(configs, 'use_basis', 1)
        if self.use_basis:
            self.basis_nums = getattr(configs, 'basis_nums', 16)
            self.basis_heads = getattr(configs, 'basis_heads', 4)
            self.basis_blocks = getattr(configs, 'basis_blocks', 1)
            self.basis_bottle = getattr(configs, 'basis_bottle', 4)
            self.basis_d_model = getattr(configs, 'basis_d_model', 128)

            # Normalized Learnable Temporal Prototype Bases
            self.bases = nn.Parameter(torch.randn(1, self.seq_len + self.pred_len, self.basis_nums) * 0.02)

            # Projections to basis_d_model
            self.project_series = wn(nn.Linear(self.seq_len, self.basis_d_model))
            self.project_basis = wn(nn.Linear(self.seq_len, self.basis_d_model))

            # Coefnet for cross-attention and score computation
            self.coefnet = Coefnet(blocks=self.basis_blocks, d_model=self.basis_d_model, heads=self.basis_heads)

            # Projection for future basis & reconstruction
            bottle = max(1, self.pred_len // self.basis_bottle)
            self.MLP_y = MLP_bottle(self.pred_len, self.basis_heads * int(self.pred_len / self.basis_heads), bottle)
            self.MLP_sy = MLP_bottle(self.basis_heads * int(self.pred_len / self.basis_heads), self.pred_len, bottle)

            # Warm-start residual gate (started at 0.01)
            self.alpha_basis = nn.Parameter(torch.full((1,), 0.01), requires_grad=True)

    def forward(self, x, hour_index=None, day_index=None, x_mark_enc=None):
        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

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
        y_time_rec = torch.fft.irfft(y_freq, n=self.seq_len, dim=2, norm="ortho")
        y_freq_out = self.model(y_time_rec).permute(0, 2, 1) # [B, pred_len, enc_in]

        # -----------------------------------------------------------------
        # 2. Time-Domain Basis Selection Branch (BasisFormer Coefnet)
        # -----------------------------------------------------------------
        if self.use_basis:
            # x is normalized by RevIN: [B, seq_len, enc_in]
            feat_series = self.project_series(x.permute(0, 2, 1))  # [B, enc_in, basis_d_model]

            # Normalize bases over temporal length
            m = self.bases / torch.sqrt(torch.sum(self.bases ** 2, dim=1, keepdim=True) + 1e-5)  # [1, seq_len + pred_len, N]
            raw_m1 = m[:, :self.seq_len].permute(0, 2, 1)  # [1, N, seq_len]
            raw_m2 = m[:, self.seq_len:].permute(0, 2, 1)  # [1, N, pred_len]

            # Broadcast bases across batch
            raw_m1_batch = raw_m1.expand(B, -1, -1)
            m1 = self.project_basis(raw_m1_batch)  # [B, N, basis_d_model]

            # Coefnet computes cross-attention and projection score: [B, heads, enc_in, N]
            score = self.coefnet(m1, feat_series)

            # Project future bases and aggregate with selected weights
            raw_m2_batch = raw_m2.expand(B, -1, -1)
            base_fut = self.MLP_y(raw_m2_batch).reshape(B, self.basis_nums, self.basis_heads, -1).permute(0, 2, 1, 3)  # [B, heads, N, pred_len / heads]
            out_basis = torch.matmul(score, base_fut).permute(0, 2, 1, 3).reshape(B, C, -1)  # [B, enc_in, heads * (pred_len / heads)]
            out_basis = self.MLP_sy(out_basis).permute(0, 2, 1)  # [B, pred_len, enc_in]

            # Dual-Stream Warm-Start Residual Fusion
            y_final = y_freq_out + self.alpha_basis * out_basis
        else:
            y_final = y_freq_out

        if self.use_revin:
            y_final = y_final * torch.sqrt(seq_var) + seq_mean

        return y_final