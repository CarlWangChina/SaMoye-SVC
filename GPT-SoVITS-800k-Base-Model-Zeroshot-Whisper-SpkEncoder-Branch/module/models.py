import torch
from torch import nn
from torch.nn import functional as F

from module import commons
from module import modules
from module import attentions

from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from module.commons import init_weights, get_padding
from module.mrte_model import MRTE
from module.quantize import ResidualVectorQuantizer



class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(
                    Conv2d(
                        1,
                        32,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        32,
                        128,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        128,
                        512,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        512,
                        1024,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        1024,
                        1024,
                        (kernel_size, 1),
                        1,
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
            ]
        )
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(Conv1d(1, 16, 15, 1, padding=7)),
                norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2, 3, 5, 7, 11]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [
            DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods
        ]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class TextEncoder(nn.Module):
    def __init__(
        self,
        out_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        latent_channels=192,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.latent_channels = latent_channels

        self.ssl_proj = nn.Conv1d(768, hidden_channels, 1)
        self.encoder_ssl = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers // 2,
            kernel_size,
            p_dropout,
        )

        self.f0_embedding = nn.Conv1d(1, hidden_channels, 1)
        self.encoder_f0 = attentions.Encoder(
            hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout
        )
    
        self.mrte = MRTE()

        self.encoder_out = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers // 2,
            kernel_size,
            p_dropout,
        )

        self.out = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, ssl, ssl_length, f0, f0_length, style_embed):
        # padding mask (part with original legnth set to True and padding part set to False)
        ssl_pad_mask = torch.unsqueeze(commons.sequence_mask(ssl_length, ssl.size(2)), 1).to(ssl.dtype)

        ssl = self.ssl_proj(ssl * ssl_pad_mask) * ssl_pad_mask
        ssl = self.encoder_ssl(ssl * ssl_pad_mask, ssl_pad_mask)

        f0_pad_mask = torch.unsqueeze(commons.sequence_mask(f0_length, f0.size(1)), 1).to(ssl.dtype)
        f0 = self.f0_embedding(f0.unsqueeze(1))
        f0 = self.encoder_f0(f0 * f0_pad_mask, f0_pad_mask)

        # Multi-reference timbre encoder embedding (ref phones and style)
        ssl = self.mrte(ssl, ssl_pad_mask, f0, f0_pad_mask, style_embed)
        ssl = self.encoder_out(ssl * ssl_pad_mask, ssl_pad_mask)

        # project to output channels
        mu_logvar = self.out(ssl) * ssl_pad_mask
        m, logs = torch.split(mu_logvar, self.out_channels, dim=1)
        return ssl, m, logs, ssl_pad_mask



class ResidualCouplingBlock(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        n_flows=4,
        gin_channels=0,
    ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                modules.ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                )
            )
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, style_embed, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, style_embed, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, style_embed, reverse=reverse)
        return x


class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.wav_conv = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        self.out = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, wave, wave_lengths, style_embed):
        style_embed = style_embed.detach()
        # create padding mask for (True for original part, False for padding part MAX_LENGH - ORIGINAL_LENGTH)
        wave_mask = torch.unsqueeze(commons.sequence_mask(wave_lengths, wave.size(2)), 1).to(wave.dtype)

        wave = self.wav_conv(wave) * wave_mask
        wave = self.enc(wave, wave_mask, style_embed)

        stats = self.out(wave) * wave_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)

        z = (m + torch.randn_like(m) * torch.exp(logs)) * wave_mask
        return z, m, logs, wave_mask


class Generator(torch.nn.Module):
    def __init__(
        self,
        initial_channel,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels=0,
    ):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(
            initial_channel, upsample_initial_channel, 7, 1, padding=3
        )
        resblock = modules.ResBlock

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()



class SynthesizerTrn(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(
        self,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        n_speakers=0,
        gin_channels=0,
        use_sdp=True,
        semantic_frame_rate=None,
        freeze_quantizer=True,
        **kwargs
    ):
        super().__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels

        self.use_sdp = use_sdp

        self.enc_p = TextEncoder(
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
        )

        self.dec = Generator(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
        )

        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            gin_channels=gin_channels,
        )
        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels
        )

        self.enc_ref = modules.MelStyleEncoder(
            spec_channels, style_vector_dim=gin_channels
        )

        ssl_dim = 768
        assert semantic_frame_rate in ["25hz", "50hz"]
        self.semantic_frame_rate = semantic_frame_rate
        if semantic_frame_rate == "25hz":
            self.ssl_proj = nn.Conv1d(ssl_dim, ssl_dim, 2, stride=2)
        else:
            self.ssl_proj = nn.Conv1d(ssl_dim, ssl_dim, 1, stride=1)

        self.quantizer = ResidualVectorQuantizer(dimension=ssl_dim, n_q=1, bins=1024)
        self.freeze_quantizer = freeze_quantizer
        
        if self.freeze_quantizer:
            print('Freeze qunatizer')
            self.ssl_proj.eval()
            self.quantizer.eval()
            
            for p in self.ssl_proj.parameters():
                p.requires_grad = False

            for p in self.quantizer.parameters():
                p.requires_grad = False


    '''
    1. 不要用quantized的量化后特征, 用他们的codes自己embedding
    2. style_embed: 考虑时长区别, style fine-grained + cross-attn
    '''
    def forward(self, ssl, wave, wave_lengths, f0, f0_length):
        wave_mask = commons.sequence_mask(wave_lengths, wave.size(2))
        wave_mask = torch.unsqueeze(wave_mask, 1)
        style_embed = self.enc_ref(wave * wave_mask, wave_mask)
        
        ssl = self.ssl_proj(ssl)
        quantized, _, commit_loss, quantized_list = self.quantizer(ssl, layers=[0])
    
        # upsample to 50hz by duplicating vq codes for twice (nearest)
        if self.semantic_frame_rate == "25hz":
            quantized = F.interpolate(
                quantized, size=int(quantized.shape[-1] * 2), mode="nearest"
            )
        

        _, m_p, logs_p, wave_mask_p = self.enc_p(quantized, wave_lengths, f0, f0_length, style_embed)
        z, m_q, logs_q, wave_mask_q = self.enc_q(wave, wave_lengths, style_embed)

        z_p = self.flow(z, wave_mask_q, style_embed)

        z_slice, ids_slice = commons.rand_slice_segments(
            z, wave_lengths, self.segment_size
        )
        dec_out = self.dec(z_slice, g=style_embed)
        
        # dec_out = self.dec(z, g=style_embed)

        return (
            dec_out,
            commit_loss,
            ids_slice,
            wave_mask_p,
            wave_mask_q,
            (z, z_p, m_p, logs_p, m_q, logs_q),
            quantized,
        )

    def infer(self, ssl, wave, wave_lengths, f0, f0_length, noise_scale=0.5):
        wave_mask = torch.unsqueeze(commons.sequence_mask(wave_lengths, wave.size(2)), 1).to(wave.dtype)
        style_embed = self.enc_ref(wave * wave_mask, wave_mask)
        ssl = self.ssl_proj(ssl)
        quantized, codes, commit_loss, _ = self.quantizer(ssl, layers=[0])

        # upsample to 50hz
        if self.semantic_frame_rate == "25hz":
            quantized = F.interpolate(
                quantized, size=int(quantized.shape[-1] * 2), mode="nearest"
            )

        _, m_p, logs_p, wave_mask = self.enc_p(quantized, wave_lengths, f0, f0_length, style_embed)

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale

        z = self.flow(z_p, wave_mask, style_embed, reverse=True)

        o = self.dec((z * wave_mask)[:, :, :], g=style_embed)

        return o, wave_mask, (z, z_p, m_p, logs_p)

    @torch.no_grad()
    def decode(self, ssl, f0, refer_spec, noise_scale=0.5):
        refer_lengths = torch.LongTensor([refer_spec.size(2)]).to(refer_spec.device)
        refer_mask = torch.unsqueeze(
            commons.sequence_mask(refer_lengths, refer_spec.size(2)), 1
        ).to(refer_spec.dtype)

        style_embed = self.enc_ref(refer_spec * refer_mask, refer_mask)

        wave_lengths = torch.LongTensor([ssl.size(2) * 2]).to(ssl.device)
        f0_lengths = torch.LongTensor([f0.size(-1)]).to(f0.device)

        ssl = self.ssl_proj(ssl)
        quantized, _, _, _ = self.quantizer(ssl, layers=[0])

        if self.semantic_frame_rate == "25hz":
            quantized = F.interpolate(
                quantized, size=int(quantized.shape[-1] * 2), mode="nearest"
            )

        x, m_p, logs_p, wave_mask = self.enc_p(
            quantized, wave_lengths, f0, f0_lengths, style_embed
        )
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale

        z = self.flow(z_p, wave_mask, style_embed, reverse=True)

        o = self.dec((z * wave_mask)[:, :, :], g=style_embed)
        return o
