import utils, os
import torch
import argparse
import torch.distributed as dist
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from random import randint
from module import commons
from module.data import (
    AudioTextLoader,
    AudioTextCollate,
)

from torch.utils.data.distributed import DistributedSampler
from module.models import (
    SynthesizerTrn,
    MultiPeriodDiscriminator,
)
from module.losses import generator_loss, discriminator_loss, feature_loss, kl_loss
from module.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from process_ckpt import savee

torch.distributed.init_process_group(backend="gloo")
rank = torch.distributed.get_rank()
torch.cuda.set_device(rank)
device = torch.device("cuda", rank)

'''
hps.model: {
    inter_channels: 192,
    hidden_channels: 192,
    filter_channels: 768,
    n_heads: 2,
    n_layers: 6,
    kernel_size: 3,
    p_dropout: 0.1,
    resblock: '1',
    resblock_kernel_sizes: [3, 7, 11],
    resblock_dilation_sizes: [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    upsample_rates: [10, 8, 2, 2, 2],
    upsample_initial_channel: 512,
    upsample_kernel_sizes: [16, 16, 8, 2, 2],
    n_layers_q: 3,
    use_spectral_norm: False,
    gin_channels: 512,
    semantic_frame_rate: '25hz',
    freeze_quantizer: True
}
'''
hps = utils.get_hparams()

# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = False
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True

global_step = 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
    else:
        n_gpus = 1
    
    run(n_gpus, hps)


def run(n_gpus, hps):
    global global_step
    logger = utils.get_logger(hps.save_weight_dir)
    logger.info(hps)

    writer = SummaryWriter(log_dir=hps.save_weight_dir) if rank == 0 else None

    torch.manual_seed(hps.train.seed)
    # dataset = AudioTextLoader(wav_roots=['example_refence_audio'], eval_mode=False)
    dataset = AudioTextLoader(wav_roots=['datasets/ali-40w-vocal_datasets_10000', 'datasets/mysong_vocal_datasets_10000'], eval_mode=False)

    train_sampler = DistributedSampler(dataset)

    train_loader = DataLoader(
        dataset,
        num_workers=8,
        # shuffle=True,
        # pin_memory=True,
        collate_fn=AudioTextCollate(),
        batch_size=hps.train.batch_size,
        # persistent_workers=True,
        prefetch_factor=4,
        sampler=train_sampler,
    )

    net_g = SynthesizerTrn(
        spec_channels=hps.data.filter_length // 2 + 1,
        segment_size=hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    )
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm)

    for name, param in net_g.named_parameters():
        if not param.requires_grad:
            print(name, "not requires_grad")

    param_mrte = list(map(id, net_g.enc_p.mrte.parameters()))

    base_params = filter(
        lambda p: id(p) not in param_mrte and p.requires_grad,
        net_g.parameters(),
    )

    # optim_g = torch.optim.AdamW(
    #     [
    #         {
    #             "params": base_params,
    #             "lr": hps.train.learning_rate
    #         },
    #         {
    #             "params": net_g.enc_p.mrte.parameters(),
    #             "lr": hps.train.learning_rate * hps.train.low_lr_rate,
    #         },
    #     ],
    #     hps.train.learning_rate,
    #     betas=hps.train.betas,
    #     eps=hps.train.eps,
    # )
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )

    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )

    if torch.cuda.is_available():
        logger.info("GPU number: %d" % n_gpus)
        net_g = DDP(net_g.cuda(), find_unused_parameters=True)
        net_d = DDP(net_d.cuda(), find_unused_parameters=True)
    else:
        net_g = net_g.to(device)
        net_d = net_d.to(device)

    try:
        _, _, _, epoch_start = utils.load_checkpoint(
            utils.latest_checkpoint_path("%s" % hps.save_weight_dir, "D_*.pth"),
            net_d,
            optim_d,
        )
        logger.info("loaded D")
        _, _, _, epoch_start = utils.load_checkpoint(
            utils.latest_checkpoint_path("%s" % hps.save_weight_dir, "G_*.pth"),
            net_g,
            optim_g,
        )

        logger.info("loaded G")

        global_step = (epoch_start - 1) * len(train_loader)
        
    except:
        epoch_start = 1
        global_step = 0

        if hps.train.pretrained_generator != "":
            logger.info("loaded pretrained %s" % hps.train.pretrained_generator)
            print(
                net_g.module.load_state_dict(
                    torch.load(hps.train.pretrained_generator, map_location="cpu")["weight"],
                    strict=False,
                ) if torch.cuda.is_available() else net_g.load_state_dict(
                    torch.load(hps.train.pretrained_generator, map_location="cpu")["weight"],
                    strict=False,
                )
            )

        # if hps.train.pretrained_discriminator != "":
        #     logger.info("loaded pretrained %s" % hps.train.pretrained_discriminator)
        #     print(
        #         net_d.module.load_state_dict(
        #             torch.load(hps.train.pretrained_discriminator, map_location="cpu")["weight"]
        #         ) if torch.cuda.is_available() else net_d.load_state_dict(
        #             torch.load(hps.train.pretrained_discriminator, map_location="cpu")["weight"]
        #         )
        #     )

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=-1
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=-1
    )

    for _ in range(epoch_start):
        scheduler_g.step()
        scheduler_d.step()

    for epoch in range(epoch_start, hps.train.epochs + 1):
        train(
            epoch,
            hps,
            [net_g, net_d],
            [optim_g, optim_d],
            [scheduler_g, scheduler_d],
            train_loader,
            logger,
            writer,
        )

        scheduler_g.step()
        scheduler_d.step()


def train(
    epoch, hps, nets, optims, schedulers, train_loader, logger, writer
):
    net_g, net_d = nets
    optim_g, optim_d = optims

    global global_step

    net_g.train()
    net_d.train()

    for batch_idx, (
        ssl,
        _,
        spec,
        spec_lengths,
        raw_wav,
        raw_wav_length,
        f0,
        f0_lengths,
    ) in tqdm(enumerate(train_loader)):

        if torch.cuda.is_available():
            spec, spec_lengths = spec.cuda(), spec_lengths.cuda()
            raw_wav, raw_wav_length = raw_wav.cuda(), raw_wav_length.cuda()
            f0, f0_lengths = f0.cuda(), f0_lengths.cuda()
            ssl = ssl.cuda()
        else:
            spec, spec_lengths = spec.to(device), spec_lengths.to(device)
            raw_wav, raw_wav_length = raw_wav.to(device), raw_wav_length.to(device)
            ssl = ssl.to(device)

        (
            y_hat,
            vq_commit_loss,
            ids_slice,
            x_mask,
            z_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q),
            stats_ssl,
        ) = net_g(ssl, spec, spec_lengths, f0, f0_lengths)
        
        # ground truth mel from y
        mel = spec_to_mel_torch(
            spec,
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.mel_fmin,
            hps.data.mel_fmax,
        )
        
        y_mel = commons.slice_segments(
            mel, ids_slice, hps.train.segment_size // hps.data.hop_length
        )
        # generated mel from y_hat
        y_hat_mel = mel_spectrogram_torch(
            y_hat.squeeze(1),
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            hps.data.mel_fmin,
            hps.data.mel_fmax,
        )

        raw_wav = commons.slice_segments(
            raw_wav, ids_slice * hps.data.hop_length, hps.train.segment_size
        )
        
        # Train Discriminator
        y_d_hat_r, y_d_hat_g, _, _ = net_d(raw_wav, y_hat.detach())
        loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
            y_d_hat_r, y_d_hat_g
        )
        loss_disc_all = loss_disc

        optim_d.zero_grad()
        loss_disc_all.backward()
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        optim_d.step()

        y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(raw_wav, y_hat)
        loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl

        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen, losses_gen = generator_loss(y_d_hat_g)
        loss_gen_all = loss_gen + loss_fm + loss_mel + vq_commit_loss + loss_kl

        optim_g.zero_grad()
        loss_gen_all.backward()
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        optim_g.step()

        lr = optim_g.param_groups[0]["lr"]
        losses = [loss_disc, loss_gen, loss_fm, loss_mel, vq_commit_loss, loss_kl]
        logger.info(
            "Train Epoch: {} [{:.0f}%]".format(
                epoch, 100.0 * batch_idx / len(train_loader)
            )
        )
        logger.info([x.item() for x in losses] + [global_step, lr])
        global_step += 1

        if rank == 0:
            logger.info('Training Writer logs')
    
            scalar_dict = {
                "loss/g/total": loss_gen_all,
                "loss/d/total": loss_disc_all,
                "learning_rate": lr,
                "grad_norm_d": grad_norm_d,
                "grad_norm_g": grad_norm_g,
            }
            scalar_dict.update(
                {
                    "loss/g/fm": loss_fm,
                    "loss/g/mel": loss_mel,
                    "loss/g/vq_commit_loss": vq_commit_loss,
                    "loss/g/kl": loss_kl,
                }
            )

            image_dict = {
                "slice/mel_org": utils.plot_spectrogram_to_numpy(
                    y_mel[0].data.cpu().numpy()
                ),
                "slice/mel_gen": utils.plot_spectrogram_to_numpy(
                    y_hat_mel[0].data.cpu().numpy()
                ),
                "all/stats_ssl": utils.plot_spectrogram_to_numpy(
                    stats_ssl[0].data.cpu().numpy()
                ),
            }
            
            audios={
                "slice/audio_org": raw_wav[0].data.cpu().numpy(),
                "slice/audio_gen": y_hat[0].data.cpu().numpy(),
            }
 
            utils.summarize(
                writer=writer,
                global_step=global_step,
                images=image_dict,
                scalars=scalar_dict,
                audios=audios,
                audio_sampling_rate=hps.data.sampling_rate,
            )

        if rank == 0 and global_step % hps.train.eval_interval == 0:

            utils.save_checkpoint(
                net_g,
                optim_g,
                hps.train.learning_rate,
                epoch,
                os.path.join(
                    "%s" % hps.save_weight_dir, "G_{}.pth".format(global_step)
                ),
            )
            utils.save_checkpoint(
                net_d,
                optim_d,
                hps.train.learning_rate,
                epoch,
                os.path.join(
                    "%s" % hps.save_weight_dir, "D_{}.pth".format(global_step)
                ),
            )

            if hasattr(net_g, "module"):
                ckpt = net_g.module.state_dict()
            else:
                ckpt = net_g.state_dict()

    logger.info("====> Epoch: {}".format(epoch))



if __name__ == "__main__":
    main()
