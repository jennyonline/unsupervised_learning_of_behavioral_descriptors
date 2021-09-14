import argparse
import json
import os
import subprocess
from contextlib import contextmanager

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.optim import AdamW
from deepVAEHelpers.hps import Hyperparams, add_vae_arguments, parse_args_and_update_hparams


def lr_schedule(H):
    def f(iteration):
        return 0.0 if iteration < H.adam_warmup_iters else 1.0

    return f


def set_up_hyperparams(s=None):
    H = Hyperparams()
    parser = argparse.ArgumentParser()
    parser = add_vae_arguments(parser)
    parse_args_and_update_hparams(H, parser, s=s)
    np.random.seed(H.seed)
    torch.manual_seed(H.seed)
    torch.cuda.manual_seed(H.seed)
    return H


def load_opt(H, vae):
    optimizer = AdamW(
        vae.parameters(), weight_decay=H.wd, lr=H.lr, betas=(H.adam_beta1, H.adam_beta2)
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule(H))
    cur_eval_loss, iterate, starting_epoch = float("inf"), 0, 0
    return optimizer, scheduler, cur_eval_loss, iterate, starting_epoch
