import os
import sys
import traceback

import numpy as np
from PIL import Image
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer

from modules.upscaler import Upscaler, UpscalerData
from modules.shared import cmd_opts, opts
from modules.realesrgan_trt_rrdb_net import RRDBNet
import torch


def get_sr_model():
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
    load_net = torch.load('models/RealESRGAN_x2plus.pth')
    model.load_state_dict(load_net['params_ema'], strict=True)
    model.eval()
    model = model.to('cuda').half()
    return model
