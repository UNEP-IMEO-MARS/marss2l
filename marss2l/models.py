import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.base import ClassificationHead
import segmentation_models_pytorch as smp
from marss2l.mars_sentinel2 import utils_torch
from typing import Optional, Union


class UnetOriginal(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 gaussian_out=False,
                 div_factor=1,
                 class_head=False,
                 batch_norm:bool=True

    ):
        super(UnetOriginal, self).__init__()

        self.n_channels = in_channels
        self.bilinear = True
        self.fp = nn.Softplus()
        self.gaussian_out = gaussian_out
        self.class_head = class_head
        self.batch_norm = batch_norm

        if self.class_head:
            self.classification_head = ClassificationHead(in_channels=512//div_factor, classes=1)
        else:
            self.classification_head = None

        def double_conv(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels) if self.batch_norm  else nn.Identity(),
                nn.GELU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels) if self.batch_norm  else nn.Identity(),
                nn.GELU(),
            )

        def down(in_channels, out_channels):
            return nn.Sequential(
                nn.MaxPool2d(2),
                double_conv(in_channels, out_channels)
            )

        class up(nn.Module):
            def __init__(self, in_channels, out_channels, bilinear=True):
                super().__init__()

                if bilinear:
                    self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                else:
                    self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2,
                                                kernel_size=2, stride=2)

                self.conv = double_conv(in_channels, out_channels)

            def forward(self, x1, x2):
                x1 = self.up(x1)
                # [?, C, H, W]
                diffY = x2.size()[2] - x1.size()[2]
                diffX = x2.size()[3] - x1.size()[3]

                x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                                diffY // 2, diffY - diffY // 2])
                x = torch.cat([x2, x1], dim=1) ## why 1?
                return self.conv(x)

        self.inc = double_conv(self.n_channels, 64//div_factor)
        self.down1 = down(64//div_factor, 128//div_factor)
        self.down2 = down(128//div_factor, 256//div_factor)
        self.down3 = down(256//div_factor, 512//div_factor)
        self.down4 = down(512//div_factor, 512//div_factor)
        self.up1 = up(1024//div_factor, 256//div_factor)
        self.up2 = up(512//div_factor, 128//div_factor)
        self.up3 = up(256//div_factor, 64//div_factor)
        self.up4 = up(128//div_factor, 128//div_factor)

        self.out = nn.Conv2d(128//div_factor, out_channels, kernel_size=1, stride=1)


           
    def forward(self, x):
        x = x["y_context_ls0_0"]
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out(x)

        if self.class_head:
            classification_output = self.classification_head(x5)
            return x[:, 0], classification_output

        return  x[:, 0] 
    


def init_weights_film(n_locs:int, nchannels:int) -> torch.Tensor:
    return torch.cat((torch.ones(1, n_locs, nchannels,1,1), 
                      torch.zeros(1, n_locs, nchannels,1,1)), dim=0)

def freeze_eval_batchnorm(model, logger):
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            logger.info(f"Freezing batchnorm layer: {m}")
            m.eval()
            m.requires_grad = False

class UnetFiLMRefactor(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 div_factor=1,
                 class_head=False,
                 batch_norm:bool=True,
                 n_locs:int=None,
                 one_param_per_channel:bool=True

    ):
        super(UnetFiLMRefactor, self).__init__()

        assert n_locs is not None, "n_locs must be provided"

        self.n_channels = in_channels
        self.bilinear = True
        self.class_head = class_head
        self.batch_norm = batch_norm
        self.n_locs = n_locs
        self.one_param_per_channel = one_param_per_channel

        def double_conv(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels) if self.batch_norm  else nn.Identity(),
                nn.GELU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels) if self.batch_norm  else nn.Identity(),
            )

        def down(in_channels, out_channels):
            return nn.Sequential(
                nn.MaxPool2d(2),
                double_conv(in_channels, out_channels)
            )

        class up(nn.Module):
            def __init__(self, in_channels, out_channels, bilinear=True):
                super().__init__()

                if bilinear:
                    self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                else:
                    self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2,
                                                kernel_size=2, stride=2)

                self.conv = double_conv(in_channels, out_channels)

            def forward(self, x1, x2):
                x1 = self.up(x1)
                # [?, C, H, W]
                diffY = x2.size()[2] - x1.size()[2]
                diffX = x2.size()[3] - x1.size()[3]

                x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                                diffY // 2, diffY - diffY // 2])
                x = torch.cat([x2, x1], dim=1) ## why 1?
                return self.conv(x)

        if self.class_head:
            self.classification_head = ClassificationHead(in_channels=512//div_factor, classes=1)
        else:
            self.classification_head = None

        self.inc = double_conv(self.n_channels, 64//div_factor)
        self.down1 = down(64//div_factor, 128//div_factor)
        self.down2 = down(128//div_factor, 256//div_factor)
        self.down3 = down(256//div_factor, 512//div_factor)
        self.down4 = down(512//div_factor, 512//div_factor)
        self.up1 = up(1024//div_factor, 256//div_factor)
        self.up2 = up(512//div_factor, 128//div_factor)
        self.up3 = up(256//div_factor, 64//div_factor)
        self.up4 = up(128//div_factor, 128//div_factor)

        self.out = nn.Conv2d(128//div_factor, out_channels, kernel_size=1, stride=1)
        self.film_params = nn.ParameterList([
            torch.nn.Parameter(init_weights_film(self.n_locs, 64//div_factor if self.one_param_per_channel else 1)),
            torch.nn.Parameter(init_weights_film(self.n_locs, 128//div_factor if self.one_param_per_channel else 1)),
            torch.nn.Parameter(init_weights_film(self.n_locs, 256//div_factor if self.one_param_per_channel else 1)),
            torch.nn.Parameter(init_weights_film(self.n_locs, 512//div_factor if self.one_param_per_channel else 1)),
            torch.nn.Parameter(init_weights_film(self.n_locs, 512//div_factor if self.one_param_per_channel else 1)),
            torch.nn.Parameter(init_weights_film(self.n_locs, 256//div_factor if self.one_param_per_channel else 1)),
            torch.nn.Parameter(init_weights_film(self.n_locs, 128//div_factor if self.one_param_per_channel else 1)),
            torch.nn.Parameter(init_weights_film(self.n_locs, 64//div_factor if self.one_param_per_channel else 1)),
            torch.nn.Parameter(init_weights_film(self.n_locs, 128//div_factor if self.one_param_per_channel else 1))
        ])
        self.activation = nn.GELU()
    
    def do_film(self, x, film_params, inc=False):

        a = film_params[0,...]
        b = film_params[1,...]
        x = x*a+b
        
        return x

           
    def forward(self, x):
        
        site_ids = x["site_ids"]
        x = x["y_context_ls0_0"]

        #print(film_params[0])

        # x = batch,channels, w, h
        # film = 18, 24, 1, 1, 
        film_params = [torch.index_select(p,1,site_ids) for p in self.film_params]

        x1 = self.activation(self.inc(x))

        x1 = self.activation(self.do_film(x1, film_params[0], inc=True)) #x1*film_params[0]+film_params[9])
        x2 = self.down1(x1)
        x2 = self.activation(self.do_film(x2, film_params[1])) #x2*film_params[1]+film_params[10])
        x3 = self.down2(x2)
        x3 = self.activation(self.do_film(x3, film_params[2])) #x3*film_params[2]+film_params[11])
        x4 = self.down3(x3)
        x4 = self.activation(self.do_film(x4, film_params[3])) #x4*film_params[3]+film_params[12])
        x5 = self.down4(x4)
        x5 = self.activation(self.do_film(x5, film_params[4])) #x5*film_params[4]+film_params[13])
        #x5 = self.middle(x5)
        x = self.up1(x5, x4)
        x = self.activation(self.do_film(x, film_params[5])) #x*film_params[5]+film_params[14])
        x = self.up2(x, x3)
        x = self.activation(self.do_film(x, film_params[6])) #x*film_params[6]+film_params[15])
        x = self.up3(x, x2)
        x = self.activation(self.do_film(x, film_params[7])) #x*film_params[7]+film_params[16])
        x = self.up4(x, x1)
        x = self.activation(self.do_film(x, film_params[8])) #x*film_params[8]+film_params[17])
        x = self.out(x)

        if self.class_head:
            classification_output = self.classification_head(x5)
            return x[:, 0], classification_output

        return  x[:, 0]
    

class MyUnetPlusPlus(smp.UnetPlusPlus):
    def forward(self, x):
        x = x["y_context_ls0_0"]

        # TODO wrap in padded forward
        pad_r = utils_torch.find_padding(x.shape[-2], 32)
        pad_c = utils_torch.find_padding(x.shape[-1], 32)

        need_pad = (pad_r[0] > 0) or (pad_r[1] > 0) or (pad_c[0] > 0) or (pad_c[1] > 0)

        if need_pad:
            tensor_padded = F.pad(x, (pad_c[0], pad_c[1], pad_r[0], pad_r[1]), mode='reflect')
        else:
            tensor_padded = x
        pred_padded = super().forward(tensor_padded)
        if not need_pad:
            return pred_padded

        if self.classification_head is not None:
            pred_padded, classification_output = pred_padded
        
        slice_rows = slice(pad_r[0], None if pad_r[1] <= 0 else -pad_r[1])
        slice_cols = slice(pad_c[0], None if pad_c[1] <= 0 else -pad_c[1])
        pred_padded = pred_padded[(slice(None), slice(None), slice_rows, slice_cols)]
        
        if self.classification_head is not None:
            return pred_padded, classification_output
         
        return pred_padded[:, 0]
    
SegmentationModelMARSS2L = Union[UnetFiLMRefactor, UnetOriginal, MyUnetPlusPlus]
import logging

def load_model(model_name:str,in_channels:int, 
               classification_head:bool, 
               max_index_film:Optional[int]=None,
               batch_norm:bool=True,
               one_param_per_channel:bool=True,
               finetune_film:bool=False,
               finetune_class_head:bool=False,
               logger:Optional[logging.Logger]=None) -> SegmentationModelMARSS2L:
    if logger is None:
        logger = logging.getLogger(__name__)

    if model_name == "film":
        logger.info("FiLM")
        assert max_index_film is not None and (max_index_film > 0), f"max_index_film should not be None if model_name is film provided {max_index_film}"

        model = UnetFiLMRefactor(in_channels=in_channels, 
                        out_channels=1, 
                        div_factor=1,
                        batch_norm=batch_norm,
                        class_head=classification_head,
                        n_locs = max_index_film,
                        one_param_per_channel=one_param_per_channel)
        if finetune_film:
            freeze_eval_batchnorm(model, logger)

            # Requires_grad only for film_params
            for param in model.parameters():
                param.requires_grad = False
            
            for param in model.film_params.parameters():
                param.requires_grad = True
            
            for name, param in model.named_parameters():
                if param.requires_grad:
                    logger.info(f"Finetuning layer: {name}")
            
    elif model_name == "UnetOriginal":
        logger.info("UnetOriginal")
        model = UnetOriginal(in_channels=in_channels, 
                        out_channels=1, 
                        div_factor=1,
                        class_head=classification_head,
                        batch_norm=batch_norm)
    elif model_name == "UnetPlusPlus":
        if classification_head:
            aux_params = {"classes": 1}
            # } "dropout": 0.5, "pooling": "avg", "activation": "sigmoid"}
        else:
            aux_params = None
        logger.info("UnetPlusPlus")
        model = MyUnetPlusPlus(encoder_name="resnext50_32x4d",
                            encoder_depth=5,
                            decoder_channels=(256, 128, 64, 32, 16),
                            encoder_weights=None,
                            in_channels=in_channels,
                            aux_params=aux_params,
                            decoder_attention_type="scse",classes=1)
    else:
        raise ValueError(f"Model {model_name} not recognized")
    
    if classification_head and finetune_class_head:
        freeze_eval_batchnorm(model, logger)
        # Set all to requires_grad=False
        for param in model.parameters():
            param.requires_grad = False
        
        for name, param in model.classification_head.named_parameters():
            logger.info(f"Finetuning layer: {name}")
            param.requires_grad = True
    
    return model

from collections import OrderedDict

def load_weights(model, weights_file:str, device:Optional[torch.device]=None, strict:bool=False):
    state_dict = torch.load(weights_file, map_location=device)["model_state_dict"]

    if next(iter(state_dict.keys())).startswith("module"):
        state_dict = OrderedDict([(k.replace("module.",""), state_dict[k]) for k in state_dict.keys()])
    elif next(iter(state_dict.keys())).startswith("_orig_mod.module"):
        state_dict = OrderedDict([(k.replace("_orig_mod.module.",""), state_dict[k]) for k in state_dict.keys()])
    # elif not next(iter(state_dict.keys())).startswith("module") and (device_name == "cuda"):
    #     state_dict = OrderedDict([("module."+k, state_dict[k]) for k in state_dict.keys()])
    
    model.load_state_dict(state_dict, strict=strict)