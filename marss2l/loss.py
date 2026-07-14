from typing import Callable, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from marss2l.sampling import WINDOW_SIZE_TRAINING

PredType = Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]
# How much to weight the classification loss vs the segmentation loss
WEIGHT_CLASSIFICATION_OUTPUT = WINDOW_SIZE_TRAINING * WINDOW_SIZE_TRAINING / 64

# See quantification module
# 8 is 8_000 / 1000 (conversion from ppb to ppmxm)
FACTOR_IME = (8 * 10 * 10 * 1_000 * 0.01604) / (1e6 * 22.4)
FACTOR_POS = 5
CH4_MAX_FOR_WEIGHTING = 2_000  # ppb
CH4_MIN_FOR_WEIGHTING = 100  # ppb
SCALE_CH4_LOSS = 1_000  # Scale CH4 values to ppm for loss weighting
DEFAULT_POS_WEIGHT = 10
DEFAULT_WEIGHT_BY_CH4 = True
DEFAULT_NOISE_WARMUP_EPOCHS = 5
DEFAULT_NOISE_TRANSITION_EPOCHS = 50


def get_snr(ch4: torch.Tensor, target: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
    """
    Compute the signal-to-noise ratio (SNR) of CH4 values given a target mask.

    Args:
        ch4 (torch.Tensor): Tensor of CH4 values.
        target (torch.Tensor): Binary target mask tensor.
        keepdim (bool): Whether to keep the dimensions after reduction. Default is True.

    Returns:
        torch.Tensor: SNR values computed as the ratio of signal to noise.
    """
    signal = (ch4 * target).mean(dim=(-2, -1), keepdim=keepdim)
    noise = (ch4 * (1 - target)).mean(dim=(-2, -1), keepdim=keepdim) + 1e-6
    npixelsplume = target.sum(dim=(-2, -1), keepdim=keepdim)
    npix = target.shape[-1] * target.shape[-2]
    npixelsbackground = npix - npixelsplume
    snr = signal / noise
    snr = snr * (npixelsbackground / (npixelsplume + 1e-6))
    return snr


def get_ch4_weight(ch4: torch.Tensor, 
                   target: torch.Tensor, 
                   pos_weight: float = 1,
                   ch4_min_for_weighting: float = CH4_MIN_FOR_WEIGHTING,
                   ch4_max_for_weighting: float = CH4_MAX_FOR_WEIGHTING,
                   scale_ch4_loss: float = SCALE_CH4_LOSS) -> torch.Tensor:
    """
    Compute the BCE loss weights based on CH4 values and target mask.

    The ch4 weight is computed as:
        clamped_ch4 = clamp(ch4, ch4_min_for_weighting, ch4_max_for_weighting)
        weight = (clamped_ch4 / scale_ch4_loss) * pos_weight * snr_factor for target == 1
        weight = 1 for target == 0

    Args:
        ch4 (torch.Tensor): Tensor of CH4 values.
        target (torch.Tensor): Binary target mask tensor.
        pos_weight (float): Positive weight factor for the loss. Default is 1.
        ch4_min_for_weighting (float): Minimum CH4 value for weighting in ppb. Default is CH4_MIN_FOR_WEIGHTING (0).
        ch4_max_for_weighting (float): Maximum CH4 value for weighting in ppb. Default is CH4_MAX_FOR_WEIGHTING (2000).
        scale_ch4_loss (float): Scale factor to convert CH4 values to ppm for loss weighting. Default is SCALE_CH4_LOSS (1000).

    Returns:
        torch.Tensor: Weights for CH4 values.
    """

    ch4_positive_weight = (
        torch.clamp(ch4, ch4_min_for_weighting, ch4_max_for_weighting) / scale_ch4_loss
    )

    ch4_positive_weight = ch4_positive_weight * pos_weight

    # Reduce the ch4_positive_weight when SNR is low
    # 5 is higher of the 75% percentile in all case study areas,
    # 2.5 is a rough estimate of the average SNR
    # if apply_snr_factor:
    #     snr = get_snr(ch4, target, keepdim=True)
    #     snr_factor = torch.clamp(snr, 0.01, 5) / 2.5
    #     ch4_positive_weight = ch4_positive_weight * snr_factor
    

    # TODO smooth also the (1-target) weight by the distance to the 1 values (to the plume)?
    # This could be done with some morphological operations on the target mask

    weight = ch4_positive_weight * target + (1 - target)
    return weight


class DeltaXCH4Loss(nn.Module):
    """
    Loss function for Delta XCH4 prediction with optional CH4-based weighting,
    classification head, and noise-based curriculum learning.
    """
    
    def __init__(
        self,
        weight_by_ch4: bool = DEFAULT_WEIGHT_BY_CH4,
        pos_weight: float = DEFAULT_POS_WEIGHT,
        class_head: bool = False,
        only_classification: bool = False,
        weight_by_ime: bool = False,
        device: torch.device = torch.device("cpu"),
        ch4_min_for_weighting: float = CH4_MIN_FOR_WEIGHTING,
        ch4_max_for_weighting: float = CH4_MAX_FOR_WEIGHTING,
        scale_ch4_loss: float = SCALE_CH4_LOSS,
        weight_by_noise: bool = False,
        noise_warmup_epochs: int = 5,
        noise_transition_epochs: int = 20,
    ):
        """
        Initialize the DeltaXCH4Loss module.

        Args:
            weight_by_ch4 (bool): Whether to weight the loss by CH4 concentration.
            pos_weight (float): Positive weight for the loss function.
            class_head (bool): Whether a classification head is used. If True,
                there will be two outputs, one for the segmentation and one for the classification.
                Default is False.
            only_classification (bool): Whether to use only the classification head. Default is False.
            weight_by_ime (bool): Whether to weight the classification loss by the size of the plume.
                Default is False.
            device (torch.device): Device to run the model (e.g., cuda or cpu).
            ch4_min_for_weighting (float): Minimum CH4 value for weighting in ppb. Default is CH4_MIN_FOR_WEIGHTING (0).
            ch4_max_for_weighting (float): Maximum CH4 value for weighting in ppb. Default is CH4_MAX_FOR_WEIGHTING (2000).
            scale_ch4_loss (float): Scale factor to convert CH4 values to ppm for loss weighting. Default is SCALE_CH4_LOSS (1000).
            weight_by_noise (bool): Whether to weight samples by retrieval noise with curriculum learning. Default is False.
            noise_warmup_epochs (int): Epochs with equal weighting before curriculum starts. Default is 5.
            noise_transition_epochs (int): Epochs to transition from noise-based to equal weighting. Default is 20.
        """
        super().__init__()
        
        self.weight_by_ch4 = weight_by_ch4
        self.class_head = class_head
        self.only_classification = only_classification
        self.weight_by_ime = weight_by_ime
        self.ch4_min_for_weighting = ch4_min_for_weighting
        self.ch4_max_for_weighting = ch4_max_for_weighting
        self.scale_ch4_loss = scale_ch4_loss
        self.weight_by_noise = weight_by_noise
        self.noise_warmup_epochs = noise_warmup_epochs
        self.noise_transition_epochs = noise_transition_epochs
        
        # Create BCE loss for segmentation
        pos_weight_tensor = torch.Tensor([pos_weight])[:, None, None].to(device)
        self.bce_segmentation = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight_tensor)
    
    def compute_noise_weight(self, ch4noise: torch.Tensor, epoch: int) -> torch.Tensor:
        """
        Compute sample weights based on retrieval noise with curriculum learning.
        
        Args:
            ch4noise: Retrieval noise in ppb, shape (batch_size,)
            epoch: Current training epoch
            
        Returns:
            Sample weights, shape (batch_size,)
        """
        if not self.weight_by_noise:
            return torch.ones_like(ch4noise)
        
        if epoch < self.noise_warmup_epochs:
            return torch.ones_like(ch4noise)
        
        # Noise-based weight: higher noise -> lower weight
        # Use inverse relationship, clamped to reasonable range [0.1, 1.0]
        noise_weight = torch.clamp(200.0 / (ch4noise + 50.0), 0.1, 1.0)
        
        # Curriculum progress: 0 at warmup, 1 at warmup+transition
        progress = min(1.0, (epoch - self.noise_warmup_epochs) / self.noise_transition_epochs)
        
        # Interpolate between noise-based and equal weighting
        return noise_weight + (1.0 - noise_weight) * progress
    
    def forward(
        self,
        target: torch.Tensor,
        pred: PredType,
        ch4: torch.Tensor = None,
        ch4noise: torch.Tensor = None,
        epoch: int = 0,
    ) -> torch.Tensor:
        """
        Compute the loss.

        Args:
            target (torch.Tensor): Target mask tensor.
            pred (PredType): Predicted output from the model.
            ch4 (torch.Tensor, optional): CH4 values for weighting (required if weight_by_ch4=True).
            ch4noise (torch.Tensor, optional): CH4 noise values for curriculum weighting.
            epoch (int, optional): Current training epoch. Default is 0.

        Returns:
            torch.Tensor: Computed loss value.
        """
        target = target.squeeze(1)
        
        ll_classification = 0
        
        if self.class_head:
            # The proposed network does not use the classification head
            target_classification = (target.sum(dim=(-2, -1)) > 0).float()
            pred, pred_classification = pred

            # Weight by size of the plume
            if self.weight_by_ime:
                if ch4 is None:
                    raise ValueError("ch4 must be provided when weight_by_ime is True")
                ch4_squeezed = ch4.squeeze(1)
                ime = (target * ch4_squeezed).sum(dim=(-2, -1)) * FACTOR_IME * FACTOR_POS
                L = torch.sqrt(target.sum(dim=(-2, -1)) * 10 * 10)
                good_inds = L > 0
                ime_L = torch.where(good_inds, ime, 1) / torch.where(good_inds, L, 1)
                pos_weight_class = torch.clamp(ime_L, 0.1, 10)
            else:
                pos_weight_class = None

            ll_classification = F.binary_cross_entropy_with_logits(
                pred_classification.squeeze(1),
                target_classification,
                pos_weight=pos_weight_class,
                reduction="mean",
            )

            ll_classification = WEIGHT_CLASSIFICATION_OUTPUT * ll_classification
            if self.only_classification:
                return ll_classification
        
        # Compute segmentation loss
        ll = self.bce_segmentation(pred.squeeze(1), target)
        
        # Apply CH4-based weighting
        if self.weight_by_ch4:
            if ch4 is None:
                raise ValueError("ch4 must be provided when weight_by_ch4 is True")
            ch4_squeezed = ch4.squeeze(1)
            weight = get_ch4_weight(
                ch4_squeezed,
                target,
                pos_weight=1, # Because pos_weight is already applied in BCE loss
                ch4_min_for_weighting=self.ch4_min_for_weighting,
                ch4_max_for_weighting=self.ch4_max_for_weighting,
                scale_ch4_loss=self.scale_ch4_loss,
            )
            ll = ll * weight
        
        ll = ll.sum(dim=(-2, -1))  # Shape: (batch_size,)
        
        # Apply noise-based curriculum weighting
        if self.weight_by_noise:
            if ch4noise is None:
                raise ValueError("ch4noise must be provided when weight_by_noise is True")
            noise_weight = self.compute_noise_weight(ch4noise, epoch)
            ll = ll * noise_weight
            return ll.sum() / noise_weight.sum() + ll_classification
        
        return ll.mean() + ll_classification
