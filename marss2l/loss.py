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
CH4_MIN_FOR_WEIGHTING = 0  # ppb
SCALE_CH4_LOSS = 1_000  # Scale CH4 values to ppm for loss weighting
DEFAULT_POS_WEIGHT = 10
DEFAULT_APPLY_SNR_FACTOR = False


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
                   apply_snr_factor: bool = DEFAULT_APPLY_SNR_FACTOR) -> torch.Tensor:
    """
    Compute the BCE loss weights based on CH4 values and target mask.

    Args:
        ch4 (torch.Tensor): Tensor of CH4 values.
        target (torch.Tensor): Binary target mask tensor.
        pos_weight (float): Positive weight factor for the loss. Default is 1.
        apply_snr_factor (bool): Whether to apply SNR factor to the weights. Default is True.

    Returns:
        torch.Tensor: Weights for CH4 values.
    """

    ch4_positive_weight = (
        torch.clamp(ch4, CH4_MIN_FOR_WEIGHTING, CH4_MAX_FOR_WEIGHTING) / SCALE_CH4_LOSS
    )

    # Reduce the ch4_positive_weight when SNR is low
    # 5 is higher of the 75% percentile in all case study areas,
    # 2.5 is a rough estimate of the average SNR
    ch4_positive_weight = ch4_positive_weight * pos_weight

    if apply_snr_factor:
        snr = get_snr(ch4, target, keepdim=True)
        snr_factor = torch.clamp(snr, 0.01, 5) / 2.5
        ch4_positive_weight = ch4_positive_weight * snr_factor
    

    # TODO smooth also the (1-target) weight by the distance to the 1 values (to the plume)?
    # This could be done with some morphological operations on the target mask

    weight = ch4_positive_weight * target + (1 - target)
    return weight


def load_loss(
    weight_by_ch4: bool,
    pos_weight: float,
    class_head: bool = False,
    only_classification: bool = False,
    weight_by_ime: bool = False,
    device: torch.device = torch.device("cpu"),
    apply_snr_factor: bool = DEFAULT_APPLY_SNR_FACTOR,
) -> Callable:
    """
    Load the loss function based on the given parameters.

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
        apply_snr_factor (bool): Whether to apply SNR factor to the weights. Default is True.

    Returns:
        Callable: The loss function to be used during training.
    """
    pos_weight_tensor = torch.Tensor([pos_weight])[:, None, None].to(device)
    bce_segmentation = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight_tensor)

    # if class_head:
    #     bce_classification = nn.BCEWithLogitsLoss(reduction="none")
    # ime = (np.sum(methane_enhancement_image_values[binary_mask]) * 8 * 10*10 * 1_000 * 0.01604) / (1e6 * 22.4)

    if weight_by_ch4:

        def loss_fn(target: torch.Tensor, pred: PredType, ch4: torch.Tensor) -> torch.Tensor:
            target = target.squeeze(1)
            ch4 = ch4.squeeze(1)

            if class_head:
                # The proposed network does not use the classification head
                target_classification = (target.sum(dim=(-2, -1)) > 0).float()
                pred, pred_classification = pred

                # Weight by size of the plume
                if weight_by_ime:
                    ime = (target * ch4).sum(dim=(-2, -1)) * FACTOR_IME * FACTOR_POS
                    L = torch.sqrt(target.sum(dim=(-2, -1)) * 10 * 10)
                    good_inds = L > 0
                    ime_L = torch.where(good_inds, ime, 1) / torch.where(good_inds, L, 1)
                    # L = np.sqrt(npix_plume * np.prod(resolution))
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
                if only_classification:
                    return ll_classification
            else:
                ll_classification = 0

            ll = bce_segmentation(pred.squeeze(1), target)
            weight = get_ch4_weight(ch4, target, pos_weight=1, apply_snr_factor=apply_snr_factor)
            ll = ll * weight
            ll = ll.sum(dim=(-2, -1))
            return ll.mean() + ll_classification

    else:

        def loss_fn(target: torch.Tensor, pred: PredType) -> torch.Tensor:
            target = target.squeeze(1)

            if class_head:
                target_classification = (target.sum(dim=(-2, -1)) > 0).float()
                pred, pred_classification = pred
                ll_classification = F.binary_cross_entropy_with_logits(
                    pred_classification.squeeze(1),
                    target_classification,
                    reduction="mean",
                )
                ll_classification = WEIGHT_CLASSIFICATION_OUTPUT * ll_classification
                if only_classification:
                    return ll_classification
            else:
                ll_classification = 0

            ll = bce_segmentation(pred.squeeze(1), target)
            ll = ll.sum(dim=(-2, -1))
            return ll.mean() + ll_classification

    return loss_fn
