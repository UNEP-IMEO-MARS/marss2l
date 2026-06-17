import json
import os
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import fsspec
import loguru
import numpy as np
import torch
from georeader.geotensor import GeoTensor
from georeader.readers import S2_SAFE_reader
from loguru._logger import Logger
from numpy.typing import NDArray
from ..mbmp_torch import to_mbmp

from ..models import MyUnetPlusPlus, UnetFiLMRefactor, UnetOriginal
from .plume_detection import (
    MINIMUM_NUMBER_PIXELS_PLUME,
    binary_connected_prediction,
    threshold_cutoff_connected_components,
)

from ..huggingface import REPO_ID
from huggingface_hub import hf_file_system, hf_hub_download

from .utils import align_images
from ..utils import fs_from_path

MEANS_S2 = [
    2250.5715,
    2252.9739,
    2622.8955,
    3521.9641,
    3636.5669,
    3841.8989,
    4093.0432,
    3889.2266,
    4217.3589,
    1455.4264,
    221.2708,
    4958.1455,
    4238.7144,
]
STDS_S2 = [
    853.1166,
    855.7593,
    855.4163,
    1059.6523,
    1099.5953,
    1149.8221,
    1207.3115,
    1177.8850,
    1241.5836,
    778.4094,
    443.5816,
    1458.2043,
    1245.1458,
]

glogger = loguru.logger


class MARSS2LModel:
    """
    MARS-S2L model for methane plume detection in Sentinel-2 and Landsat imagery.
    
    This model processes multispectral satellite imagery to detect and segment methane plumes.
    This class allows different configurations including multipass, wind and cloud mask inputs.
    The model can optionally use FiLM (Feature-wise Linear Modulation) for location-specific 
    conditioning.
    
    Parameters
    ----------
    bands : List[str]
        List of spectral band names to use (e.g., ["B02", "B03", "B04", "B08", "B11", "B12"]).
    device : torch.device, default torch.device("cpu")
        Device to run the model on (CPU or CUDA).
    wind : bool, default True
        If True, include wind vector data (U, V components) as model input.
    multipass : bool, default True
        If True, use both current and background images for change detection.
    norm_data : bool, default False
        If True, normalize input data using precomputed mean/std statistics.
    cloud_mask : bool, default True
        If True, include cloud mask as an additional input channel.
    cat_mbmp : bool, default True
        If True, concatenate Modified Band Matched Products as input feature.
    norm_wind : bool, default False
        If True, normalize wind vectors by dividing by 8 m/s.
    threshold_prediction : float, default 0.5
        Probability threshold for binarizing continuous predictions.
    corregister : bool, default True
        If True, co-register background image to current image geometry.
    threshold_pixels : int, default MINIMUM_NUMBER_PIXELS_PLUME
        Minimum number of connected pixels required for a valid plume detection.
    architecture : Optional[str], default None
        Model architecture ("UnetOriginal", "UnetPlusPlus", or "film"). Auto-detected if None.
    logger : Optional[Logger], default None
        Logger instance for logging messages.
    weights : str, default "weights/ch4_model.pt"
        Path to model weights file.
    film_dict_mapping : Optional[Dict[str, int]], default None
        Mapping from location names to FiLM IDs for location-specific conditioning.
    max_index_film : Optional[int], default None
        Maximum FiLM ID index (number of unique locations). Required if film_dict_mapping is set.
    film_train_zero_id : bool, default False
        If True, use zero ID for unknown locations instead of raising an error.
    use_zero_id_always : bool, default False
        If True, always use FiLM ID 0 (ignore film_dict_mapping). Requires film_train_zero_id=True.
    
    Attributes
    ----------
    model : torch.nn.Module
        The underlying neural network model.
    in_channels : int
        Total number of input channels based on enabled features.
    
    Examples
    --------
    Load a pre-trained model and perform inference:
    
    >>> import fsspec
    >>> import torch
    >>> from georeader.geotensor import GeoTensor
    >>> import numpy as np
    >>> 
    >>> # Initialize model
    >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    >>> 
    >>> # Load model with unetpaperrev_20250326v309 configuration
    >>> model = MARSS2LModel(
    ...     bands=["B02", "B03", "B04", "B08", "B11", "B12"],
    ...     device=device,
    ...     wind=True,
    ...     multipass=True,
    ...     norm_data=False,
    ...     cloud_mask=True,
    ...     cat_mbmp=True,
    ...     norm_wind=True,
    ...     threshold_prediction=0.5,
    ...     threshold_pixels=100,
    ...     architecture="UnetOriginal",
    ...     weights="/path/to/weights.ckpt",
    ...     film_dict_mapping=None,
    ...     film_train_zero_id=True,
    ... )
    >>> 
    >>> # Prepare input data (example shapes)
    >>> # image_predict: current image with shape (6, 200, 200)
    >>> # background_image: background image with shape (6, 200, 200)
    >>> # wind_vector: wind data with shape (2,) - [wind_u, wind_v] in m/s
    >>> # validmask: cloud mask with shape (200, 200)
    >>> 
    >>> # Run prediction
    >>> binary_mask, scene_score, is_plume, continuous_pred = model.predict(
    ...     image_predict=image_predict,  # GeoTensor (6, H, W)
    ...     background_image=background_image,  # GeoTensor (6, H, W)
    ...     wind_vector=np.array([3.5, -2.1]),  # (U, V) in m/s
    ...     validmask=validmask,  # GeoTensor (H, W)
    ...     location_name=None,  # Not needed for non-FiLM models
    ... )
    >>> 
    >>> print(f"Plume detected: {is_plume}")
    >>> print(f"Scene-level score: {scene_score:.3f}")
    >>> print(f"Number of plume pixels: {binary_mask.values.sum()}")
    
    Notes
    -----
    - Input images should be in TOA reflectance units multiplied by 10000.
    - The model expects georeferenced data (GeoTensor objects with CRS and transform).
    - For FiLM models, location_name must be provided in the predict() method.
    """
    def __init__(
        self,
        bands: List[str],
        device: torch.device = torch.device("cpu"),
        wind: bool = True,
        multipass: bool = True,
        norm_data: bool = False,
        cloud_mask: bool = True,
        cat_mbmp: bool = True,
        norm_wind: bool = False,
        threshold_prediction: float = 0.5,
        corregister: bool = True,
        threshold_pixels: int = MINIMUM_NUMBER_PIXELS_PLUME,
        architecture: Optional[str] = None,
        logger: Optional[Logger] = None,
        weights: str = "weights/ch4_model.pt",
        film_dict_mapping: Optional[Dict[str, int]] = None,
        max_index_film: Optional[int] = None,
        film_train_zero_id: bool = False,
        use_zero_id_always: bool = False,
    ):
        super().__init__()

        self.wind = wind
        self.multipass = multipass
        self.cloud_mask = cloud_mask
        self.cat_mbmp = cat_mbmp
        self.norm_wind = norm_wind
        self.corregister = corregister
        self.film_dict_mapping = film_dict_mapping
        self.film = film_dict_mapping is not None
        self.film_train_zero_id = film_train_zero_id
        self.use_zero_id_always = use_zero_id_always
        self.architecture = architecture

        if multipass:
            self.in_channels = len(bands) * 2
        else:
            self.in_channels = len(bands)

        if wind:
            self.in_channels += 2

        if cloud_mask:
            self.in_channels += 1

        if self.cat_mbmp:
            self.in_channels += 1

        self.norm_data = norm_data
        self.device = device
        self.bands = bands

        if self.norm_data:
            self.set_norm_factors()

        self.architecture = self.architecture or "UnetOriginal"
        
        if self.film:
            assert (
                max_index_film is not None and max_index_film > 0
            ), "max_index_film must be provided for FiLM model and must greater than 0"
            if self.use_zero_id_always:
                assert self.film_train_zero_id, "use_zero_id_always requires film_train_zero_id"
            assert (
                self.architecture == "film"
            ), f"Found different architecture but FiLM is on {self.architecture}"
            self.model = UnetFiLMRefactor(
                in_channels=self.in_channels,
                out_channels=1,
                div_factor=1,
                one_param_per_channel=True,
                batch_norm=True,
                n_locs=max_index_film,
            )
        elif self.architecture == "UnetOriginal":
            self.model = UnetOriginal(
                in_channels=self.in_channels, out_channels=1, div_factor=1
            )
        elif self.architecture == "UnetPlusPlus":
            aux_params = None
            if logger is not None:
                logger.info("UnetPlusPlus")
            self.model = MyUnetPlusPlus(
                encoder_name="resnext50_32x4d",
                encoder_depth=5,
                decoder_channels=(256, 128, 64, 32, 16),
                encoder_weights=None,
                in_channels=self.in_channels,
                aux_params=aux_params,
                decoder_attention_type="scse",
                classes=1,
            )

        state_dict = load_weights(weights, device)
        self.model.load_state_dict(state_dict, strict=False)

        self.model.eval()
        self.model.to(self.device)

        self.threshold_prediction = threshold_prediction
        self.threshold_pixels = threshold_pixels
        if logger is None:
            self.logger = glogger
        else:
            self.logger = logger

    def scene_binary_prediction(
        self,
        pred_continuous: Union[GeoTensor, NDArray],
        threshold_prediction: Optional[float] = None,
        threshold_pixels: Optional[float] = None,
    ) -> Tuple[Union[GeoTensor, NDArray], float, bool]:
        """
        Convert continuous predictions to binary plume mask and compute scene-level detection score.
        Returns binary mask, scene probability (threshold for connected components), and boolean plume presence.

        Args:
            pred_continuous (Union[GeoTensor, NDArray]): Continuous prediction values (0-1 range).
            threshold_prediction (Optional[float], optional): Threshold for binarizing predictions. Defaults to self.threshold_prediction.
            threshold_pixels (Optional[float], optional): Minimum pixels for connected components. Defaults to self.threshold_pixels.

        Returns:
            Tuple[Union[GeoTensor, NDArray], float, bool]: Binary prediction mask, scene probability, and plume presence flag.
        """
        if threshold_prediction is None:
            threshold_prediction = self.threshold_prediction
        if threshold_pixels is None:
            threshold_pixels = self.threshold_pixels

        pred = binary_connected_prediction(
            pred_continuous,
            threshold_prediction=threshold_prediction,
            threshold_pixels=threshold_pixels,
        )

        scene_out = threshold_cutoff_connected_components(
            pred_continuous, threshold_pixels=threshold_pixels, tol=1e-3
        )
        isplume = pred.values.sum() > self.threshold_pixels

        return pred, float(scene_out), isplume

    def predict(
        self,
        image_predict: GeoTensor,
        background_image: Optional[GeoTensor] = None,
        wind_vector: Optional[NDArray] = None,
        validmask: Optional[GeoTensor] = None,
        location_name: Optional[str] = None,
    ) -> Tuple[GeoTensor, float, bool, GeoTensor]:
        """

        Args:
            image_predict: np.array (len(bands), H, W) of TOA reflectances multiplied by 10000
            background_image: np.array (len(bands), H, W) of TOA reflectances multiplied by 10000
            wind_vector: np.array (2,) of wind vector in m/s
            validmask: np.array (H, W) of valid pixels

        Returns:
            per_pixel_probability uint8 np.array (H, W): with interpretation {0: no-plume, 1: plume}
            scene_probability (float): probability of a scene having a plume
        """

        pred_continuous = self.predict_continuous(
            image_predict, background_image, wind_vector, validmask, location_name
        )

        discretestuff = self.scene_binary_prediction(pred_continuous)

        return discretestuff + (pred_continuous,)

    def tensor_to_predict(
        self,
        image_predict: GeoTensor,
        background_image: Optional[GeoTensor] = None,
        wind_vector: Optional[NDArray] = None,
        validmask: Optional[GeoTensor] = None,
    ) -> NDArray:
        """
        Prepares input tensor for model inference by concatenating spectral bands with optional multipass, wind, cloud mask, and MBMP features.
        Applies normalization (via self.means/self.stds or division by 5000) and returns a numpy array ready for model input.

        Args:
            image_predict (GeoTensor): Primary image tensor with shape (C, H, W) containing spectral bands.
            background_image (Optional[GeoTensor], optional): Background image for multipass mode. Required if self.multipass=True.
            wind_vector (Optional[NDArray], optional): Wind vector (U, V) in m/s. Required if self.wind=True.
            validmask (Optional[GeoTensor], optional): Binary mask indicating valid pixels. Required if self.cloud_mask=True.

        Returns:
            NDArray: Preprocessed tensor with shape (C_out, H, W) where C_out depends on enabled features.
        """
        assert (
            len(image_predict.shape) == 3
        ), f"Expected 3D tensor, found {len(image_predict.shape)}D tensor"
        assert image_predict.shape[0] == len(
            self.bands
        ), f"Expected {len(self.bands)} channels found {image_predict.shape[0]}"

        assert (
            not self.wind or wind_vector is not None
        ), "Wind vector must be provided if wind is True"
        assert (
            not self.multipass or background_image is not None
        ), "Background image must be provided if multipass is True"
        assert (
            not self.cloud_mask or validmask is not None
        ), "Validmask must be provided if cloud_mask is True"

        image_predict = image_predict.astype(np.float32)
        if self.multipass:
            # geoaling the images if needed
            assert (
                len(background_image.shape) == 3
            ), f"Expected 3D tensor for background, found {len(background_image.shape)}D tensor"
            assert background_image.shape[0] == len(
                self.bands
            ), f"Expected {len(self.bands)} channels for background found {background_image.shape[0]}"

            background_image = align_images(
                image_predict,
                background_image.astype(np.float32),
                corregister=self.corregister,
            )
            tensor = np.concatenate([image_predict.values, background_image.values], axis=0)
        else:
            tensor = image_predict.values

        # Normalize the radiances
        if self.norm_data:
            tensor = (tensor - self.means[:, np.newaxis, np.newaxis]) / self.stds[
                :, np.newaxis, np.newaxis
            ]
        else:
            tensor /= 5000

        if self.wind:
            assert wind_vector is not None, "Wind vector must be provided if wind is True"
            wind_vector = np.array(wind_vector).astype(np.float32)
            wind_image = (
                np.ones_like(tensor[:2]) * wind_vector[:, None, None]
            )  # (2, H, W) * (2, 1, 1) = (2, H, W)
            if self.norm_wind:
                wind_image /= 8
            tensor = np.concatenate([tensor, wind_image], axis=0)

        if self.cloud_mask:
            assert validmask is not None, "Validmask must be provided if cloud_mask is True"
            cloudmask = 1 - validmask.values.astype(np.float32)
            tensor = np.concatenate([tensor, cloudmask[np.newaxis]], axis=0)

        if self.cat_mbmp:
            mbmp = to_mbmp(
                torch.tensor(tensor),
                b11_index=len(self.bands) - 2,
                b12_index=len(self.bands) - 1,
                b11_index_prev=2 * len(self.bands) - 2,
                b12_index_prev=2 * len(self.bands) - 1,
            ).numpy()

            tensor = np.concatenate([mbmp[np.newaxis], tensor], axis=0)

        return tensor

    def _probabilities(
        self,
        image_predict: GeoTensor,
        background_image: Optional[GeoTensor] = None,
        wind_vector: Optional[NDArray] = None,
        validmask: Optional[GeoTensor] = None,
        location_name: Optional[str] = None,
    ) -> NDArray:
        """
        Runs the neural network model on input to produce per-pixel plume probabilities.
        Applies sigmoid activation to logits and returns a (H, W) probability map in [0, 1] range.

        Args:
            image_predict (GeoTensor): Primary image tensor with shape (C, H, W).
            background_image (Optional[GeoTensor], optional): Background image for multipass mode.
            wind_vector (Optional[NDArray], optional): Wind vector (U, V) in m/s.
            validmask (Optional[GeoTensor], optional): Binary mask indicating valid pixels.
            location_name (Optional[str], optional): Location name for FiLM conditioning. Required if self.film=True.

        Raises:
            ValueError: If location_name is None when FiLM is enabled, or if location not in film_dict_mapping and film_train_zero_id=False.

        Returns:
            NDArray: Per-pixel probability map with shape (H, W) and values in [0, 1].
        """
        tensor = self.tensor_to_predict(
            image_predict,
            background_image=background_image,
            wind_vector=wind_vector,
            validmask=validmask,
        )

        tensor_torch = torch.tensor(tensor, device=self.device)[None]  # Add batch dim

        if self.film:
            if self.use_zero_id_always:
                idx = 0
            elif location_name is None:
                raise ValueError("Location name must be provided if film is True")
            elif location_name not in self.film_dict_mapping:
                if self.film_train_zero_id:
                    self.logger.info(
                        f"Location name {location_name} NOT found in film_dict_mapping using zero id"
                    )
                else:
                    raise ValueError(
                        f"Location name {location_name} NOT found in film_dict_mapping"
                    )
                idx = 0
            else:
                self.logger.info(f"Location name {location_name} found in film_dict_mapping")
                idx = self.film_dict_mapping[location_name]

            data_input = {
                "site_ids": torch.tensor([idx], device=self.device),
                "y_context_ls0_0": tensor_torch,
            }
        else:
            data_input = {"y_context_ls0_0": tensor_torch}

        with torch.no_grad():
            logits = self.model(data_input)[0]  # Remove batch dim
            pred = torch.sigmoid(logits)

        probs = np.array(pred.cpu())

        # 16 because there are 4 downsampling steps in the model (2**4 = 16)
        # probs =  utils_torch.padded_predict(tensor, self, divisor=1, device=self.device)
        return probs

    def predict_continuous(
        self,
        image_predict: GeoTensor,
        background_image: Optional[GeoTensor] = None,
        wind_vector: Optional[NDArray] = None,
        validmask: Optional[GeoTensor] = None,
        location_name: Optional[str] = None,
    ) -> GeoTensor:
        """
        Runs inference on input to produce per-pixel plume probabilities as a GeoTensor.
        Optionally masks probabilities using the valid pixel mask and preserves geospatial metadata (CRS, transform).

        Args:
            image_predict (GeoTensor): Primary image tensor with shape (C, H, W).
            background_image (Optional[GeoTensor], optional): Background image for multipass mode.
            wind_vector (Optional[NDArray], optional): Wind vector (U, V) in m/s.
            validmask (Optional[GeoTensor], optional): Binary mask indicating valid pixels to apply to predictions.
            location_name (Optional[str], optional): Location name for FiLM conditioning. Required if self.film=True.

        Returns:
            GeoTensor: Probability map with shape (H, W), values in [0, 1], and geospatial metadata from image_predict.
        """
        probs = self._probabilities(
            image_predict, background_image, wind_vector, validmask, location_name
        )

        if validmask is not None:
            assert (
                validmask.shape == probs.shape
            ), f"Expected validmask shape {probs.shape} found {validmask.shape}"
            # TODO erode validmask to avoid border effects?
            probs = probs * validmask.values

        return GeoTensor(
            probs,
            transform=image_predict.transform,
            crs=image_predict.crs,
            fill_value_default=0,
        )

    def set_norm_factors(self) -> None:
        means = torch.tensor([MEANS_S2[S2_SAFE_reader.BANDS_S2_L1C.index(b)] for b in self.bands])
        stds = torch.tensor([STDS_S2[S2_SAFE_reader.BANDS_S2_L1C.index(b)] for b in self.bands])
        if self.multipass:
            means = torch.cat([means, means])
            stds = torch.cat([stds, stds])

        self.means = means  # .to(self.device)
        self.stds = stds  # .to(self.device)


BANDS_S2L_MODEL_S2 = ["B02", "B03", "B04", "B08", "B11", "B12"]
MODELS = {
    "MARS-S2L": {
        "weights": f"trained_models/MARSS2L_20250326/best_epoch",
        "config_experiment": f"trained_models/MARSS2L_20250326/config_experiment.json",
        "bands": BANDS_S2L_MODEL_S2,
        "multipass": True,
        "threshold_prediction": 0.5,
        "threshold_pixels": 100,
        "wind": True,
        "norm_wind": True,
        "cat_mbmp": True,
        "cloud_mask": True,
        "norm_data": False,
        "film": False,
        "film_train_zero_id": True,
    },
    "MARS-S2L-offshore": {
        "weights": f"trained_models/MARSS2L_off_20250523/best_epoch",
        "config_experiment": f"trained_models/MARSS2L_off_20250523/config_experiment.json",
        "bands": BANDS_S2L_MODEL_S2,
        "multipass": True,
        "threshold_prediction": 0.5,
        "threshold_pixels": 100,
        "wind": True,
        "norm_wind": True,
        "cat_mbmp": True,
        "cloud_mask": True,
        "norm_data": False,
        "film": False,
        "film_train_zero_id": True,
    },
    "CH4Net":  {
        "weights": f"trained_models/CH4Net_20250329/best_epoch",
        "config_experiment": f"trained_models/CH4Net_20250329/config_experiment.json",
        "bands": BANDS_S2L_MODEL_S2,
        "multipass": False,
        "threshold_prediction": 0.5,
        "threshold_pixels": 100,
        "wind": False,
        "norm_wind": True,
        "cat_mbmp": False,
        "cloud_mask": False,
        "norm_data": False,
        "film": False,
        "film_train_zero_id": True,
    },
}

import shutil

def load_model(
    model_name: Optional[str] = "MARS-S2L",
    weights_folder: Optional[str] = None,
    device: torch.device = torch.device("cpu"),
    logger: Optional[Logger] = None,
) -> MARSS2LModel:
    """
    Load the CH4 model for inference.
    It will download the weights from remote if not found in weights_folder.

    Args:
        model_name (Optional[str]): name of the model to load. Defaults to "MARS-S2L".
        weights_folder (str, optional): local folder to store the weights. Defaults to None.
            If None, the weights will be stored in ~/.georeader/
        device (torch.device, optional): device to load the model. Defaults to torch.device("cpu").
        logger (Logger, optional): logger instance. Defaults to None.
    
    Returns:
        MARSS2LModel: loaded model
    """
    assert model_name in MODELS, f"Model name {model_name} not in {MODELS.keys()}"
    model_info = MODELS[model_name]


    if weights_folder is None:
        weights_folder = os.path.join(os.path.expanduser('~'),".georeader")

    os.makedirs(weights_folder, exist_ok=True)

    architecture = model_info.get("architecture", None)
    weights_file_local = os.path.join(weights_folder, f"IMEO_UNEP_{model_name}.ckpt")
    
    if not os.path.exists(weights_file_local):
        if logger is not None:
            logger.info(
                f"Downloading weights from {model_info['weights']} to {weights_file_local}"
            )
        model_file_path = hf_hub_download(repo_id=REPO_ID,
                                          repo_type="dataset",
                                          filename=model_info["weights"],
                                          local_dir=weights_folder)
        shutil.move(model_file_path, weights_file_local)

    assert os.path.exists(
        weights_file_local
    ), f"Could not find weights file {weights_file_local}"

    if model_info["film"]:
        raise NotImplementedError("FiLM models are not supported in this version.")
    else:
        film_dict_mapping = None
        max_index_film = None

    film_train_zero_id = model_info.get("film_train_zero_id", False)
    use_zero_id_always = model_info.get("use_zero_id_always", False)
    threshold_pixels = model_info.get("threshold_pixels", MINIMUM_NUMBER_PIXELS_PLUME)

    return MARSS2LModel(
        device=device,
        weights=weights_file_local,
        bands=model_info["bands"],
        multipass=model_info["multipass"],
        wind=model_info["wind"],
        cloud_mask=model_info["cloud_mask"],
        cat_mbmp=model_info["cat_mbmp"],
        norm_wind=model_info["norm_wind"],
        threshold_prediction=model_info["threshold_prediction"],
        norm_data=model_info["norm_data"],
        film_dict_mapping=film_dict_mapping,
        max_index_film=max_index_film,
        threshold_pixels=threshold_pixels,
        architecture=architecture,
        logger=logger,
        film_train_zero_id=film_train_zero_id,
        use_zero_id_always=use_zero_id_always,
    )


def fix_weights_file_if_needed(
    weights_file: str, fs: fsspec.AbstractFileSystem, logger: Logger
):
    """
    Fixes the weights file so that it can be loaded with weights_only=True

    Args:
        weights_file (str): weights file path
        fs (fsspec.AbstractFileSystem): filesystem to use to load the weights from remote
        logger (Logger): logger object
    """
    assert fs.exists(weights_file), f"weights file {weights_file} not found"

    try:
        with fs.open(weights_file, "rb") as fh:
            state_dict = torch.load(fh, map_location="cpu", weights_only=True)
        logger.info(f"weights file {weights_file} is OK")
    except Exception as e:
        name_without_ext, ext = os.path.splitext(weights_file)
        weights_file_new = name_without_ext + "_onlyweights" + ext
        logger.opt(exception=True).error(
            f"Loading with weights_only=True failed. Trying to load with weights_only=False. Saving only weights to {weights_file_new}"
        )
        with fs.open(weights_file, "rb") as fh:
            state_dict = torch.load(fh, map_location="cpu", weights_only=False)

        state_dict_new = {k: state_dict[k] for k in ["model_state_dict", "optimizer_state_dict"]}
        with fs.open(weights_file_new, "wb") as fh:
            torch.save(state_dict_new, fh)


def load_weights(weights: str, device: torch.device) -> OrderedDict:
    """
    Load a PyTorch model state_dict from a weights file, removing any 'module.' or '_orig_mod.module.' prefixes.

    Args:
        weights (str): Path to the weights file.
        device (torch.device): Device to load the weights.

    Returns:
        OrderedDict: Cleaned state_dict ready for model loading.
    """
    state_dict = torch.load(weights, map_location=device, weights_only=True)["model_state_dict"]

    # remove "module." from weights name
    if next(iter(state_dict.keys())).startswith("module"):
        state_dict = OrderedDict(
            [(k.replace("module.", ""), state_dict[k]) for k in state_dict.keys()]
        )
    elif next(iter(state_dict.keys())).startswith("_orig_mod.module"):
        state_dict = OrderedDict(
            [(k.replace("_orig_mod.module.", ""), state_dict[k]) for k in state_dict.keys()]
        )
    return state_dict
