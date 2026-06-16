"""Background-image selection for the S2/Landsat MBMP retrieval.

:class:`BackgroundImageSelector` owns the full "given a target image, find its
background" workflow. In marss2l the candidate source is Google Earth Engine only
(no database). The two methods :meth:`BackgroundImageSelector.query_background_images`
and :meth:`BackgroundImageSelector.download_image` are the only overridable steps;
in marsml a subclass overrides them to read from the database / blob storage.

The filtering and ranking code is **pure**: it reads ``percentage_clear`` /
``observability`` as already-populated attributes and never downloads. Populating
them is the job of the candidate step — in marss2l that means downloading the cloud
mask locally for a bounded set of candidates; in marsml the values come from the DB.
"""

from __future__ import annotations

import logging
import math
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Optional, Protocol, TypeVar, runtime_checkable

import matplotlib.pyplot as plt
import numpy as np
from georeader.geotensor import GeoTensor
from georeader.plot import show

from marss2l.mars_sentinel2 import query_images
from marss2l.mars_sentinel2.ee import ee_initialize
from marss2l.mars_sentinel2.location_image import LocationImageProtocol, S2LLocationImage
from marss2l.mars_sentinel2.mixing_ratio_methane import difference_bands, ratio_IL
from marss2l.mars_sentinel2.s2lutils import RELATION_CHANNELS_S2_L89, download_image_and_angles

if TYPE_CHECKING:
    import uuid

# Filtering/sorting a list of background candidates preserves the element type, so the
# pure helpers are generic in it (bound to the LocationImageProtocol contract).
LocImageT = TypeVar("LocImageT", bound=LocationImageProtocol)

# cloudsen12 cloud-mask encoding: class 0 == clear/land == valid (same in marsml).
CLEAR_CLASS = 0

DEFAULT_BANDS_DIFFERENCES = ["B02", "B03", "B04", "B11"]

# Landsat-8/9 constellation: LC08/LO08 are both Landsat-8 and LC09/LO09 both Landsat-9.
# L8 and L9 share an orbit (8-day offset) so they are interchangeable as backgrounds and
# are grouped; a plain ``satellite[:2]`` prefix check could not express this. The older
# missions (LT04/LT05/LE07) are decades apart and are NOT interchangeable — each is its
# own constellation (see _satellite_constellation).
LANDSAT_89 = ("LC08", "LO08", "LC09", "LO09")


@runtime_checkable
class SimilarityCache(Protocol):
    """Pluggable cache of pairwise image similarities.

    Keys are symmetric in the two ``id_loc_image`` and qualified by the bands and
    corregistration flag the similarity was computed with.
    """

    def get(
        self, id_a: "uuid.UUID", id_b: "uuid.UUID", bands: tuple[str, ...], corregister: bool
    ) -> Optional[float]: ...

    def put(
        self,
        id_a: "uuid.UUID",
        id_b: "uuid.UUID",
        bands: tuple[str, ...],
        corregister: bool,
        similarity: float,
        metadata: dict[str, Any],
    ) -> None: ...


class InMemorySimilarityCache:
    """Default in-memory :class:`SimilarityCache`."""

    def __init__(self) -> None:
        self._d: dict[Any, tuple[float, dict[str, Any]]] = {}

    @staticmethod
    def _key(
        id_a: "uuid.UUID", id_b: "uuid.UUID", bands: tuple[str, ...], corregister: bool
    ) -> tuple:
        return (frozenset((id_a, id_b)), tuple(bands), bool(corregister))

    def get(
        self, id_a: "uuid.UUID", id_b: "uuid.UUID", bands: tuple[str, ...], corregister: bool
    ) -> Optional[float]:
        entry = self._d.get(self._key(id_a, id_b, bands, corregister))
        return None if entry is None else entry[0]

    def put(
        self,
        id_a: "uuid.UUID",
        id_b: "uuid.UUID",
        bands: tuple[str, ...],
        corregister: bool,
        similarity: float,
        metadata: dict[str, Any],
    ) -> None:
        self._d[self._key(id_a, id_b, bands, corregister)] = (similarity, metadata)


class BackgroundImageSelector:
    """Select the background image for the MBMP retrieval (GEE-only candidate source)."""

    def __init__(
        self,
        method_bg_image: str = "most_similar",
        margin_days_background: int = 120,
        limit_images_most_similar: int = 20,
        threshold_max_noclear: float = 50.0,
        threshold_max_noclear_background_image: float = 5.0,
        threshold_max_noclear_background_image_second_round: float = 35.0,
        threshold_wind_similarity: float = 0.6,
        full_product_threshold: float = 95.0,
        bands_differences: Optional[list[str]] = None,
        cache: Optional[SimilarityCache] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if method_bg_image not in ("most_similar", "nearest_same_orbit"):
            raise ValueError(
                f"method_bg_image must be 'most_similar' or 'nearest_same_orbit', got {method_bg_image}"
            )
        self.method_bg_image = method_bg_image
        self.margin_days_background = margin_days_background
        self.limit_images_most_similar = limit_images_most_similar
        self.threshold_max_noclear = threshold_max_noclear
        self.threshold_max_noclear_background_image = threshold_max_noclear_background_image
        self.threshold_max_noclear_background_image_second_round = (
            threshold_max_noclear_background_image_second_round
        )
        self.threshold_wind_similarity = threshold_wind_similarity
        self.full_product_threshold = full_product_threshold
        self.bands_differences = bands_differences or list(DEFAULT_BANDS_DIFFERENCES)
        self.cache = cache
        self.logger = logger or logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Overridable steps (DB / storage in the marsml subclass)
    # ------------------------------------------------------------------
    def query_background_images(
        self,
        image_to_process: LocationImageProtocol,
        *,
        same_satellite: bool = False,
        force_same_orbit: bool = False,
        same_satellite_constellation: bool = True,
    ) -> list[S2LLocationImage]:
        """Fetch background candidates from GEE, already filtered and sorted.

        Downloads (and computes the local cloud mask for) candidates in expanding,
        date-ordered batches: it stops once ``limit_images_most_similar`` clear
        survivors are found, otherwise keeps going until a usable background is found
        or all metadata-survivors are exhausted. The downloaded images carry populated
        ``percentage_clear`` / ``observability``; before returning they are run through
        the two-pass cloud filter (:meth:`_filter_and_sort_background_images`) and sorted
        by date proximity, so the result is ready for similarity scoring. A subclass that
        overrides this step (e.g. the DB-backed one in marsml) is expected to return an
        already filtered-and-sorted list too.
        """
        ee_initialize()
        producttype, add_landsat457 = self._producttype(
            image_to_process, same_satellite, same_satellite_constellation
        )
        df = query_images.query_gee(
            image_to_process.location.geometry,
            date_start=image_to_process.tile_date - timedelta(days=self.margin_days_background),
            date_end=image_to_process.tile_date + timedelta(days=self.margin_days_background),
            producttype=producttype,
            add_landsat457=add_landsat457,
            with_wind=not image_to_process.location.offshore,
            logger=self.logger,
        )
        if df is None or len(df) == 0:
            return []

        # Discard near-fully-clouded scenes only (mirror full_product_threshold = 95).
        if "cloudcoverpercentage" in df.columns:
            df = df[df["cloudcoverpercentage"] <= self.full_product_threshold]

        candidates = [
            S2LLocationImage.from_gee_row(row, location=image_to_process.location)
            for _, row in df.iterrows()
        ]

        # Metadata-only prune (no pixels), then nearest-in-date first.
        candidates = [
            c
            for c in candidates
            if not self.filter_background_image(
                image_to_process,
                c,
                same_satellite=same_satellite,
                force_same_orbit=force_same_orbit,
                same_satellite_constellation=same_satellite_constellation,
                check_cloud=False,
            )
        ]
        candidates.sort(key=lambda c: abs(c.tile_date - image_to_process.tile_date))

        # Download (image + local cloud mask -> percentage_clear) in expanding batches.
        downloaded: list[S2LLocationImage] = []
        n_clear = 0
        for c in candidates:
            self.download_image(c)
            downloaded.append(c)
            if not self.filter_background_image(
                image_to_process,
                c,
                same_satellite=same_satellite,
                force_same_orbit=force_same_orbit,
                same_satellite_constellation=same_satellite_constellation,
            ):
                n_clear += 1
            if n_clear >= self.limit_images_most_similar:
                break

        # Filter (two-pass cloud loosening) and sort by date proximity before returning.
        return self._filter_and_sort_background_images(
            image_to_process,
            downloaded,
            same_satellite=same_satellite,
            force_same_orbit=force_same_orbit,
            same_satellite_constellation=same_satellite_constellation,
        )

    def download_image(self, image: S2LLocationImage) -> None:
        """Download pixels, cloud mask and angles from GEE; populate ``percentage_clear``.

        Passes the stored ``asset_id``/``crs``/``transform``/``tile`` so the GEE
        catalog is not re-queried.
        """
        if image.image is not None:
            return
        image_to_download = {
            "asset_id": image.asset_id,
            "gee_id": image.gee_id,
            "crs": image.crs,
            "transform": image.transform,
            "tile": image.tile,
        }
        # Forward the Sentinel-2 solar/view zenith angles (download_image_and_angles reads
        # them from these GEE properties for S2; Landsat computes them from the SZA/VZA bands).
        if image.sza is not None:
            image_to_download["MEAN_SOLAR_ZENITH_ANGLE"] = image.sza
        if image.vza is not None:
            image_to_download["MEAN_INCIDENCE_ZENITH_ANGLE_B12"] = image.vza

        geotensor, cloudmask, sza, vza, band_names = download_image_and_angles(
            geometry=image.location.geometry,
            image_to_download=image_to_download,
            logger=self.logger,
        )
        image.image = geotensor
        image.cloudmask = cloudmask
        image.band_names = band_names
        if sza is not None:
            image.sza = sza
        if vza is not None:
            image.vza = vza
        if image.percentage_clear < 0:
            image.percentage_clear = self.compute_percentage_clear(cloudmask)
            image.observability = (
                "cloudy" if image.percentage_clear < self.threshold_max_noclear else "clear"
            )

    # ------------------------------------------------------------------
    # Helpers (marsml names; derive from the protocol, never from a processor)
    # ------------------------------------------------------------------
    def validmask(self, image: LocationImageProtocol) -> Optional[GeoTensor]:
        """Boolean valid mask (``cloudmask == 0``, i.e. clear/land)."""
        if image.cloudmask is None:
            return None
        cloudmask = image.cloudmask.squeeze()
        # Build a new GeoTensor: this GeoTensor forbids in-place dtype changes (uint8 -> bool).
        validmask = GeoTensor(
            cloudmask.values == CLEAR_CLASS,
            transform=cloudmask.transform,
            crs=cloudmask.crs,
            fill_value_default=False,
        )
        assert len(validmask.shape) == 2, f"Invalid shape {validmask.shape}"
        return validmask

    def band_index(self, image: LocationImageProtocol, band: str) -> int:
        """Index of a logical S2 band name (e.g. ``"B11"``) in ``image.band_names``."""
        names = image.band_names
        if names is None:
            raise ValueError("image.band_names is None; download the image first")
        if band in names:
            return names.index(band)
        # Landsat: map the logical S2 band name to the L8/9 band name.
        target = RELATION_CHANNELS_S2_L89.get(band, band)
        return names.index(target)

    def compute_percentage_clear(self, cloudmask: GeoTensor) -> float:
        """Percentage of clear (class-0) pixels in the cloud mask."""
        vals = cloudmask.values
        total = vals.size
        if total == 0:
            return 0.0
        return float(100.0 * np.count_nonzero(vals == CLEAR_CLASS) / total)

    # ------------------------------------------------------------------
    # Shared algorithm (pure — no downloads)
    # ------------------------------------------------------------------
    def filter_background_image(
        self,
        image_to_process: LocationImageProtocol,
        image: LocationImageProtocol,
        *,
        same_satellite: bool = False,
        force_same_orbit: bool = False,
        same_satellite_constellation: bool = True,
        query_only_validated_images: bool = False,
        query_only_images_without_plumes: bool = False,
        threshold_max_noclear_background_image: Optional[float] = None,
        check_cloud: bool = True,
        margin_days_background: Optional[int] = None,
        verbose: bool = False,
    ) -> bool:
        """Return ``True`` if ``image`` is NOT a valid background for ``image_to_process``.

        Pure: reads attributes only, never downloads. ``check_cloud=False`` skips the
        observability / ``percentage_clear`` clauses (for the pre-download metadata prune).
        """
        if check_cloud and image.observability not in ["clear", "bad_retrieval"]:
            self._log(verbose, f"Filtering {image.tile}: observability {image.observability}")
            return True

        if query_only_validated_images and not image.validated:
            self._log(verbose, f"Filtering {image.tile}: not validated")
            return True

        if query_only_images_without_plumes and image.isplume:
            self._log(verbose, f"Filtering {image.tile}: has a plume")
            return True

        if same_satellite and (image.satellite != image_to_process.satellite):
            self._log(verbose, f"Filtering {image.tile}: satellite {image.satellite}")
            return True

        # Constellation restriction (Sentinel-2 / Landsat-8-9 / each older Landsat on its own);
        # see _satellite_constellation for why family lists, not a satellite[:2] prefix, are used.
        if same_satellite_constellation and (
            self._satellite_constellation(image.satellite)
            != self._satellite_constellation(image_to_process.satellite)
        ):
            self._log(
                verbose,
                f"Filtering {image.tile}: constellation {image.satellite} "
                f"!= {image_to_process.satellite}",
            )
            return True

        if force_same_orbit and image_to_process.satellite.startswith("S2"):
            if image.tile[33:37] != image_to_process.tile[33:37]:
                self._log(verbose, f"Filtering {image.tile}: relative orbit {image.tile[33:37]}")
                return True

        if check_cloud:
            if threshold_max_noclear_background_image is None:
                threshold_max_noclear_background_image = self.threshold_max_noclear_background_image
            if image.percentage_clear < (100 - threshold_max_noclear_background_image):
                self._log(
                    verbose,
                    f"Filtering {image.tile}: percentage clear {image.percentage_clear:.1f}%",
                )
                return True

        # Discard the target's own acquisition and any near-simultaneous pass from the same
        # satellite constellation (|Δt| < 5 min, as in marsml's s2l89_processor). S2A and S2C
        # flew in tandem for a few months, so the tandem twin's scene is useless as a
        # background: in the ~minutes between the two passes the plume has not moved. The key is
        # the *constellation*, not the exact satellite (marsml compares satellite, but that
        # would miss the S2A/S2C tandem). This relies on the target and candidate sharing the
        # same tile_date convention (the S2 datatake stamp) — see S2LLocationImage.from_tile.
        if (
            self._satellite_constellation(image.satellite)
            == self._satellite_constellation(image_to_process.satellite)
        ) and (
            abs((image.tile_date - image_to_process.tile_date).total_seconds()) < 5 * 60
        ):
            self._log(verbose, f"Filtering {image.tile}: same acquisition / tandem pass")
            return True

        if margin_days_background is None:
            margin_days_background = self.margin_days_background
        if abs((image.tile_date - image_to_process.tile_date).days) > margin_days_background:
            self._log(verbose, f"Filtering {image.tile}: outside {margin_days_background} days")
            return True

        # Exclude images with a plume blowing in a similar wind direction.
        if image.isplume and image.haswind():
            if not image_to_process.haswind():
                return False
            norm_c = math.sqrt(image_to_process.wind_u**2 + image_to_process.wind_v**2)
            norm_b = math.sqrt(image.wind_u**2 + image.wind_v**2)
            # A zero wind vector has no direction → can't assess alignment, so don't reject.
            if norm_c > 0 and norm_b > 0:
                cosine_similarity = (
                    image_to_process.wind_u / norm_c * image.wind_u / norm_b
                    + image_to_process.wind_v / norm_c * image.wind_v / norm_b
                )
                if cosine_similarity > self.threshold_wind_similarity:
                    self._log(verbose, f"Filtering {image.tile}: plume with similar wind")
                    return True

        return False

    def _filter_and_sort_background_images(
        self,
        image_to_process: LocationImageProtocol,
        background_images: list[LocImageT],
        *,
        same_satellite: bool = False,
        force_same_orbit: bool = False,
        same_satellite_constellation: bool = True,
        query_only_validated_images: bool = False,
        query_only_images_without_plumes: bool = False,
        margin_days_background: Optional[int] = None,
        verbose: bool = False,
    ) -> list[LocImageT]:
        """Filter candidates (two-pass cloud loosening) and sort by date proximity."""

        def _passes(bg, threshold):
            return not self.filter_background_image(
                image_to_process,
                bg,
                same_satellite=same_satellite,
                force_same_orbit=force_same_orbit,
                same_satellite_constellation=same_satellite_constellation,
                query_only_validated_images=query_only_validated_images,
                query_only_images_without_plumes=query_only_images_without_plumes,
                threshold_max_noclear_background_image=threshold,
                margin_days_background=margin_days_background,
                verbose=verbose,
            )

        filtered = [
            bg for bg in background_images if _passes(bg, self.threshold_max_noclear_background_image)
        ]

        if len(filtered) == 0:
            self.logger.info(
                f"No background image for {image_to_process.tile} below "
                f"{self.threshold_max_noclear_background_image}% cloud; retrying at "
                f"{self.threshold_max_noclear_background_image_second_round}%"
            )
            filtered = [
                bg
                for bg in background_images
                if _passes(bg, self.threshold_max_noclear_background_image_second_round)
            ]
            if len(filtered) == 0:
                return filtered

        filtered.sort(key=lambda bg: abs(bg.tile_date - image_to_process.tile_date))
        return filtered

    def background_images_most_similar_sorted(
        self,
        image_to_process: LocationImageProtocol,
        background_images: Optional[list[LocationImageProtocol]] = None,
        limit_images: Optional[int] = None,
        top: Optional[int] = None,
        bands_differences: Optional[list[str]] = None,
        corregister: bool = True,
    ) -> list[tuple[LocationImageProtocol, float]]:
        """Score candidates by band similarity; return ``[(image, difference), ...]`` ascending."""
        if background_images is None:
            # query_background_images already returns a filtered-and-sorted list.
            background_images = self.query_background_images(image_to_process)
        if not background_images:
            return []

        if image_to_process.image is None:
            self.download_image(image_to_process)
        validmask = self.validmask(image_to_process)

        bands = bands_differences or self.bands_differences
        bands_key = tuple(bands)
        limit = limit_images if limit_images is not None else self.limit_images_most_similar
        bands_indexes = [self.band_index(image_to_process, b) for b in bands]

        difference_values = []
        for bg_image in background_images[:limit]:
            cached = (
                self.cache.get(
                    image_to_process.id_loc_image, bg_image.id_loc_image, bands_key, corregister
                )
                if self.cache is not None
                else None
            )
            if cached is not None:
                difference_values.append((bg_image, cached))
                continue

            if bg_image.image is None:
                self.download_image(bg_image)
            validmask_bg = self.validmask(bg_image)
            difference_bg = difference_bands(
                image_to_process.image,
                bg_image.image,
                bands_indexes=bands_indexes,
                valid_mask_curr=validmask,
                valid_mask_bg=validmask_bg,
                corregister=corregister,
            )
            difference_value = float(
                np.mean(
                    difference_bg.values[difference_bg.values != difference_bg.fill_value_default]
                )
            )
            difference_values.append((bg_image, difference_value))
            if self.cache is not None:
                self.cache.put(
                    image_to_process.id_loc_image,
                    bg_image.id_loc_image,
                    bands_key,
                    corregister,
                    difference_value,
                    {"bands": list(bands), "corregister": corregister},
                )
            bg_image.image = None  # free after scoring

        difference_values.sort(key=lambda x: x[1])

        if difference_values:
            self.download_image(difference_values[0][0])  # reload the winner

        if top is not None:
            difference_values = difference_values[:top]
        return difference_values

    def compute_background_image(
        self,
        image_to_process: LocationImageProtocol,
        recompute_background_image: bool = False,
        method_bg_image: Optional[str] = None,
    ) -> Optional[S2LLocationImage]:
        """Return the chosen background image (pixels loaded), or ``None``.

        The single selection entry point. With ``recompute_background_image=False`` it
        reuses the in-memory choice in ``metadata["background_image"]`` if present.
        """
        if not recompute_background_image and "background_image" in image_to_process.metadata:
            return image_to_process.metadata["background_image"]

        method = method_bg_image or self.method_bg_image
        # query_background_images already returns a filtered-and-sorted candidate list.
        candidates = self.query_background_images(
            image_to_process,
            same_satellite=(method == "nearest_same_orbit"),
            force_same_orbit=(method == "nearest_same_orbit"),
        )
        image_to_process.metadata["background_images_list"] = candidates

        if not candidates:
            self.logger.info(f"No background image found for {image_to_process.tile}")
            image_to_process.metadata["background_image_not_found"] = True
            return None

        if method == "nearest_same_orbit":
            background_image = candidates[0]
            self.download_image(background_image)
        else:
            ranked = self.background_images_most_similar_sorted(image_to_process, candidates)
            background_image = ranked[0][0] if ranked else None

        if background_image is None:
            image_to_process.metadata["background_image_not_found"] = True
            return None

        image_to_process.metadata["background_image"] = background_image
        return background_image

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------
    def plot_all_differences(
        self,
        image_to_process: LocationImageProtocol,
        background_images: Optional[list[LocationImageProtocol | tuple[LocationImageProtocol, float]]] = None,
        n_images_plot: int = 16,
        bands_differences: Optional[list[str]] = None,
        corregister: bool = True,
    ):
        """Plot, per background, three panels: RGB | difference | MBMP I-L ratio.

        ``background_images`` may be a pre-ordered similarity list (e.g. from
        :meth:`background_images_most_similar_sorted`) so the panels read
        most-similar -> least-similar. Returns a matplotlib ``Figure``.
        """
        if background_images is None:
            ranked = self.background_images_most_similar_sorted(
                image_to_process, bands_differences=bands_differences, corregister=corregister
            )
            background_images = [bg for bg, _ in ranked]
        else:
            background_images = [
                bg[0] if isinstance(bg, tuple) else bg for bg in background_images
            ]
        if not background_images:
            return None

        if image_to_process.image is None:
            self.download_image(image_to_process)

        bands = bands_differences or self.bands_differences
        bands_indexes = [self.band_index(image_to_process, b) for b in bands]

        n = min(n_images_plot, len(background_images))
        background_images = background_images[:n]
        fig, ax = plt.subplots(n, 3, figsize=(15, 5 * n), tight_layout=True, squeeze=False)

        curr_img = image_to_process.image.astype(np.float64)
        validmask = self.validmask(image_to_process)
        rgb_idx = [self.band_index(image_to_process, b) for b in ("B04", "B03", "B02")]
        for i, bg_image in enumerate(background_images):
            if bg_image.image is None:
                self.download_image(bg_image)
            bg_img = bg_image.image.astype(np.float64)
            validmask_bg = self.validmask(bg_image)

            diffimage = difference_bands(
                curr_img,
                bg_img,
                bands_indexes=bands_indexes,
                valid_mask_curr=validmask,
                valid_mask_bg=validmask_bg,
                corregister=corregister,
            )
            diffs = np.mean(diffimage.values[diffimage.values != diffimage.fill_value_default])

            rgb_bg = (bg_image.image.isel({"band": rgb_idx}) / 4500).clip(0, 1)
            show(rgb_bg, ax=ax[i, 0], title=f"{bg_image.satellite} {bg_image.day}")
            show(
                diffimage,
                add_colorbar_next_to=True,
                ax=ax[i, 1],
                mask=True,
                vmin=0,
                vmax=0.07,
                title=f"Diff |{image_to_process.day}-{bg_image.day}|={diffs * 100:.2f}%",
            )
            mbmp = ratio_IL(
                curr_img,
                bg_img,
                b12_index=self.band_index(image_to_process, "B12"),
                b11_index=self.band_index(image_to_process, "B11"),
                b12_index_bg=self.band_index(bg_image, "B12"),
                b11_index_bg=self.band_index(bg_image, "B11"),
                plumemaskbool=None,
                normalize=True,
                validmask=validmask,
                validmask_bg=validmask_bg,
                corregister=corregister,
            )
            min_value = mbmp.values.min()
            show(mbmp, add_colorbar_next_to=True, ax=ax[i, 2], vmax=1,
                 cmap="plasma_r",
                 vmin=max(min_value, 0.92), title="MBMP I-L")
            for axs in ax[i]:
                axs.axis("off")
        return fig

    # ------------------------------------------------------------------
    def _producttype(
        self,
        image_to_process: LocationImageProtocol,
        same_satellite: bool,
        same_satellite_constellation: bool,
    ) -> tuple[str, bool]:
        """GEE ``producttype`` and ``add_landsat457`` for the target's constellation."""
        sat = image_to_process.satellite
        if not same_satellite_constellation:
            add457 = not (sat.startswith("S2") or sat in ["LC08", "LO08", "LC09", "LO09"])
            return "both", add457
        if sat.startswith("S2"):
            return "S2", False
        if sat in ["LC08", "LO08", "LC09", "LO09"]:
            return "Landsat", False
        if sat in ["LE07", "LT05", "LT04"]:
            return "Landsat", True
        raise ValueError(f"Unknown satellite {sat}")

    @staticmethod
    def _satellite_constellation(satellite: str) -> str:
        """Constellation key grouping satellites whose images are interchangeable backgrounds.

        Sentinel-2 (S2A/S2B/S2C…) is one group and Landsat-8-9 (LC08/LO08/LC09/LO09) another.
        The older Landsat missions (LT04/LT05/LE07) are decades apart and not interchangeable,
        so each falls through to its own key (returned verbatim) — an L4 scene is never a
        background for an L5, etc. An unknown satellite is likewise its own group.
        """
        if satellite.startswith("S2"):
            return "S2"
        if satellite in LANDSAT_89:
            return "Landsat-8-9"
        return satellite

    def _log(self, verbose: bool, message: str) -> None:
        if verbose:
            self.logger.info(message)
