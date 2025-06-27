import matplotlib

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

from .inference import inference_function_from_torch_model
from .plot_location_for_inference import plot_location
from .plot_histograms_predictions import plot_row
from .time_series import plot_time_series
from .prob_vs_emission_rate import plot_prob_vs_emission_rate
from .images_by_month import plot_images_by_month
from .geographical_location import plot_geographical_location
from .recall_fluxrate_plot import plot_recall_fpr_fluxrate
from .colors import PALETTE_ALL, C0, C1, C2, C3, C4


