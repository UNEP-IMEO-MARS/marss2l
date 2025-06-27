from . import mixing_ratio_methane
from scipy import interpolate
from typing import Callable, Union, Tuple, Optional, Dict, Any, List
from numpy.typing import NDArray
from georeader.geotensor import GeoTensor
import numpy as np
import json


class TransmittanceCH4Interpolation:
    def __init__(self, 
                 amf_arr:NDArray,
                 mr_ch4_arr:NDArray,
                 background_concentration:float=mixing_ratio_methane.BACKGROUND_CONCENTRATION) -> None:
        """
    
        Class to do interpolation from transmittances to methane concentrations and vice versa.

        Args:
            amf_arr (str, optional): Array with the air mass factor values.
            mr_ch4_arr (str, optional): Array with the methane concentration values.
            background_concentration (float, optional): Background concentration of methane in ppb. Defaults to `mixing_ratio_methane.BACKGROUND_CONCENTRATION`.
            
        """
        self.background_concentration = background_concentration
        
        self.amf_arr = amf_arr
        self.amf_arr_max = self.amf_arr.max()

        self.mr_ch4_arr = mr_ch4_arr
        self.f_ch4_mr = interpolate.interp1d(self.amf_arr, self.mr_ch4_arr, axis=0, kind='cubic')
        
        self.cache:Dict[str, Callable[[NDArray], NDArray]] = {}
    
    def load_interpolation_functions_satellite(self, satellite:str) -> Dict[str, Callable[[NDArray], NDArray]]:
        if satellite in self.cache:
            return self.cache[satellite]
        
        self.cache[satellite] = self._load_interpolation_functions_satellite(satellite)
        return self.cache[satellite]
    
    def _load_interpolation_functions_satellite(self, satellite:str) -> Dict[str, Callable[[NDArray], NDArray]]:
        raise NotImplementedError("This method should be implemented in the subclass.")

    def air_mass_factor(self, sza:float, vza:float) -> float:
        return np.clip(mixing_ratio_methane.air_mass_factor(sza, vza), None, self.amf_arr_max)
    
    def _transmittance_B12_interpfun(self, satellite:str, amf:float) -> Callable[[NDArray], NDArray]:
        interpfun_B12 = self.load_interpolation_functions_satellite(satellite)["interpolation_funtion_amf_transmittance_b12"]
        transmittance_b12_arr_1d = interpfun_B12(amf)
        mr_ch4_arr_1d = self.f_ch4_mr(amf)
        return interpolate.interp1d(mr_ch4_arr_1d, transmittance_b12_arr_1d , axis=0,
                                    fill_value="extrapolate", kind='cubic')
    
    def _transmittance_B11_interpfun(self, satellite:str, amf:float) -> Callable[[NDArray], NDArray]:
        interpfun_B11 = self.load_interpolation_functions_satellite(satellite)["interpolation_funtion_amf_transmittance_b11"]
        interpfun_B12 = self.load_interpolation_functions_satellite(satellite)["interpolation_funtion_amf_transmittance_b12"]
        transmittance_b11_arr_1d = interpfun_B11(amf)
        transmittance_b12_arr_1d = interpfun_B12(amf)

        return interpolate.interp1d(transmittance_b12_arr_1d, transmittance_b11_arr_1d , axis=0,
                                    fill_value="extrapolate", kind='cubic')
    
    def _transmittance_B12_bg(self, satellite:str, amf:float) -> NDArray:
        interpfun_B12_bg = self.load_interpolation_functions_satellite(satellite)["interpolation_funtion_amf_transmittance_b12_bg"]
        return interpfun_B12_bg(amf)
    
    def _transmittance_B11_bg(self, satellite:str, amf:float) -> NDArray:
        interpfun_B11_bg = self.load_interpolation_functions_satellite(satellite)["interpolation_funtion_amf_transmittance_b11_bg"]
        return interpfun_B11_bg(amf)

    def transmittance_B12_B11(self, satellite:str, sza:float, vza:float, deltach4:Union[NDArray,GeoTensor]) -> Tuple[Union[NDArray,GeoTensor], Union[NDArray,GeoTensor]]:
        r"""
        Returns the transmittance correction factor for bands B12 and B11 that should be applied to an image 
        without a plume to simulate the presence of a plume with a given methane concentration.    

        For band 12 this is given by:
        $$
            T_{B12}^{corr} = \frac{ \int_{B12} E_g(\lambda) T_{atm}(\lambda) T_{\Delta \text{CH}_4}(\lambda) d\lambda}{\int_{B12} E_g(\lambda) T_{atm}(\lambda) d\lambda}
        $$

        For band 11 we interpolate the transmittance of B12 to get the transmittance of B11.

        Args:
            satellite (str): Name of the satellite. Either 'S2A', 'S2B', 'LC08' or 'LC09'.
            sza (float): Solar zenith angle in degrees.
            vza (float): View zenith angle in degrees.
            deltach4 (Union[NDArray,GeoTensor]): Image with $\Delta$CH$_4$: the methane concentration in ppb.

        Returns:
            Union[NDArray,GeoTensor]: transmittance_b12, transmittance_b11
              Image with the transmittance correction factor for bands B12 and B11.
        """
        amf = self.air_mass_factor(sza, vza)
        interpfun_B12 = self._transmittance_B12_interpfun(satellite, amf)
        interpfun_B11 = self._transmittance_B11_interpfun(satellite, amf)
        
        ch4 = self._deltach4_to_ch4(deltach4)
        transmittance_b12 = mixing_ratio_methane.apply_interpfun_to_image(interpfun_B12, ch4, fill_value_default=1)
        transmittance_b11 = mixing_ratio_methane.apply_interpfun_to_image(interpfun_B11, transmittance_b12, fill_value_default=1)
        
        transmittance_b12_bg = self._transmittance_B12_bg(satellite, amf)
        transmittance_b11_bg = self._transmittance_B11_bg(satellite, amf)

        if isinstance(deltach4, GeoTensor):
            transmittance_b12_corr = transmittance_b12.copy()
            transmittance_b11_corr = transmittance_b11.copy()
            invalids = transmittance_b12.values == transmittance_b12.fill_value_default
            transmittance_b11_corr.values/= transmittance_b11_bg
            transmittance_b12_corr.values/= transmittance_b12_bg

            # Clip the values to be between 0 and 1
            # transmittance_b11_corr.values = np.clip(transmittance_b11_corr.values, 0, 1)
            # transmittance_b12_corr.values = np.clip(transmittance_b12_corr.values, 0, 1)
            
            transmittance_b11_corr.values[invalids] = transmittance_b11.fill_value_default
            transmittance_b12_corr.values[invalids] = transmittance_b12.fill_value_default
        else:
            transmittance_b12_corr = transmittance_b12/transmittance_b12_bg
            transmittance_b11_corr = transmittance_b11/transmittance_b11_bg

            # Clip the values to be between 0 and 1
            # transmittance_b11_corr = np.clip(transmittance_b11_corr, 0, 1)
            # transmittance_b12_corr = np.clip(transmittance_b12_corr, 0, 1)

        return transmittance_b12_corr, transmittance_b11_corr
    
    def _deltach4_to_ch4(self, deltach4:Union[NDArray, GeoTensor]) -> Union[NDArray, GeoTensor]:
        r"""
        Returns deltach4 + background_concentration.

        Args:
            deltach4 (Union[NDArray, GeoTensor]): Image with $\Delta$CH$_4$: the methane concentration in ppb.

        Returns:
            Union[NDArray, GeoTensor]: Image with CH$_4$: the methane concentration in ppb.
        """
        if isinstance(deltach4, GeoTensor):
            ch4 = deltach4.copy()
            invalids = ch4.values == ch4.fill_value_default
            ch4.values+= self.background_concentration
            ch4.values[invalids] = 0
            ch4.fill_value_default = 0
        else:
            ch4 = deltach4 + self.background_concentration
        
        return ch4

    def deltach4_from_ratio_transmittance(self, satellite:str, sza:float, vza:float, ratio_il:Union[NDArray, GeoTensor]) -> Union[NDArray, GeoTensor]:
        r"""
        Returns the methane concentration in ppb from the observed ratio of the transmittance of bands B12 and B11.

        $$
            IL = \frac{L_{B12}^{\text{plume}} /L_{B12}}{L_{B11}^{\text{plume}}/L_{B11}}
        $$

        Args:
            satellite (str): name of the satellite. Either 'S2A', 'S2B', 'LC08' or 'LC09'.
            sza (float): solar zenith angle in degrees.
            vza (float): view zenith angle in degrees.
            ratio_il (Union[NDArray, GeoTensor]): Observed ratio of transmittance between two images.

        Returns:
            Union[NDArray, GeoTensor]: Image with $\Delta$CH$_4$: the methane concentration in ppb.
        """
        amf = self.air_mass_factor(sza, vza)
        mr_ch4_arr_1d = self.f_ch4_mr(amf)

        interpfun_B12 = self._transmittance_B12_interpfun(satellite, amf)
        interpfun_B11 = self._transmittance_B11_interpfun(satellite, amf)
        transmittance_b12 = interpfun_B12(mr_ch4_arr_1d)
        transmittance_b11 = interpfun_B11(transmittance_b12)
        
        transmittance_b12_bg = self._transmittance_B12_bg(satellite, amf)
        transmittance_b11_bg = self._transmittance_B11_bg(satellite, amf)

        ratio_il_corrected = ratio_il * transmittance_b12_bg/transmittance_b11_bg

        interpfun = interpolate.interp1d(transmittance_b12/transmittance_b11, mr_ch4_arr_1d, axis=0,
                                         fill_value="extrapolate", kind='cubic')
        final_fun = lambda x: np.clip(interpfun(x) - self.background_concentration, 0, None)
        
        return mixing_ratio_methane.apply_interpfun_to_image(final_fun, ratio_il_corrected)
        

class TransmittanceCH4InterpolationFromLUT(TransmittanceCH4Interpolation):
    def __init__(self, lut_file:str=mixing_ratio_methane.FILE_LUT_GAS, 
                 with_ltoa_correction:bool=True, 
                 background_concentration:float=mixing_ratio_methane.BACKGROUND_CONCENTRATION,
                 trans_tot_as_tbg:bool=False):
        wvl_mod, t_full_arr, mr_ch4_arr, amf_arr, eg_arr, trans_tot_arr = mixing_ratio_methane.load_all_lut(lut_file)
        super().__init__(amf_arr, mr_ch4_arr, background_concentration=background_concentration)

        self.with_ltoa_correction = with_ltoa_correction
        self.trans_tot_as_tbg = trans_tot_as_tbg
        
        if self.trans_tot_as_tbg:
            if not self.with_ltoa_correction:
                raise ValueError("The total atmospheric transmittance is only used when the LTOA correction is enabled.\n Set `trans_tot_as_tbg=False` or `with_ltoa_correction=True`.")
        
        self.wvl_mod, self.t_full_arr, self.mr_ch4_arr, self.eg_arr, self.trans_tot_arr = wvl_mod, t_full_arr, mr_ch4_arr, eg_arr, trans_tot_arr
        
        self.f_t_amf = interpolate.interp1d(self.amf_arr, self.t_full_arr, axis=0, kind='cubic')

        self._t_background_transmittance:Optional[NDArray] = None
    
    @property
    def t_background_transmittance(self) -> NDArray:
        if self._t_background_transmittance is not None:
            return self._t_background_transmittance
        
        # Calculate the transmittance for the background concentration
        self._t_background_transmittance = np.zeros((len(self.amf_arr), len(self.wvl_mod)), 
                                                   dtype=self.t_full_arr.dtype)
        for i,amf in enumerate(self.amf_arr):
            transmittance_fun_for_cch4 = self._transmittance_fun_for_cch4(amf)
            self._t_background_transmittance[i] = transmittance_fun_for_cch4(self.background_concentration)
        return self._t_background_transmittance
    
    def _transmittance_fun_for_cch4(self, amf:float) -> Callable[[NDArray], NDArray]:
        return interpolate.interp1d(self.f_ch4_mr(amf), self.f_t_amf(amf) , axis=0,
                                    fill_value="extrapolate", kind='cubic')
    
    def _load_interpolation_functions_satellite(self, satellite:str) -> Dict[str, Callable[[NDArray], NDArray]]:        
        self.b12srf_interpfunc, self.b11srf_interpfunc = mixing_ratio_methane.load_srfinterpfun(satellite)
        b12srf_interp = self.b12srf_interpfunc(self.wvl_mod) # (len(wvl_mod),)
        b11srf_interp = self.b11srf_interpfunc(self.wvl_mod) # (len(wvl_mod),)

        if self.with_ltoa_correction:
            # \int_{B12} E_g(\lambda) T_{atm}(\lambda) T_{Bg\text{CH}_4 + \Delta \text{CH}_4}(\lambda) d\lambda
            if self.trans_tot_as_tbg:
                transmittance_b12 = (self.eg_arr * self.t_background_transmittance[:, np.newaxis, :] * self.t_full_arr  * b12srf_interp).sum(axis=-1) / b12srf_interp.sum() # (8, 9)
                transmittance_b12_bg = (self.eg_arr * self.t_background_transmittance[:, np.newaxis, :] * b12srf_interp).sum(axis=-1) / b12srf_interp.sum() # (8,1)
                transmittance_b12_bg = transmittance_b12_bg[:,0] # (8,)
                # transmittance_b12/= transmittance_b12_bg
            else:
                transmittance_b12 = (self.eg_arr * self.trans_tot_arr * self.t_full_arr / self.t_background_transmittance[:, np.newaxis, :] * b12srf_interp).sum(axis=-1) / b12srf_interp.sum() # (8, 9)
                transmittance_b12_bg = (self.eg_arr * self.trans_tot_arr * b12srf_interp).sum(axis=-1) / b12srf_interp.sum() # float
                transmittance_b12_bg = np.repeat(transmittance_b12_bg, len(self.amf_arr)) # (8,)
                # transmittance_b12/= transmittance_b12_bg
        else:
            transmittance_b12 = ((self.t_full_arr * b12srf_interp).sum(axis=-1) / b12srf_interp.sum()) # (8, 9)
            transmittance_b12_bg = (self.t_background_transmittance[:, np.newaxis, :] * b12srf_interp).sum(axis=-1) / b12srf_interp.sum() # (8,1)
            transmittance_b12_bg = transmittance_b12_bg[:,0] # (8,)
            # transmittance_b12/= transmittance_b12_bg

        transmittance_b11 = ((self.t_full_arr * b11srf_interp).sum(axis=-1) / b11srf_interp.sum()) # (8, 9)
        transmittance_b11_bg = (self.t_background_transmittance[:, np.newaxis, :] * b11srf_interp).sum(axis=-1) / b11srf_interp.sum() # (8,1)
        # transmittance_b11/= transmittance_b11_bg

        interpolation_funtion_amf_transmittance_b12 = interpolate.interp1d(self.amf_arr, transmittance_b12, axis=0, kind='cubic')
        interpolation_funtion_amf_transmittance_b11 = interpolate.interp1d(self.amf_arr, transmittance_b11, axis=0, kind='cubic')
        interpolation_funtion_amf_transmittance_b12_bg = interpolate.interp1d(self.amf_arr, transmittance_b12_bg, axis=0, kind='cubic')
        interpolation_funtion_amf_transmittance_b11_bg = interpolate.interp1d(self.amf_arr, transmittance_b11_bg, axis=0, kind='cubic')

        return {
            "interpolation_funtion_amf_transmittance_b12": interpolation_funtion_amf_transmittance_b12,
            "interpolation_funtion_amf_transmittance_b11": interpolation_funtion_amf_transmittance_b11,
            "interpolation_funtion_amf_transmittance_b12_bg": interpolation_funtion_amf_transmittance_b12_bg,
            "interpolation_funtion_amf_transmittance_b11_bg": interpolation_funtion_amf_transmittance_b11_bg,
            "transmittance_b12": transmittance_b12,
            "transmittance_b11": transmittance_b11,
            "transmittance_b12_bg": transmittance_b12_bg,
            "transmittance_b11_bg": transmittance_b11_bg          
        }
    

def transmittances_to_export(satellites:List[str]=["S2A", "S2B", "LC08", "LC09", "LE07", "LT04", "LT05"],
                             lut_file:str=mixing_ratio_methane.FILE_LUT_GAS,
                             with_ltoa_correction:bool=True, 
                             background_concentration:float=mixing_ratio_methane.BACKGROUND_CONCENTRATION,
                             trans_tot_as_tbg:bool=False) -> Dict[str, Any]:
    """
    Export the transmittance interpolation functions to a dictionary.

    Args:
        satellites (List[str], optional): Satellites to export the transmittance interpolation functions.
                   Defaults to ["S2A", "S2B", "LC08", "LC09", "LE07", "LT04", "LT05"].
        lut_file (str, optional): Path to the LUT file. Defaults to mixing_ratio_methane.FILE_LUT_GAS.
        with_ltoa_correction (bool, optional): Apply the LTOA correction factor. Defaults to True.
        background_concentration (float, optional): Background concentration of methane in ppb. Defaults to mixing_ratio_methane.BACKGROUND_CONCENTRATION.
        trans_tot_as_tbg (bool, optional): Use the background transmittance as the total atmospheric transmittance. That assumes that 
            there's only methane absoption in the integrated wavelengths. Defaults to False.

    Returns:
        Dict[str, Any]:  Dictionary with the transmittance interpolation functions.
    """
    out:Dict[str, Any] = {}

    simulator = TransmittanceCH4InterpolationFromLUT(lut_file=lut_file,
                                                     with_ltoa_correction=with_ltoa_correction,
                                                     background_concentration=background_concentration,
                                                     trans_tot_as_tbg=trans_tot_as_tbg)
    out["amf_arr"] = simulator.amf_arr.tolist()
    out["mr_ch4_arr"] = simulator.mr_ch4_arr.tolist()
    out["background_concentration"] = background_concentration
    out[with_ltoa_correction] = with_ltoa_correction
    out[trans_tot_as_tbg] = trans_tot_as_tbg
    out[lut_file] = lut_file

    for satellite in satellites:
        out_sat = simulator.load_interpolation_functions_satellite(satellite)
        out[satellite]  = {}
        for tname in ["transmittance_b12", "transmittance_b11", "transmittance_b12_bg", "transmittance_b11_bg"]:
            out[satellite][tname] = out_sat[tname].tolist()
    
    return out   
    
LUT_PUBLIC_FILE = "az://public/MARS-S2L/lut/integrated_transmittances.json"

class TransmittanceCH4InterpolationFromDict(TransmittanceCH4Interpolation):
    def __init__(self, dict_or_json_file:Union[str, Dict[str, Any]]=LUT_PUBLIC_FILE):
        if isinstance(dict_or_json_file, str):
            if dict_or_json_file.startswith("az://public"):
                from adlfs import AzureBlobFileSystem
                fs = AzureBlobFileSystem(account_name="unepazeconomyadlsstorage")
                with fs.open(dict_or_json_file, "r") as fh:
                    data = json.load(fh)
            else:
                with open(dict_or_json_file, "r") as f:
                    data = json.load(f)
        else:
            data = dict_or_json_file
        super().__init__(np.array(data["amf_arr"]), 
                         np.array(data["mr_ch4_arr"]),
                         background_concentration=data["background_concentration"])
        
        self.data = data
    
    def _load_interpolation_functions_satellite(self, satellite: str) -> Dict[str, Callable[[NDArray], NDArray]]:
        
        if satellite not in self.data:
            raise ValueError(f"The satellite {satellite} is not in the data.")
        
        data_sat = self.data[satellite]

        transmittance_b12 = np.array(data_sat["transmittance_b12"])
        transmittance_b11 = np.array(data_sat["transmittance_b11"])
        transmittance_b12_bg = np.array(data_sat["transmittance_b12_bg"])
        transmittance_b11_bg = np.array(data_sat["transmittance_b11_bg"])

        interpolation_funtion_amf_transmittance_b12 = interpolate.interp1d(self.amf_arr, transmittance_b12, axis=0, kind='cubic')
        interpolation_funtion_amf_transmittance_b11 = interpolate.interp1d(self.amf_arr, transmittance_b11, axis=0, kind='cubic')
        interpolation_funtion_amf_transmittance_b12_bg = interpolate.interp1d(self.amf_arr, transmittance_b12_bg, axis=0, kind='cubic')
        interpolation_funtion_amf_transmittance_b11_bg = interpolate.interp1d(self.amf_arr, transmittance_b11_bg, axis=0, kind='cubic')

        return {
            "interpolation_funtion_amf_transmittance_b12": interpolation_funtion_amf_transmittance_b12,
            "interpolation_funtion_amf_transmittance_b11": interpolation_funtion_amf_transmittance_b11,
            "interpolation_funtion_amf_transmittance_b12_bg": interpolation_funtion_amf_transmittance_b12_bg,
            "interpolation_funtion_amf_transmittance_b11_bg": interpolation_funtion_amf_transmittance_b11_bg,
        }
    
    
        