import geopandas as gpd
import os
from shapely import validation

COUNTRY_GDF = None
LINK_GEOJSON_COUNTRIES = (
    "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson"
)
# https://public.opendatasoft.com/api/records/1.0/search/?dataset=world-administrative-boundaries&q=&rows=-1&facet=status&facet=color_code&facet=continent&facet=region&timezone=UTC

# {'Ashmore and Cartier Islands',
#  'Brunei',
#  'Hong Kong S.A.R.',
#  'Iran',
#  'Ivory Coast',
#  'Moldova',
#  'Palestine',
#  'Republic of Serbia',
#  'Russia',
#  'South Korea',
#  'Syria',
#  'Turkey',
#  'Vietnam'}

FIX_COUNTRY_NAMES = {
    "Ashmore and Cartier Islands": "Australia",
    "Brunei": "Brunei Darussalam",
    "Hong Kong S.A.R.": "China",
    "Ivory Coast": "Côte d'Ivoire",
    "Iran": "Iran (Islamic Republic of)",
    "Republic of Serbia": "Serbia",
    "Palestine": "Gaza",
    "Russia": "Russian Federation",
    "South Korea": "Republic of Korea",
    "Moldova": "Republic of Moldova",
    "Syria": "Syrian Arab Republic",
    "Turkey": "Türkiye",
    "Vietnam": "Viet Nam",
}

def _export_original_countries(output_path:str):
    import fsspec
    with fsspec.open(LINK_GEOJSON_COUNTRIES, "rb") as f:
        countries = gpd.read_file(f)
    
    countries["geometry"] = [validation.make_valid(g) for g in countries["geometry"]]
    countries["name"] = countries["name"].apply(lambda x: FIX_COUNTRY_NAMES.get(x, x))
    countries["ADMIN"] = countries["name"]
    countries["romnam"] = countries["ADMIN"]
    countries["iso3cd"] = countries["ISO3166-1-Alpha-3"]
    countries.to_file(output_path, driver="GPKG")

def all_countries(cache: bool = True) -> gpd.GeoDataFrame:
    if cache:
        global COUNTRY_GDF
        if COUNTRY_GDF is not None:
            return COUNTRY_GDF

    home_dir = os.path.join(os.path.expanduser("~"), ".georeader")
    name_file = os.path.splitext(os.path.basename(LINK_GEOJSON_COUNTRIES))[0]
    name_file = name_file+".gpkg"
    countries_gpkg = os.path.join(home_dir, name_file)

    if not os.path.exists(countries_gpkg):
        os.makedirs(home_dir, exist_ok=True)
        _export_original_countries(countries_gpkg)

    countries = gpd.read_file(countries_gpkg, driver="GPKG")

    if cache:
        COUNTRY_GDF = countries

    return countries