import os
import requests
import h3
import geopy.distance
import geopy.point
from openai import OpenAI
import io
import base64
import git
import math
import matplotlib.pyplot as plt
import contextily as cx
from pyproj import Transformer, CRS
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
import time
import geopandas as gpd
from shapely.geometry import Point

# --- CONFIGURATION CONSTANTS ---
SEED = 42
SAMPLE_SIZE = 1000
SPATIAL_RELATION_WITH_AZIMUTH = {
    'north': 0, 'northeast': 45, 'east': 90, 'southeast': 135,
    'south': 180, 'southwest': 225, 'west': 270, 'northwest': 315,
    'between': None, 'near': None
}
H3_RESOLUTIONS = [5, 6, 7]
NATURAL_DISASTERS = ['storm', 'flood', 'landslide', 'wild fire']
COLUMNS = [
    'landmark_id', 'landmark_name', 'asciiname', 'alternatenames',
    'landmark_latitude', 'landmark_longitude', 'feature_class',
    'landmark_feature_code', 'landmark_country_code', 'cc2', 'admin1_code',
    'admin2_code', 'admin3_code', 'admin4_code', 'landmark_population',
    'elevation', 'dem', 'timezone', 'modification_date'
]

load_dotenv()
API_KEY = os.getenv("SAIA_API_KEY")
BASE_URL = "https://chat-ai.academiccloud.de/v1"
MODEL = "gemma-3-27b-it"
GEONAMES_USERNAME = os.getenv("GEONAMES_USERNAME")
TILE_PROVIDER = cx.providers.OpenStreetMap.Mapnik

repo_root = git.Repo('.', search_parent_directories=True).working_tree_dir

def configure_logging(logfilename):
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)
    file_handler = logging.FileHandler(logfilename)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(log_formatter)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(include_us=True, include_eu=False, file_path=f'{repo_root}/data/gandr_prelims/cities5000.txt'):
    logging.info("Loading data from file...")
    df = pd.read_csv(
        file_path,
        sep='\t',
        header=None,
        names=COLUMNS,
        usecols=['landmark_id', 'landmark_name', 'landmark_latitude',
                 'landmark_longitude', 'landmark_feature_code',
                 'landmark_population', 'landmark_country_code'],
        low_memory=False,
        encoding='utf-8'
    )
    landmark_country_codes = []
    if include_us:
        landmark_country_codes.append('US')
    if include_eu:
        eu_codes = ['AT', 'BE', 'BG', 'CY', 'CZ', 'DE', 'DK', 'EE', 'ES', 'FI', 'FR', 'GR', 'HR', 'HU', 'IE', 'IT',
                    'LT', 'LU', 'LV', 'MT', 'NL', 'PL', 'PT', 'RO', 'SE', 'SI', 'SK']
        landmark_country_codes.extend(eu_codes)
    data = df[df['landmark_country_code'].isin(landmark_country_codes)]
    data = data[data['landmark_feature_code'].isin(['PPLA', 'PPLA2', 'PPLA3', 'PPLA4'])]
    data = data[data['landmark_population'] > 5000]
    logging.info("Data loading completed.")
    return data

def sample_data(df, sample_size=SAMPLE_SIZE, seed=SEED):
    logging.info(f"Sampling {sample_size} data points...")
    if sample_size > len(df):
        sample_size = len(df)
        logging.warning(f"Sample size exceeds available data. Adjusted sample_size={sample_size}")
    return df.sample(sample_size, random_state=seed)

def get_nearby_city(landmark_id, latitude, longitude, population):
    try:
        url = (f"http://api.geonames.org/findNearbyPlaceNameJSON?"
               f"lat={latitude}&lng={longitude}&cities=cities15000&"
               f"radius={min((population / 100), 300)}&maxRows=10&username={GEONAMES_USERNAME}")
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if data['geonames']:
                return data['geonames'][1] if len(data['geonames']) > 1 else None
    except Exception as exc:
        logging.warning(f"GeoNames API unavailable for {landmark_id}: {exc}")
    return None

def prepare_geospatial_data(sample, rng):
    logging.info("Preparing geospatial data...")
    preps = list(SPATIAL_RELATION_WITH_AZIMUTH.keys())
    sample["spatial_relation"] = rng.choice(preps, size=len(sample))
    sample["azimuth"] = sample["spatial_relation"].map(SPATIAL_RELATION_WITH_AZIMUTH)
    sample["azimuth"] = sample.apply(
        lambda x: rng.integers(0, 361) if x["spatial_relation"] == "near" else x["azimuth"],
        axis=1
    )
    sample["between_landmark_2"] = sample.apply(
        lambda x: get_nearby_city(x["landmark_id"], x["landmark_latitude"], x["landmark_longitude"], x["landmark_population"])
        if x["spatial_relation"] == "between" else None,
        axis=1
    )
    sample["distance_to_landmark"] = sample.apply(
        lambda x: min(200_000, rng.lognormal(mean=np.log(x["landmark_population"] / 3), sigma=1))
        if x["spatial_relation"] != "between" else None,
        axis=1
    )
    before = len(sample)
    sample = sample[~((sample["spatial_relation"] == "between") & (sample["between_landmark_2"].isnull()))].copy()
    after = len(sample)
    if after < before:
        logging.info(f"Removed {before - after} rows with no nearby city for 'between'")
    logging.info("Geospatial data preparation completed.")
    return sample

def generate_locative_expression(row):
    spatial_relation = row['spatial_relation']
    name = row['landmark_name']
    if spatial_relation == "between":
        if row['between_landmark_2'] is None:
            return None
        else:
            return f"{spatial_relation} {name} and {row['between_landmark_2']['name']}"
    elif spatial_relation == "near":
        return f"{spatial_relation} {name}"
    else:
        return f"{round(row['distance_to_landmark'] / 1000, 1)} km {spatial_relation} of {name}"

def prepare_target_location_info(sample, rng):
    logging.info("Preparing target location information...")
    sample['locative_expression'] = sample.apply(generate_locative_expression, axis=1)
    def _get_between_trajector_center(row):
        if row['between_landmark_2'] is not None:
            return geopy.point.Point(
                latitude=(float(row['between_landmark_2']['lat']) + float(row['landmark_latitude'])) / 2,
                longitude=(float(row['between_landmark_2']['lng']) + float(row['landmark_longitude'])) / 2
            )
        return None
    sample["trajector_center"] = sample.apply(
        lambda x: geopy.distance.distance(kilometers=x["distance_to_landmark"]/1000).destination(
            (x["landmark_latitude"], x["landmark_longitude"]),
            x["azimuth"]
        ) if x["spatial_relation"] != "between" else _get_between_trajector_center(x),
        axis=1
    )

    land_poly = gpd.read_file(f"{repo_root}/data/gandr_prelims/ne_10m_land") # Change for your path
    def _is_on_land(longitude, latitude, land_poly):
        location_point = gpd.GeoSeries([Point(longitude, latitude)], crs="EPSG:4326")
        joined = gpd.sjoin(location_point.to_frame("geometry"), land_poly, predicate="within")
        return not joined.empty
    sample["on_water"] = sample.apply(
        lambda x: not _is_on_land(x["trajector_center"].longitude, x["trajector_center"].latitude, land_poly),
        axis=1
    )

    def _sample_based_on_population(population):
        center_5, width_5 = 300_000, 100_000
        center_6, width_6 = 60_000, 100_000
        center_7, width_7 = 5_000, 100_000
        p5 = np.exp(-((population - center_5) / width_5) ** 2)
        p6 = np.exp(-((population - center_6) / width_6) ** 2)
        p7 = np.exp(-((population - center_7) / width_7) ** 2)
        total = p5 + p6 + p7
        if total == 0 or np.isnan(total):
            probs = [1 / 3, 1 / 3, 1 / 3]
        else:
            probs = [p5 / total, p6 / total, p7 / total]
        return np.random.choice(H3_RESOLUTIONS, p=probs)
    sample["trajector_h3_level"] = sample.apply(
        lambda x: _sample_based_on_population(x["landmark_population"]),
        axis=1
    )
    sample["trajector_cell_index"] = sample.apply(
        lambda x: h3.latlng_to_cell(
            lat=x["trajector_center"].latitude,
            lng=x["trajector_center"].longitude,
            res=x["trajector_h3_level"]
        ),
        axis=1
    )
    sample["trajector_area"] = sample.apply(
        lambda x: h3.cell_area(x["trajector_cell_index"], unit="m^2"),
        axis=1
    )
    logging.info("Target location information preparation completed.")
    return sample

# --- THREAD-SAFE API RATE-LIMITER ---
api_lock = threading.Lock()
request_times = []

def rate_limited_call(func):
    def wrapper(*args, **kwargs):
        while True:
            with api_lock:
                now = time.time()
                global request_times
                request_times[:] = [t for t in request_times if now - t < 60]
                enough_time_passed = len(request_times) < 2 or (now - request_times[-2]) >= 0.5
                if len(request_times) < 60 and enough_time_passed:
                    request_times.append(now)
                    break
            time.sleep(0.05)
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"API call failed: {e}")
            return None
    return wrapper

# --- MAP IMAGE GENERATION (CPU-bound, parallelized across processes) ---
def make_area_map_base64(idx, lat, lon, map_size_meters, image_size_pixels=1024):
    try:
        meters_per_degree_lat = 111320.0
        lat_radians = math.radians(lat)
        meters_per_degree_lon = 111320.0 * math.cos(lat_radians)
        d_lat = (map_size_meters / meters_per_degree_lat) / 2.0
        d_lon = (map_size_meters / meters_per_degree_lon) / 2.0
        lat_min, lat_max = lat - d_lat, lat + d_lat
        lon_min, lon_max = lon - d_lon, lon + d_lon
        wgs84 = CRS.from_epsg(4326)
        web_merc = CRS.from_epsg(3857)
        transformer_to_web = Transformer.from_crs(wgs84, web_merc, always_xy=True)
        x_min, y_min = transformer_to_web.transform(lon_min, lat_min)
        x_max, y_max = transformer_to_web.transform(lon_max, lat_max)
        fig, ax = plt.subplots(figsize=(image_size_pixels / 100, image_size_pixels / 100), dpi=100)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.axis('off')
        cx.add_basemap(ax, crs=web_merc.to_string(), source=TILE_PROVIDER)
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        time.sleep(1.5)
        return idx, image_base64
    except Exception as ex:
        logging.error(f"Image generation failed for {idx}: {ex}")
        return idx, None

# --- API CALLS (rate-limited, thread-parallelized) ---

@rate_limited_call
def generate_area_description_from_mapimage(idx, landmark_id, image_base64, natural_disaster):
    if not image_base64:
        logging.warning(f"No map image for landmark ID {landmark_id}")
        return idx, None
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Please create one concise description of this area which can be directly used in a news article about a {natural_disaster} in this area. Please only describe the area based on the provided map, not the natural disaster. Start with 'The area is...'"
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                    }
                ]
            }
        ]
    )
    if not response.choices or response.choices[0].message.content == "":
        logging.warning(f"No description generated for landmark ID {landmark_id}.")
        return idx, None
    return idx, response.choices[0].message.content

@rate_limited_call
def generate_disaster_news_article(idx, landmark_id, description, disaster, locative_expression):
    if not description:
        logging.warning(f"Empty description for landmark ID {landmark_id}")
        return idx, None
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Please provide an AP-style news article about a {disaster} which happened "
                                f"{locative_expression}. The article should be about 300 words long and "
                                f"explicitly include the location description '{locative_expression}'. "
                                f"The area where the {disaster} occurred is described as: {description}",
                    }
                ]
            }
        ]
    )
    if not response.choices or response.choices[0].message.content == "":
        logging.warning(f"No news article generated for landmark ID {landmark_id}.")
        return idx, None
    return idx, response.choices[0].message.content

def save_sample_to_json(sample, filename):
    sample.to_json(filename, orient='records', index=False)

def main(include_us=True, include_eu=False):
    timestamp = pd.Timestamp.now().strftime("%Y%m%d")
    final_dataset_name = f"gandr_{timestamp}"
    if include_us:
        final_dataset_name += "_us"
    if include_eu:
        final_dataset_name += "_eu"
    configure_logging(f"{final_dataset_name}_log.log")
    logging.info("Starting the dataset generation...")
    df = load_data(include_us, include_eu)
    sample = sample_data(df)
    rng = np.random.default_rng(SEED)
    sample = prepare_geospatial_data(sample, rng)
    sample = prepare_target_location_info(sample, rng)
    sample = sample[sample['trajector_center'].notnull()].copy()
    sample["natural_disaster"] = rng.choice(NATURAL_DISASTERS, size=len(sample))

    # --- First: make all maps in parallel with ProcessPoolExecutor ---
    logging.info("Generating map images in parallel...")
    maps_args = []
    for idx, row in sample.iterrows():
        lat = row["trajector_center"].latitude
        lon = row["trajector_center"].longitude
        map_size_meters = 2 * np.sqrt(row["trajector_area"])
        maps_args.append((idx, lat, lon, map_size_meters))
    map_results = {}
    with ProcessPoolExecutor(max_workers=1) as pexec:
        futures = [pexec.submit(make_area_map_base64, *args) for args in maps_args]
        for fut in tqdm(as_completed(futures), desc="Map Images", total=len(futures)):
            idx, img64 = fut.result()
            map_results[idx] = img64

    # --- Next: for each map, generate the area description (rate-limited, thread-pool-ed) ---
    logging.info("Generating all target area descriptions (rate-limited in parallel)...")
    area_desc_results = {}
    with ThreadPoolExecutor(max_workers=20) as texec:
        futures = {}
        for idx, row in sample.iterrows():
            img64 = map_results.get(idx)
            futures[texec.submit(generate_area_description_from_mapimage, idx, row["landmark_id"], img64, row["natural_disaster"])] = idx
        for fut in tqdm(as_completed(futures), desc="Area Descriptions", total=len(futures)):
            idx, desc = fut.result()
            area_desc_results[idx] = desc
            save_sample_to_json(sample, "sample_after_map_generation.json")
    sample["trajector_area_description"] = sample.index.map(area_desc_results.get)
    logging.info("Completed generating target area descriptions.")

    # --- Now, for each description, generate the news article (rate-limited, thread-pool-ed) ---
    logging.info("Generating all disaster news articles (rate-limited in parallel)...")
    results_article = {}
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {}
        for idx, row in sample.iterrows():
            futures[executor.submit(
                generate_disaster_news_article, idx, row["landmark_id"],
                row["trajector_area_description"], row["natural_disaster"], row["locative_expression"]
            )] = idx
        for fut in tqdm(as_completed(futures), desc="News Articles", total=len(futures)):
            idx, val = fut.result()
            results_article[idx] = val
            save_sample_to_json(sample, "sample_after_news_article_generation.json")
    sample["disaster_news_article"] = sample.index.map(results_article.get)
    logging.info("Completed generating disaster news articles.")

    logging.info("Final clean-ups...")
    sample["trajector_center"] = sample.apply(
        lambda x: x["trajector_center"].format_decimal(altitude=False) if x["trajector_center"] is not None else None,
        axis=1
    )
    logging.info(f"Saving dataset to {final_dataset_name}.json...")
    save_sample_to_json(sample, f"{final_dataset_name}.json")
    logging.info("Dataset generation completed.")
    return sample

tqdm.pandas()
if __name__ == "__main__":
    main(include_us=False, include_eu=True)