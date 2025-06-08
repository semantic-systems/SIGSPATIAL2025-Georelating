import os
import time
import logging
import threading
import json

import h3
import stanza
import pandas as pd

from threading import Lock
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

from agent_components.llms.chatAI import ChatAIHandler
from agent_components.environment.internal_tools import OutputParser
from helpers.helpers import preprocess_data
from models.candidates import CandidateGenerationState, GeoCodingState
from modules.reflective_geocoding import ReflectiveGeoCoder

"""
Threading and Rate Limiting
"""

class MultiWindowRateLimiter:
    def __init__(self, limits):
        """
        limits: list of tuples (max_calls, period_sec), e.g.
            [(2, 1), (60, 60), (3000, 3600)]
        """
        self.limits = limits
        self.lock = threading.Lock()
        self.timestamps = []  # all call times for sliding window checks

    def acquire(self):
        while True:
            with self.lock:
                now = time.time()
                # Remove timestamps older than the max period
                max_period = max(period for _, period in self.limits)
                self.timestamps = [t for t in self.timestamps if now - t < max_period]

                waits = []
                for max_calls, period in self.limits:
                    window = [t for t in self.timestamps if now - t < period]
                    if len(window) >= max_calls:
                        # How many seconds until next slot becomes available?
                        wait = period - (now - window[0])
                        waits.append(wait)
                    else:
                        waits.append(0)
                max_wait = max(waits)
                if max_wait > 0:
                    # Must release lock before sleep to not block other threads
                    pass
                else:
                    # Allowed, record this call and return
                    self.timestamps.append(now)
                    return  # Ready to proceed
            # Sleep outside lock to prevent deadlocks
            time.sleep(max_wait)


"""
Logging Configuration
"""
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
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s')

"""
Helpers
"""
def extract_title_text_from_row(row):
    """
    Extracts the title and text from the 'disaster_news_article' column of a DataFrame row.

    Args:
        row (pd.Series): A row of the DataFrame containing the 'disaster_news_article' column.

    Returns:
        pd.Series: A Series with 'title' and 'text' columns. Returns a Series with empty strings if input is invalid.
    """

    if not isinstance(row, pd.Series) or 'disaster_news_article' not in row.index:
        print("Error: Input must be a Pandas Series with a 'disaster_news_article' index.")
        return pd.Series({'title': '', 'text': ''})  # Return a Series with empty strings to signal an error

    article_string = row['disaster_news_article']

    if not isinstance(article_string, str):
        return pd.Series({'title': '', 'text': ''})  #Handle cases where the value isn't a string

    lines = article_string.split('\n')
    title = lines[0].strip()  # First line is the title
    text = '\n'.join(
        [line.strip() for line in lines[1:] if line.strip()])  # Join remaining lines as text, removing empty lines

    return pd.Series({'title': title, 'text': text})


def extract_geocoded_toponyms(candidate_resolution: GeoCodingState):
    """
    Extracts geocoded toponyms from the candidate resolution state.
    Args:
        candidate_resolution (GeoCodingState): The candidate resolution state containing geocoded toponyms.
    Returns:
        List[dict]: A list of dictionaries representing the geocoded toponyms, each containing the toponym and its coordinates.
    """
    geocoded_toponyms = []
    for topo in candidate_resolution.valid_geocoded_toponyms:
        for item in candidate_resolution.toponyms_with_candidates:
            if item.toponym_with_search_arguments.toponym.casefold() == topo.toponym.casefold():
                # get the candidate with the correct geonameid
                for candidate in item.candidates:
                    if topo.selected_candidate_geonameId == candidate["geonameId"]:
                        topo.coordinates = {"latitude": candidate["lat"],
                                            "longitude": candidate["lng"]}
                        geocoded_toponyms.append(topo)
                        break
                break
    geocoded_toponyms = [toponym.model_dump() for toponym in geocoded_toponyms]
    return geocoded_toponyms


"""
Toponym Recognition
"""


def recognize_toponyms(article_text):
    doc = nlp(article_text)
    entities = doc.entities
    toponyms = []

    i = 0
    while i < len(entities):
        current = entities[i]
        current_text = current.text.strip()
        current_end = current.end_char

        # Check if the current entity is a valid toponym type
        if current.type not in ["LOC", "GPE"]:
            i += 1
            continue

        # Look ahead
        if i + 1 < len(entities):
            next_entity = entities[i + 1]
            between_text = article_text[current_end:next_entity.start_char].strip()

            # Case 1: Merge if comma-separated and no "and" (likely compound toponym)
            if between_text == "," and next_entity.type in ["GPE"]:
                merged_toponym = f"{current_text}, {next_entity.text.strip()}"
                # Check if the merged toponym is already in the list
                if merged_toponym not in toponyms:
                    toponyms.append(merged_toponym)
                i += 2
                continue

        if current_text not in toponyms:
            toponyms.append(current_text)

        i += 1

    return toponyms


"""
GeoRelating
"""


def georelate(model_name: str, long_term_memory, article_text: str, mentioned_toponyms: List[dict]):
    """
    Georelate the given article text with the mentioned toponyms using a language model.
    Args:
        model_name (str): The name of the language model to use.
        long_term_memory: The long term memory object containing the model.
        article_text (str): The text of the article to georelate.
        mentioned_toponyms (List[str]): A list of toponyms mentioned in the article.
    Returns:
        dict: The georelation result containing the geocoded toponyms and their coordinates.
    """
    prompt = long_term_memory.generate_geoelation_prompt(article_text=article_text,
                                                          mentioned_toponyms=mentioned_toponyms,
                                                          example_path=r"data/few_shot_example_georelating.json")
    handler = ChatAIHandler()
    model = handler.get_model(model_name)
    llm_answer = model.invoke(prompt)
    cleaned_output = OutputParser.clean_and_parse_json_content(content=llm_answer.content,
                                                               start_token='{',
                                                               end_token='}')
    return cleaned_output

# H3 cell areas by resolution (in m^2)
h3_areas = {res: h3.average_hexagon_area(res, unit="m^2") for res in range(16)}  # resolutions 0–15

# Function to get lowest H3 resolution that fits the area
def get_h3_resolution_for_area(area_m2):
    for res in sorted(h3_areas, reverse=True):
        if h3_areas[res] >= area_m2:
            return res
    return 5  # fallback to 5

def safe_latlng_to_cell(x):
    geor = x.get("georelated")
    if not geor:
        return None
    if isinstance(geor, dict):
        center = geor.get("center coordinates of affected area")
        area = geor.get("affected area in square km")
        if center is None:
            center = geor.get("center_coordinates_of_affected_area")
        if area is None:
            area = geor.get("affected_area_in_square_km")
    else:
        center = None
        area = None
    if not center:
        return None
    lat = center.get("latitude") if isinstance(center, dict) else None
    lng = center.get("longitude") if isinstance(center, dict) else None
    if lat is None or lng is None or area is None:
        return None
    try:
        return h3.latlng_to_cell(
            lat=float(lat),
            lng=float(lng),
            res=get_h3_resolution_for_area(float(area) * 1e6),
        )
    except Exception as e:
        return None

# Instantiate
rate_limiter = MultiWindowRateLimiter([
    (1, 1),        # 2 per second
    (50, 60),      # 60 per minute
    (2700, 3600),  # 3000 per hour
])

def safe_llm_call(fn, *args, **kwargs):
    thread_id = threading.get_ident()
    logging.info(f"Thread-{thread_id} waiting for rate limiter...")
    rate_limiter.acquire()
    logging.info(f"Thread-{thread_id} passed rate limit and calling {fn.__name__}...")
    result = fn(*args, **kwargs)
    logging.info(f"Thread-{thread_id} finished {fn.__name__}")
    return result


def process_row_save_as_jsonl(row, geocoder, generation_agent_graph, resolution_agent_graph, output_path, processed_indices_set, lock):
    idx = row.name
    if idx in processed_indices_set:
        logging.info(f"Article {idx} already processed, skipping.")
        return None  # Already processed

    attempt = 0
    max_attempts = 2
    last_exception = None
    while attempt < max_attempts:
        try:
            logging.info(f"Processing article {row['landmark_id']} (Attempt {attempt+1})")

            input_state = {
                "article_id": str(row['landmark_id']),
                "article_title": row['disaster_news_article_title'],
                "article_text": row['disaster_news_article_text'],
                "toponyms": row['toponyms']
            }

            generation_agent_graph_answer = safe_llm_call(generation_agent_graph.invoke, input_state)
            candidate_generation = CandidateGenerationState(**generation_agent_graph_answer)

            candidate_generation_dict = preprocess_data(candidate_generation.model_dump(), GeoCodingState)
            resolution_agent_graph_answer = safe_llm_call(resolution_agent_graph.invoke, candidate_generation_dict)
            candidate_resolution = GeoCodingState(**resolution_agent_graph_answer)
            geocoded_toponyms = extract_geocoded_toponyms(candidate_resolution)

            prompt = geocoder.working_memory.long_term_memory.generate_georelating_prompt(
                article_text=row['disaster_news_article_text'],
                mentioned_toponyms=geocoded_toponyms,
                example_path=r"data/few_shot_example_georelating.json"
            )
            llm_answer = safe_llm_call(geocoder.llm.invoke, prompt)
            cleaned_output = OutputParser.clean_and_parse_json_content(llm_answer.content, '{', '}')

            result = {
                "index": idx,
                "landmark_id": row['landmark_id'],
                "generated_candidates": candidate_generation.model_dump() if candidate_generation else None,
                "geocoded_toponyms": candidate_resolution.model_dump() if candidate_resolution else None,
                "georelated": cleaned_output
            }

            # Save immediately, thread-safe
            with lock:
                with open(output_path, "a", encoding="utf-8") as fout:
                    fout.write(json.dumps(result) + "\n")

            logging.info(f"Completed and saved article {row['landmark_id']}")
            return result

        except Exception as e:
            last_exception = e
            attempt += 1
            if attempt < max_attempts:
                logging.warning(f"Error processing article {row['landmark_id']}: {e}. Retrying...")
                time.sleep(1)

    # Only reached if all attempts failed
    logging.error(f"Failed to process article {row['landmark_id']} after {max_attempts} attempts.")
    result = {
        "index": idx,
        "landmark_id": row['landmark_id'],
        "generated_candidates": None,
        "geocoded_toponyms": None,
        "georelated": None,
        "error": str(last_exception)
    }

    # Save error result too
    with lock:
        with open(output_path, "a", encoding="utf-8") as fout:
            fout.write(json.dumps(result) + "\n")

    return result

def load_already_processed_indices(output_path):
    if not os.path.exists(output_path):
        return set()
    indices = set()
    with open(output_path, "r", encoding="utf-8") as fin:
        for line in fin:
            try:
                record = json.loads(line)
                # Defensive: might be integer or string, normalize as pandas does
                indices.add(record["index"])
            except Exception:
                continue
    return indices

def parallel_process_dataframe_jsonl(df, geocoder, output_path):
    """
    Parallel processing and save each row's result as JSONL.
    Skips already completed rows.
    """
    processed_indices = load_already_processed_indices(output_path)
    logging.info(f"Already processed {len(processed_indices)} rows, will skip them.")

    generation_graph_builder = geocoder.build_graph()
    generation_agent_graph = generation_graph_builder.compile()
    resolution_graph_builder = geocoder.build_resolution_graph()
    resolution_agent_graph = resolution_graph_builder.compile()

    lock = Lock()  # for file write safety

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        for idx, row in df.iterrows():
            # Each thread gets the lock, processed_indices (could be updated, but OK for append-only)
            futures.append(
                executor.submit(
                    process_row_save_as_jsonl,
                    row, geocoder,
                    generation_agent_graph,
                    resolution_agent_graph,
                    output_path,
                    processed_indices,
                    lock
                )
            )
        for future in as_completed(futures):
            _ = future.result()  # We do not need results in-process, data is on disk

    logging.info("Processing finished.")

def load_processed_jsonl(jsonl_path):
    results = []
    with open(jsonl_path, "r", encoding="utf-8") as fin:
        for line in fin:
            try:
                record = json.loads(line)
                results.append(record)
            except Exception:
                continue
    # Turn into DataFrame by index for later merge
    df_results = pd.DataFrame(results)
    return df_results



if __name__ == "__main__":
    data_file = "gandr.json"
    timestamp = pd.Timestamp.now().strftime("%Y%m%d")

    nlp = stanza.Pipeline(lang='en', processors='tokenize,ner')

    actor = "llama-3.3-70b-instruct"
    critic = "mistral-large-instruct"
    dataset = "New"
    data_dir = "data"
    output_dir = "output/georelating"
    output_file = f"processed_{timestamp}_{data_file}l"

    data_path = os.path.join(data_dir, data_file)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file {data_path} does not exist.")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)
    configure_logging(output_path.replace('.jsonl', "_log.log"))
    logging.info(f"Starting georelating for {data_file}")

    geocoder = ReflectiveGeoCoder(
        actor_model_name=actor,
        critic_model_name=critic,
        call_times=[],
        skip_few_shot_loader=False,
        data_set=dataset
    )

    df = pd.read_json(data_path, orient='records')

    df[['disaster_news_article_title', 'disaster_news_article_text']] = df.apply(extract_title_text_from_row, axis=1)
    df['toponyms'] = df['disaster_news_article_text'].apply(recognize_toponyms)

    parallel_process_dataframe_jsonl(df, geocoder, output_path)

    processed_df = load_processed_jsonl(output_path)

    merged_df = pd.merge(
        df,
        processed_df[['landmark_id', 'georelated']],
        on='landmark_id',
        how='left'
    )
    merged_df['pred_cell'] = df.apply(safe_latlng_to_cell, axis=1)

    output_merged_path = output_path.replace('.jsonl', '.json')
    merged_df.to_json(output_merged_path, orient="records", force_ascii=False, indent=4)
    logging.info(f"Enriched output written to: {output_merged_path}")