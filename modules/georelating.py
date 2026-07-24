import os
import time
import logging
import json

import stanza
import pandas as pd

from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from agent_components.environment.external_tools import safe_latlng_to_cell, safe_smallest_covering_cell
from agent_components.llms.api_error_handler import MultiWindowRateLimiter, default_llm_rate_limits
from models.candidates import GeoRelatingState
from models.errors import ExecutionStep
from modules.reflective_geocoding import ReflectiveGeoCoder

"""
Configuration
"""

MAX_WORKERS = 4
# Per-LLM-call throttle: GWDG ChatAI limits if CHATAI_BASE_URL points there,
# else a higher self-hosted-vLLM ceiling (see api_error_handler.default_llm_rate_limits).
LLM_RATE_LIMITS = default_llm_rate_limits()
# The unified three-stage graph exceeds langgraph's default recursion limit of 25
# when reflection loops fire in several stages.
GRAPH_RECURSION_LIMIT = 50
# If True, the full graph state (incl. all prompts and raw LLM outputs) is written
# to the JSONL file per row; otherwise only a compact result record is saved.
SAVE_FULL_STATE = False
# If True, actor LLM calls request the API's structured output mode
# (response_format json_object), guaranteeing syntactically valid JSON. Off by
# default so the effect can be evaluated cleanly against the baseline.
USE_STRUCTURED_OUTPUT = False
# v2: cache entries depend on the toponym extraction heuristic; bump the file name
# whenever extract_toponyms_from_doc changes so stale entries are not reused.
TOPONYM_CACHE_FILE = "toponym_cache_v2.json"


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
    # every chat completion emits an INFO line otherwise
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

"""
Helpers
"""

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ExecutionStep):
            return obj.value
        return super().default(obj)


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


"""
Toponym Recognition
"""


def extract_toponyms_from_doc(doc):
    """Extract toponyms from a processed stanza Document, merging comma-separated
    compound toponyms (e.g. 'Yaiza, Lanzarote')."""
    article_text = doc.text
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
                # Do not merge enumerations like "Sudley, Yorkshire, Matthews and
                # Battery Heights": if yet another location entity follows the pair,
                # joined by a comma or "and", the comma is a list separator rather
                # than part of a compound toponym like "NEWARK, Ohio".
                is_enumeration = False
                if i + 2 < len(entities):
                    after_next = entities[i + 2]
                    connector = article_text[next_entity.end_char:after_next.start_char].strip().lower()
                    if after_next.type in ["LOC", "GPE"] and connector in {",", ", and", "and"}:
                        is_enumeration = True

                if not is_enumeration:
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


def recognize_toponyms_bulk(texts, nlp):
    """Run stanza NER over a list of article texts in one batched call."""
    if hasattr(nlp, "bulk_process"):
        docs = nlp.bulk_process(list(texts))
    else:
        docs = [nlp(text) for text in texts]
    return [extract_toponyms_from_doc(doc) for doc in docs]


def compute_toponyms_with_cache(df, cache_path):
    """
    Return the toponym list per row of ``df`` (keyed by landmark_id), running the
    stanza NER pipeline only for articles that are not in the cache yet. The stanza
    pipeline is only initialized when at least one article is missing from the cache.
    """
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)

    ids = df['landmark_id'].astype(str)
    missing_mask = ~ids.isin(cache.keys())
    if missing_mask.any():
        logging.info(f"Running NER for {int(missing_mask.sum())} articles "
                     f"({int((~missing_mask).sum())} cached).")
        nlp = stanza.Pipeline(lang='en', processors='tokenize,ner',
                              download_method=stanza.DownloadMethod.REUSE_RESOURCES)
        texts = df.loc[missing_mask, 'disaster_news_article_text'].tolist()
        toponym_lists = recognize_toponyms_bulk(texts, nlp)
        for landmark_id, toponyms in zip(ids[missing_mask], toponym_lists):
            cache[landmark_id] = toponyms
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
    else:
        logging.info(f"All {len(df)} articles found in the toponym cache, skipping NER.")

    return ids.map(cache)


"""
GeoRelating
"""


def build_result_row(idx, row, state: GeoRelatingState) -> dict:
    """Compact per-row result record for the JSONL file."""
    if SAVE_FULL_STATE:
        return {
            "index": idx,
            "landmark_id": row['landmark_id'],
            "state": state.model_dump(),
            "georelated": state.georelated if state.georelated else None
        }

    errors = [error.model_dump() for error in (state.fatal_errors +
                                               state.resolution_fatal_errors +
                                               state.georelating_fatal_errors +
                                               state.georelating_invalid_output_errors)]
    return {
        "index": idx,
        "landmark_id": row['landmark_id'],
        "toponyms": state.toponyms,
        "geocoded_toponyms": [
            {
                "toponym": topo.toponym,
                "selected_candidate_geonameId": topo.selected_candidate_geonameId,
                "coordinates": topo.coordinates
            }
            for topo in state.valid_geocoded_toponyms
        ],
        "invalid_toponyms": [
            {"toponym": topo.toponym,
             "error": topo.errors_per_toponym[-1].error_message if topo.errors_per_toponym else None}
            for topo in state.invalid_toponyms
        ] or None,
        "unresolved_toponyms": [
            {"toponym": topo.toponym,
             "error": topo.errors[-1].error_message if topo.errors else None}
            for topo in state.invalid_geocoded_toponyms
        ] or None,
        "georelated": state.georelated if state.georelated else None,
        "errors": errors if errors else None
    }


def process_row_save_as_jsonl(row, agent_graph, output_path, processed_indices_set, lock):
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

            agent_graph_answer = agent_graph.invoke(
                input_state, config={"recursion_limit": GRAPH_RECURSION_LIMIT})
            state = GeoRelatingState(**agent_graph_answer)

            result = build_result_row(idx, row, state)

            # Save immediately, thread-safe
            with lock:
                with open(output_path, "a", encoding="utf-8") as fout:
                    fout.write(json.dumps(result, cls=CustomJSONEncoder) + "\n")

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
        "georelated": None,
        "error": str(last_exception)
    }

    # Save error result too
    with lock:
        with open(output_path, "a", encoding="utf-8") as fout:
            fout.write(json.dumps(result, cls=CustomJSONEncoder) + "\n")

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

    agent_graph = geocoder.build_full_graph().compile()

    lock = Lock()  # for file write safety

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(
                process_row_save_as_jsonl,
                row,
                agent_graph,
                output_path,
                processed_indices,
                lock
            )
            for idx, row in df.iterrows()
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Georelating articles"):
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
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    # Model names are env-overridable so this batch script can target the same
    # self-hosted vLLM endpoint as the demo (set GEORELATING_ACTOR_MODEL /
    # GEORELATING_CRITIC_MODEL to the vLLM --served-model-name, e.g. "georelating-llm",
    # and CHATAI_BASE_URL to the vLLM URL). The defaults are ChatAI models for the
    # standalone research run. (llama-3.3-70b / mistral-large are no longer served.)
    actor = os.getenv("GEORELATING_ACTOR_MODEL", "mistral-medium-3.5-128b")
    critic = os.getenv("GEORELATING_CRITIC_MODEL", "gemma-4-31b-it")
    dataset = "New"

    # Get the project root directory (parent of modules/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    data_dir = os.path.join(project_root, "data")
    output_dir = os.path.join(project_root, "output/georelating")
    output_file = f"processed_{timestamp}_{data_file}l"

    data_path = os.path.join(data_dir, data_file)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file {data_path} does not exist.")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)
    configure_logging(output_path.replace('.jsonl', "_log.log"))
    logging.info(f"Starting georelating for {data_file}")

    rate_limiter = MultiWindowRateLimiter(LLM_RATE_LIMITS)
    geocoder = ReflectiveGeoCoder(
        actor_model_name=actor,
        critic_model_name=critic,
        call_times=[],
        skip_few_shot_loader=False,
        data_set=dataset,
        rate_limiter=rate_limiter,
        use_structured_output=USE_STRUCTURED_OUTPUT
    )

    df = pd.read_json(data_path, orient='records')

    df[['disaster_news_article_title', 'disaster_news_article_text']] = df.apply(extract_title_text_from_row, axis=1)

    # process only 3 rows for testing
    df = df.head(10)

    # NER only for the selected rows; cached results are reused across runs
    df['toponyms'] = compute_toponyms_with_cache(df, os.path.join(output_dir, TOPONYM_CACHE_FILE))

    parallel_process_dataframe_jsonl(df, geocoder, output_path)

    processed_df = load_processed_jsonl(output_path)

    merged_df = pd.merge(
        df,
        processed_df[['landmark_id', 'georelated']],
        on='landmark_id',
        how='left'
    )
    merged_df['pred_cell'] = merged_df.apply(safe_latlng_to_cell, axis=1)
    merged_df['pred_cell_covering'] = merged_df.apply(safe_smallest_covering_cell, axis=1)

    output_merged_path = output_path.replace('.jsonl', '.json')
    merged_df.to_json(output_merged_path, orient="records", force_ascii=False, indent=4)
    logging.info(f"Enriched output written to: {output_merged_path}")
