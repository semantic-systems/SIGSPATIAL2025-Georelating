"""
Georelating demo service: a thin, reusable wrapper around the reflective
georelating pipeline for API/UI consumption. It does not modify the pipeline —
it composes the public building blocks (ReflectiveGeoCoder, stanza NER, H3
helpers) and adds a display-only spatial-relation extraction step.
"""
import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import h3
import pandas as pd
import stanza
from dotenv import load_dotenv

from agent_components.environment.external_tools import (get_h3_resolution_for_area,
                                                         smallest_covering_cell)
from agent_components.environment.internal_tools import OutputParser
from agent_components.llms.api_error_handler import MultiWindowRateLimiter, default_llm_rate_limits
from models.candidates import GeoRelatingState
from modules.georelating import extract_title_text_from_row, extract_toponyms_from_doc
from modules.reflective_geocoding import ReflectiveGeoCoder

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GANDR_PATH = os.path.join(PROJECT_ROOT, "data", "gandr.json")

# Default to the vLLM served-model-name from the deployment; override via env.
DEFAULT_ACTOR = os.getenv("DEMO_ACTOR_MODEL", "georelating-llm")
DEFAULT_CRITIC = os.getenv("DEMO_CRITIC_MODEL", "georelating-llm")
GRAPH_RECURSION_LIMIT = 50
MAX_TEXT_LENGTH = 8000

# Ordered pipeline stages surfaced to the UI, and the graph nodes that belong to
# each. Used to translate langgraph's per-node stream into a coarse progress signal.
STAGE_ORDER = ["toponyms", "generation", "resolution", "georelating"]
STAGE_LABELS = {
    "toponyms": "Toponym recognition",
    "generation": "Candidate generation",
    "resolution": "Candidate resolution",
    "georelating": "Geospatial reasoning",
}
STAGE_BY_NODE = {
    "create_prompt": "generation", "call_actor": "generation", "extract_output": "generation",
    "validate_output": "generation", "retrieve_candidates": "generation", "criticize": "generation",
    "add_generation_critique_error": "generation",
    "create_candidate_resolution_prompt": "resolution", "call_resolution_actor": "resolution",
    "extract_resolution_output": "resolution", "validate_resolution_output": "resolution",
    "criticize_resolution": "resolution", "add_resolution_critique_error": "resolution",
    "create_georelating_prompt": "georelating", "call_georelating_actor": "georelating",
    "extract_georelating_output": "georelating", "validate_georelating_output": "georelating",
    "criticize_georelating": "georelating", "add_georelating_critique_error": "georelating",
}

logger = logging.getLogger(__name__)


class GeorelatingService:
    """Singleton-style service holding the heavyweight pipeline components."""

    def __init__(self, actor_model: str = DEFAULT_ACTOR, critic_model: str = DEFAULT_CRITIC):
        load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
        self.actor_model = actor_model
        self.critic_model = critic_model
        self.rate_limiter = MultiWindowRateLimiter(default_llm_rate_limits())
        self._lock = threading.Lock()
        self._graphs = {}      # use_structured_output -> (geocoder, compiled graph)
        self._nlp = None
        self._articles_df = None

    ####################################################################################################################
    # Lazy heavyweight components
    ####################################################################################################################

    def _get_graph(self, use_structured_output: bool):
        with self._lock:
            if use_structured_output not in self._graphs:
                logger.info("Initializing pipeline (structured_output=%s)...", use_structured_output)
                geocoder = ReflectiveGeoCoder(
                    actor_model_name=self.actor_model,
                    critic_model_name=self.critic_model,
                    call_times=[],
                    skip_few_shot_loader=False,
                    data_set="New",
                    rate_limiter=self.rate_limiter,
                    use_structured_output=use_structured_output,
                )
                graph = geocoder.build_full_graph().compile()
                self._graphs[use_structured_output] = (geocoder, graph)
            return self._graphs[use_structured_output]

    def _get_nlp(self):
        with self._lock:
            if self._nlp is None:
                logger.info("Initializing stanza NER pipeline...")
                self._nlp = stanza.Pipeline(lang='en', processors='tokenize,ner',
                                            download_method=stanza.DownloadMethod.REUSE_RESOURCES,
                                            verbose=False)
            return self._nlp

    def _get_articles(self) -> pd.DataFrame:
        if self._articles_df is None:
            df = pd.read_json(GANDR_PATH, orient="records")
            df[['title', 'text']] = df.apply(extract_title_text_from_row, axis=1)
            self._articles_df = df
        return self._articles_df

    def warm_up(self):
        """Pre-initialize the default pipeline so the first request is fast."""
        self._get_graph(False)
        self._get_nlp()
        self._get_articles()

    ####################################################################################################################
    # Articles
    ####################################################################################################################

    def list_articles(self):
        df = self._get_articles()
        return [
            {"landmark_id": int(row.landmark_id),
             "title": str(row.title).lstrip("# ").strip(),
             "natural_disaster": getattr(row, "natural_disaster", None)}
            for row in df.itertuples()
        ]

    def get_article(self, landmark_id: int):
        df = self._get_articles()
        match = df[df['landmark_id'] == landmark_id]
        if match.empty:
            return None
        row = match.iloc[0]
        return {"landmark_id": int(row['landmark_id']),
                "title": str(row['title']),
                "text": str(row['text'])}

    ####################################################################################################################
    # Georelating
    ####################################################################################################################

    def georelate(self, text: str = None, article_id: int = None,
                  use_structured_output: bool = False, geonames_username: str = None,
                  progress_callback=None) -> dict:
        """Run the full pipeline for a GANDR article or user-provided text and
        return a display-ready result document. ``geonames_username`` (optional)
        overrides the server's GeoNames account for this request. ``progress_callback``
        (optional) receives a dict as the pipeline advances through its stages."""
        def _emit(stage, node=None, reflection=False):
            if progress_callback:
                progress_callback({
                    "current": stage,
                    "current_index": STAGE_ORDER.index(stage),
                    "reflection_active": reflection,
                    "detail": node,
                })

        if article_id is not None:
            article = self.get_article(int(article_id))
            if article is None:
                raise ValueError(f"Unknown article id {article_id}")
            title, body = article["title"], article["text"]
        elif text:
            text = text.strip()[:MAX_TEXT_LENGTH]
            lines = text.split("\n")
            title = lines[0].strip() if len(lines) > 1 else "User-provided report"
            body = "\n".join(lines[1:]).strip() if len(lines) > 1 else text
        else:
            raise ValueError("Either article_id or text must be provided.")

        _emit("toponyms")
        nlp = self._get_nlp()
        toponyms = extract_toponyms_from_doc(nlp(body))

        geocoder, graph = self._get_graph(use_structured_output)
        input_state = {
            "article_id": str(article_id) if article_id is not None else "user_input",
            "article_title": title,
            "article_text": body,
            "toponyms": toponyms,
            "geonames_username": (geonames_username or "").strip() or None,
        }

        # The display-only spatial-relation extraction needs only the report text
        # and the recognized toponyms, both available now, so run it concurrently
        # with the multi-minute pipeline instead of adding a call afterwards.
        with ThreadPoolExecutor(max_workers=1) as pool:
            relations_future = pool.submit(self._extract_spatial_relations, geocoder, body, toponyms)
            # Stream the graph node-by-node so the UI can show real stage progress;
            # the final "values" chunk is the complete end state.
            final_state = None
            for mode, chunk in graph.stream(input_state,
                                            config={"recursion_limit": GRAPH_RECURSION_LIMIT},
                                            stream_mode=["updates", "values"]):
                if mode == "updates":
                    node = next(iter(chunk), None)
                    stage = STAGE_BY_NODE.get(node)
                    if stage:
                        _emit(stage, node, reflection=bool(node) and node.startswith("criticize"))
                else:
                    final_state = chunk
            state = GeoRelatingState(**final_state)
            spatial_relations = relations_future.result()

        result = {
            "article": {"title": title, "text": body},
            "options": {"actor_model": self.actor_model, "critic_model": self.critic_model,
                        "use_structured_output": use_structured_output},
            "toponyms": state.toponyms,
            "geocoded_toponyms": self._geocoded_toponym_details(state),
            "spatial_relations": spatial_relations,
            "georelated": None,
            "cells": {},
            "errors": self._collect_errors(state),
        }

        if state.georelated:
            center = state.georelated.get("center coordinates of affected area") or {}
            area = state.georelated.get("affected area in square km")
            if isinstance(center, dict) and area is not None:
                lat, lng = float(center["latitude"]), float(center["longitude"])
                result["georelated"] = {"center": {"latitude": lat, "longitude": lng},
                                        "area_km2": float(area)}
                pred_cell = h3.latlng_to_cell(lat=lat, lng=lng,
                                              res=get_h3_resolution_for_area(float(area) * 1e6))
                covering = smallest_covering_cell(lat, lng, float(area))
                result["cells"] = {
                    "pred_cell": self._cell_details(pred_cell),
                    "pred_cell_covering": self._cell_details(covering),
                }
        return result

    @staticmethod
    def _cell_details(cell: str) -> dict:
        return {
            "index": cell,
            "resolution": h3.get_resolution(cell),
            "area_km2": h3.cell_area(cell, unit="km^2"),
            "boundary": [[lat, lng] for lat, lng in h3.cell_to_boundary(cell)],
        }

    @staticmethod
    def _geocoded_toponym_details(state: GeoRelatingState) -> list:
        candidates_per_toponym = {
            topo.toponym_with_search_arguments.toponym.casefold(): {
                c["geonameId"]: c for c in topo.candidates if "geonameId" in c}
            for topo in state.toponyms_with_candidates
        }
        details = []
        for resolved in state.valid_geocoded_toponyms:
            candidate = (candidates_per_toponym.get(resolved.toponym.casefold()) or {}).get(
                resolved.selected_candidate_geonameId)
            if not candidate:
                continue
            details.append({
                "toponym": resolved.toponym,
                "geonameId": resolved.selected_candidate_geonameId,
                "name": candidate.get("name"),
                "country": candidate.get("countryName"),
                "feature": candidate.get("fcodeName"),
                "latitude": float(candidate["lat"]),
                "longitude": float(candidate["lng"]),
                "reasoning": resolved.reasoning,
            })
        return details

    @staticmethod
    def _collect_errors(state: GeoRelatingState) -> list:
        errors = []
        for error in (state.fatal_errors + state.resolution_fatal_errors +
                      state.georelating_fatal_errors + state.georelating_invalid_output_errors):
            step = error.execution_step
            errors.append({"step": step.value if hasattr(step, "value") else str(step),
                           "message": error.error_message})
        for topo in state.invalid_toponyms:
            if topo.errors_per_toponym:
                errors.append({"step": "candidate_generation",
                               "message": f"'{topo.toponym}': {topo.errors_per_toponym[-1].error_message}"})
        for topo in state.invalid_geocoded_toponyms:
            if topo.errors:
                errors.append({"step": "candidate_resolution",
                               "message": f"'{topo.toponym}': {topo.errors[-1].error_message}"})
        return errors

    def _extract_spatial_relations(self, geocoder: ReflectiveGeoCoder, text: str, toponyms: list) -> list:
        """Display-only extraction of the complex locative expressions considered.
        Runs outside the scientific pipeline; failures are non-fatal."""
        prompt = (
            "Extract every complex locative expression from the disaster report below. "
            "Respond strictly with a JSON array; each element must have the keys: "
            '"locative_expression" (the verbatim phrase), "landmark" (the reference toponym), '
            '"spatial_relation" (e.g. north of, between, near, in), and '
            '"distance_km" (number or null).\n'
            f"Known toponyms: {toponyms}\n"
            f"Report:\n{text}\n"
            "JSON array only:"
        )
        try:
            llm = geocoder.llm.bind(response_format={"type": "json_object"}) \
                if geocoder.use_structured_output else geocoder.llm
            answer = geocoder._invoke_llm(llm, prompt)
            relations = OutputParser.clean_and_parse_json_content(answer.content, '[', ']')
            if isinstance(relations, dict):
                relations = [relations]
            return [r for r in relations if isinstance(r, dict)]
        except Exception as e:
            logger.warning("Spatial relation extraction failed: %s", e)
            return []
