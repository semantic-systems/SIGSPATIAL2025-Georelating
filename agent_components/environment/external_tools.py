import os
import urllib.parse

import h3
import requests

from models.candidates import ToponymWithCandidates, CandidateGenerationOutput, ReflectionPhase
from models.errors import Error, ExecutionStep
from models.llm_output import ValidatedOutput


class GeoNamesAPI:
    def __init__(self, article_id: str = None):
        self.base_url = "http://api.geonames.org/search?"

    def search(self, params):
        params.update({'username': os.getenv('GEONAMES_USERNAME')})
        url = self.base_url + urllib.parse.urlencode(params)
        response = requests.get(url)
        if response.status_code != 200:
            response = requests.get(url) #retry once
            if response.status_code != 200:
                raise Exception(f"Error in GeoNamesAPI.search: {response.text}")
        json_response = response.json()
        if 'geonames' not in json_response:
            raise Exception(f"Error in GeoNamesAPI search: {json_response['status']['message']}")
        return response.json()

    def retrieve_candidates(self, validated_output: ValidatedOutput) -> CandidateGenerationOutput:
        candidate_generation_output = CandidateGenerationOutput(**validated_output.model_dump())
        try:
            topos_to_search = validated_output.valid_toponyms
            correct_duplicates = validated_output.duplicate_toponyms
            if hasattr(validated_output, 'reflection_phase'):
                if validated_output.reflection_phase == ReflectionPhase.ACTOR_RETRY_ON_INVALID_TOPONYMS:
                    topos_to_search = [topo for topo in validated_output.valid_toponyms if topo.generated_by_retry]
                    correct_duplicates = [topo for topo in validated_output.duplicate_toponyms if topo.generated_by_retry]
            for toponym_to_search_for in topos_to_search:
                response = self.search(toponym_to_search_for.params)
                toponym_with_candidates = ToponymWithCandidates(
                    toponym_with_search_arguments=toponym_to_search_for,
                    total_results = response['totalResultsCount'],
                    candidates=response['geonames'],
                    nof_retrieved_candidates=len(response['geonames'])
                )
                candidate_generation_output.toponyms_with_candidates.append(toponym_with_candidates)
            for duplicate_toponym in correct_duplicates:
                for toponym_with_candidates in candidate_generation_output.toponyms_with_candidates:
                    if toponym_with_candidates.toponym_with_search_arguments.toponym.casefold() == duplicate_toponym.duplicate_of.casefold():
                        candidate_generation_output.toponyms_with_candidates.append(
                            ToponymWithCandidates(
                                toponym_with_search_arguments=duplicate_toponym,
                                total_results=toponym_with_candidates.total_results,
                                candidates=toponym_with_candidates.candidates,
                                nof_retrieved_candidates=toponym_with_candidates.nof_retrieved_candidates
                            )
                        )
                        break
            return candidate_generation_output
        except Exception as e:
            candidate_generation_output.fatal_errors = [Error(execution_step=ExecutionStep.GEOAPI,
                                                              error_message=str(e))]
            return candidate_generation_output


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
