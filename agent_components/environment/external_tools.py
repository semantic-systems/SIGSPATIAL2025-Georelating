import math
import os
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

import h3
import pyproj
import requests
from shapely.geometry import Polygon

from models.candidates import ToponymWithCandidates, CandidateGenerationOutput, ReflectionPhase
from models.errors import Error, ExecutionStep
from models.llm_output import ToponymSearchArgumentsWithErrors, ValidatedOutput


class GeoNamesAPI:
    def __init__(self, article_id: str = None):
        self.base_url = "http://api.geonames.org/search?"

    REQUEST_TIMEOUT_S = 20
    # The per-toponym searches are independent I/O; a few run concurrently. Kept
    # modest to stay within the GeoNames free webservice's per-second throttle.
    MAX_SEARCH_CONCURRENCY = 5

    def search(self, params, username=None):
        # Per-request GeoNames account (if supplied) overrides the server default.
        params.update({'username': username or os.getenv('GEONAMES_USERNAME')})
        url = self.base_url + urllib.parse.urlencode(params)
        try:
            response = requests.get(url, timeout=self.REQUEST_TIMEOUT_S)
        except requests.exceptions.Timeout:
            response = None
        if response is None or response.status_code != 200:
            response = requests.get(url, timeout=self.REQUEST_TIMEOUT_S)  # retry once
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
            # Optional per-request GeoNames account carried on the graph state.
            geonames_username = getattr(validated_output, 'geonames_username', None)
            if hasattr(validated_output, 'reflection_phase'):
                if validated_output.reflection_phase == ReflectionPhase.ACTOR_RETRY_ON_INVALID_TOPONYMS:
                    topos_to_search = [topo for topo in validated_output.valid_toponyms if topo.generated_by_retry]
                    correct_duplicates = [topo for topo in validated_output.duplicate_toponyms if topo.generated_by_retry]
            def _search_with_fallback(toponym_to_search_for):
                """Search one toponym; on zero hits retry once with the plain toponym
                as 'q'. Returns (toponym, response). Independent I/O per toponym, so
                these run concurrently; only disjoint objects are touched."""
                response = self.search(toponym_to_search_for.params, username=geonames_username)
                if not response['geonames']:
                    # Deterministic repair before involving the critic: no hits are
                    # usually caused by overly specific search arguments, so retry
                    # once with the plainest possible query for the toponym.
                    fallback_params = {
                        'q': toponym_to_search_for.toponym,
                        'isNameRequired': True,
                        'maxRows': (toponym_to_search_for.params or {}).get('maxRows', 10),
                        'type': 'json'
                    }
                    response = self.search(fallback_params, username=geonames_username)
                    if response['geonames']:
                        # record the arguments that actually produced the candidates
                        toponym_to_search_for.params = fallback_params
                return toponym_to_search_for, response

            if topos_to_search:
                with ThreadPoolExecutor(
                        max_workers=min(self.MAX_SEARCH_CONCURRENCY, len(topos_to_search))) as pool:
                    # map preserves input order and re-raises the first worker
                    # exception here, keeping the fatal-error-on-failure semantics
                    search_results = list(pool.map(_search_with_fallback, topos_to_search))
            else:
                search_results = []

            for toponym_to_search_for, response in search_results:
                if not response['geonames']:
                    # Still nothing: flag the toponym as invalid so the critic can
                    # refine the arguments instead of passing an unfindable toponym on.
                    candidate_generation_output.invalid_toponyms.append(ToponymSearchArgumentsWithErrors(
                        **toponym_to_search_for.model_dump(),
                        errors_per_toponym=[Error(
                            execution_step=ExecutionStep.GEOAPI,
                            error_message="The GeoNames search returned zero results for these search arguments, "
                                          "even after a fallback search with the plain toponym as 'q'. The toponym "
                                          "may not exist in GeoNames under this name; try a different spelling or "
                                          "a coarser containing place."
                        )]
                    ))
                    # remove by name: object equality fails because search() adds the
                    # username to the params of the searched copy
                    candidate_generation_output.valid_toponyms = [
                        topo for topo in candidate_generation_output.valid_toponyms
                        if topo.toponym.casefold() != toponym_to_search_for.toponym.casefold()
                    ]
                    continue
                toponym_with_candidates = ToponymWithCandidates(
                    toponym_with_search_arguments=toponym_to_search_for,
                    total_results = response['totalResultsCount'],
                    candidates=response['geonames'],
                    nof_retrieved_candidates=len(response['geonames'])
                )
                candidate_generation_output.toponyms_with_candidates.append(toponym_with_candidates)
            for duplicate_toponym in correct_duplicates:
                duplicate_resolved = False
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
                        duplicate_resolved = True
                        break
                if not duplicate_resolved:
                    # The referenced toponym has no candidates (e.g. zero search hits),
                    # so this duplicate would silently lose coverage: flag it for the
                    # critic so the actor generates own search arguments for it.
                    candidate_generation_output.invalid_toponyms.append(ToponymSearchArgumentsWithErrors(
                        **duplicate_toponym.model_dump(),
                        errors_per_toponym=[Error(
                            execution_step=ExecutionStep.GEOAPI,
                            error_message=f"This toponym was marked as a duplicate of "
                                          f"'{duplicate_toponym.duplicate_of}', but no candidates could be "
                                          f"retrieved for that toponym. Generate own search arguments for this "
                                          f"toponym instead of marking it as a duplicate."
                        )]
                    ))
                    candidate_generation_output.duplicate_toponyms = [
                        topo for topo in candidate_generation_output.duplicate_toponyms
                        if topo.toponym.casefold() != duplicate_toponym.toponym.casefold()
                    ]
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


def _normalize_lng(lng, reference_lng):
    """Shift a longitude by +-360 so that it lies within 180 degrees of the reference
    longitude (avoids polygons broken at the antimeridian)."""
    while lng - reference_lng > 180:
        lng -= 360
    while lng - reference_lng < -180:
        lng += 360
    return lng


def _geodesic_circle(lat, lng, radius_m, n_points=64):
    """Geodesically accurate circle around (lat, lng) as a shapely Polygon in
    (lng, lat) coordinates, longitudes normalized around the center."""
    geod = pyproj.Geod(ellps="WGS84")
    azimuths = [i * 360.0 / n_points for i in range(n_points)]
    lngs, lats, _ = geod.fwd([lng] * n_points, [lat] * n_points, azimuths, [radius_m] * n_points)
    return Polygon([(_normalize_lng(x, lng), y) for x, y in zip(lngs, lats)])


def _cell_polygon(cell, reference_lng):
    boundary = h3.cell_to_boundary(cell)  # tuples of (lat, lng)
    return Polygon([(_normalize_lng(b_lng, reference_lng), b_lat) for b_lat, b_lng in boundary])


def smallest_covering_cell(lat, lng, area_km2):
    """
    The smallest single H3 cell that fully encompasses the predicted affected area
    (Definition 2.1 of the Georelating task). The area is modeled as a circle of
    radius sqrt(area / pi) around the predicted center. Starting from the resolution
    matching the area size, coarser resolutions are tried until the circle lies
    completely inside the cell containing the center (resolution 0 at the coarsest).
    Any covering cell must contain the center, so the candidate at each resolution
    is the center's own cell — deliberately NOT the previous cell's parent, since
    H3 parents do not fully contain their children geographically.
    """
    lat, lng = float(lat), float(lng)
    radius_m = math.sqrt(float(area_km2) * 1e6 / math.pi)
    circle = _geodesic_circle(lat, lng, radius_m)
    start_res = get_h3_resolution_for_area(float(area_km2) * 1e6)
    cell = None
    for res in range(start_res, -1, -1):
        cell = h3.latlng_to_cell(lat=lat, lng=lng, res=res)
        if _cell_polygon(cell, lng).contains(circle):
            return cell
    return cell


def safe_smallest_covering_cell(x):
    """Row-safe wrapper around smallest_covering_cell for DataFrame usage,
    mirroring safe_latlng_to_cell."""
    geor = x.get("georelated")
    if not geor or not isinstance(geor, dict):
        return None
    center = geor.get("center coordinates of affected area")
    area = geor.get("affected area in square km")
    if center is None:
        center = geor.get("center_coordinates_of_affected_area")
    if area is None:
        area = geor.get("affected_area_in_square_km")
    if not isinstance(center, dict):
        return None
    lat, lng = center.get("latitude"), center.get("longitude")
    if lat is None or lng is None or area is None:
        return None
    try:
        return smallest_covering_cell(lat, lng, area)
    except Exception:
        return None
