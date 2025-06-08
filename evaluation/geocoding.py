import json
import os
import traceback
import pickle
from typing import Any

from pydantic import BaseModel, Field
from geopy.distance import geodesic
import numpy as np

from data_handler.xml_parsing import XMLDataHandler
from models.candidates import GeoCodedArticle
from models.errors import ExecutionStep


class CandidateGenerationMetrics(BaseModel):
    recall_at_10: float = Field(description="Recall@10 metric for the candidate generation task",
                                default=0)

    percentage_toponyms_with_fatal_errors: float = Field(
        description="Number of toponyms for which a fatal error occurred during the candidate generation",
        default=0)
    percentage_toponyms_without_valid_search_arguments: float = Field(
        description="Number of toponyms for which no parsable search arguments were generated",
        default=0)
    percentage_toponyms_without_candidates: float = Field(
        description="Number of toponyms for which the API call did not return any candidates",
        default=0)
    percentage_toponyms_without_correct_candidates: float = Field(
        description="Number of toponyms for which the correct candidate was not found in the top 10 candidates",
        default=0)
    percentage_too_many_generated_toponym_candidates: float = Field(
        description="Number of toponyms for which too many toponym candidates were generated",
        default=0)

    median_nof_candidates: np.floating = Field(description="Median number of candidates per toponym if candidates were found",
                                      default=0)
    nof_articles_with_fatal_errors: int = Field(
        description="Number of articles for which a fatal error occurred during the candidate generation",
        default=0)
    nof_all_gt_toponyms: int = Field(description="Total number of ground truth toponyms",
                                     default=0)
    nof_all_generated_toponyms: int = Field(description="Total number of generated toponyms",
                                            default=0)


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ExecutionStep):
            return str(obj)
        return super().default(obj)

class CandidateGenerationEvaluator:
    def __init__(self, data_directory: str, output_directory: str):
        self.data_directory = data_directory
        self.output_directory = output_directory
        if "GeoCoDe" in output_directory:
            self.data_handler = None
        else:
            self.data_handler = XMLDataHandler(data_directory)

    def load_generation_for_article(self,
                                    docid: str) -> Any | None:
        file_path = os.path.join(self.output_directory, f"{docid}.pkl")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Generated candidates for docid {docid} not found in {self.output_directory}.")
        try:
            with open(file_path, "rb") as f:
                document = pickle.load(f)
                return document
        except Exception as e:
            print(f"Error loading generation for article {docid}: {e}")
            traceback.print_exc()

    def calculate_candidate_generation_metrics(self) -> CandidateGenerationMetrics:
        total_toponyms = 0
        matched_toponyms = 0

        nof_toponyms_with_candidates = 0
        total_nof_candidates = []

        nof_toponyms_without_valid_search_arguments = 0
        nof_toponyms_without_candidates = 0

        too_many_generated_toponyms = 0
        nof_articles_with_too_many_generated_toponyms = 0

        too_few_generated_toponyms = 0
        nof_articles_with_too_few_generated_toponyms = 0

        nof_all_generated_toponyms = 0

        articles_with_fatal_errors = 0
        toponyms_with_fatal_errors = 0
        articles_with_critc = []

        generated_by_retry = 0
        article_with_critic = 0

        self.data_handler.parse_xml("LGL_test.xml")

        # Iterate through generated candidate files
        for candidate_file in os.listdir(self.output_directory):
            if not candidate_file.endswith(".pkl"):
                continue

            docid = candidate_file.replace(".pkl", "")

            try:
                generation_for_article = self.load_generation_for_article(docid)
            except FileNotFoundError:
                continue

            if (hasattr(generation_for_article, "reflected_prompt") and
                    generation_for_article.reflected_prompt is not None):
                article_with_critic += 1

            # Retrieve toponyms for the current article
            article_toponyms = self.data_handler.get_toponyms_for_article(docid)

            nof_article_generated_toponyms = len(
                generation_for_article.toponyms_with_candidates + generation_for_article.invalid_toponyms)
            nof_all_generated_toponyms += nof_article_generated_toponyms


            if generation_for_article.fatal_errors:
                articles_with_fatal_errors += 1
                print(f"Fatal error for article {docid}: {generation_for_article.fatal_errors}")
                if nof_article_generated_toponyms == 0:
                    toponyms_with_fatal_errors += len(article_toponyms)
            else:

                nof_toponyms_without_valid_search_arguments += len(generation_for_article.invalid_toponyms)

                # only for code error checking
                nof_article_toponyms = len(article_toponyms)
                if nof_article_generated_toponyms > nof_article_toponyms:
                    too_many_generated_toponyms += nof_article_generated_toponyms - nof_article_toponyms
                    nof_articles_with_too_many_generated_toponyms += 1
                elif nof_article_generated_toponyms < nof_article_toponyms:
                    too_few_generated_toponyms += nof_article_toponyms - nof_article_generated_toponyms
                    nof_articles_with_too_few_generated_toponyms += 1

            for toponym_row in article_toponyms:
                total_toponyms += 1
                correct_geonameid = toponym_row["geonameid"]

                # Check if the correct geonameid is in the top 10 candidates
                generated_toponyms_with_candidates = generation_for_article.toponyms_with_candidates.copy()
                for toponym in generated_toponyms_with_candidates:
                    if toponym.toponym_with_search_arguments.toponym == toponym_row["phrase"]:
                        generation_for_article.toponyms_with_candidates.remove(toponym)
                        if toponym.total_results == 0:
                            nof_toponyms_without_candidates += 1
                            break
                        else:
                            nof_toponyms_with_candidates += 1
                            total_nof_candidates.append(toponym.total_results)
                            if any(str(candidate.get("geonameId")) == correct_geonameid for candidate in
                                   toponym.candidates[:10]):
                                matched_toponyms += 1
                                if (hasattr(toponym.toponym_with_search_arguments, "generated_by_retry") and
                                        toponym.toponym_with_search_arguments.generated_by_retry):
                                    generated_by_retry += 1
                                break

        if total_toponyms > 0:
            recall_at_10 = matched_toponyms / total_toponyms

            percentage_toponyms_without_valid_search_arguments = nof_toponyms_without_valid_search_arguments / total_toponyms
            percentage_toponyms_without_candidates = nof_toponyms_without_candidates / total_toponyms
            percentage_toponyms_without_correct_candidates = (
                                                                         nof_toponyms_with_candidates - matched_toponyms) / total_toponyms
            percentage_toponyms_with_fatal_errors = toponyms_with_fatal_errors / total_toponyms
            percentage_too_many_generated_toponyms = too_many_generated_toponyms / total_toponyms
        else:
            recall_at_10 = \
                percentage_toponyms_without_valid_search_arguments = \
                percentage_toponyms_without_candidates = \
                percentage_toponyms_without_correct_candidates = \
                percentage_too_many_generated_toponyms = \
                percentage_toponyms_with_fatal_errors = \
                0

        # Calculate average number of candidates per toponym
        median_nof_candidates = np.median(total_nof_candidates)

        if nof_articles_with_too_few_generated_toponyms > 0:
            print(f'FATAL: articles with too few generated toponyms: {nof_articles_with_too_few_generated_toponyms}')
            print(f'FATAL: too few generated toponyms: {too_few_generated_toponyms}')

        print(f'Generated by retry: {generated_by_retry}, articles with critic: {article_with_critic}')
        print(f'w/o_critic: {(matched_toponyms - generated_by_retry) / (total_toponyms)}\n'
              f'w_critic: {matched_toponyms / (total_toponyms)}\n'
              f'critic_percentage: {generated_by_retry / (total_toponyms)}')

        return CandidateGenerationMetrics(
            recall_at_10=recall_at_10,
            percentage_toponyms_with_fatal_errors=percentage_toponyms_with_fatal_errors,
            percentage_toponyms_without_valid_search_arguments=percentage_toponyms_without_valid_search_arguments,
            percentage_toponyms_without_candidates=percentage_toponyms_without_candidates,
            percentage_toponyms_without_correct_candidates=percentage_toponyms_without_correct_candidates,
            percentage_too_many_generated_toponym_candidates=percentage_too_many_generated_toponyms,
            median_nof_candidates=median_nof_candidates,
            nof_articles_with_fatal_errors=articles_with_fatal_errors,
            nof_all_gt_toponyms=total_toponyms,
            nof_all_generated_toponyms=nof_all_generated_toponyms
        )

    def calculate_candidate_resolution_metrics(self,
                                               directory: str = "",
                                               k=161):
        """
        Calculate Accuracy@k and AUC for geocoding.

        Parameters:
        - ground_truth_coords: List of tuples [(lat1, lon1), ...] for ground truth points.
        - predicted_coords: List of tuples [(lat2, lon2), ...] for predicted points.
        - k: Threshold distance in km for Accuracy@k (default 161 km).

        Returns:
        - accuracy_at_k: Fraction of geocoded points within k km.
        - strict_accuracy_at_k: Fraction of all points within k km.
        - auc: Area under the curve value.
        - mean_error_distance: Mean error distance in km.
        - median_error_distance: Median error distance in km.
        - correct_articles: List of GeoCodedArticle objects with all correct toponyms.
        - nof_articles_with_fatal_errors: Number of articles with generation or resolution fatal errors.
        """


        total_toponyms = 0
        topos_with_incorrect_geonameid = 0
        topos_without_geonameid = 0

        correct_articles = []
        articles_with_at_least_one_correct_toponym = []
        nof_articles_with_fatal_errors = 0

        error_distances = []
        error_distances_for_correct_articles = []

        generated_by_retry = 0
        critic_intervention = 0
        articles_with_critic = []
        nof_articles = 0

        if "GeoCoDe" in directory:
            with open("data/processed_GeoCoDe_test.json", "r", encoding="utf-8") as f:
                geocode_articles = json.load(f)
        else:
            self.data_handler.parse_xml("LGL_test.xml")

        # Iterate through generated candidate files
        for candidate_file in os.listdir(directory):
            if not candidate_file.endswith(".pkl"):
                continue

            docid = candidate_file.replace(".pkl", "")

            try:
                generation_for_article = self.load_generation_for_article(docid)
            except FileNotFoundError:
                continue

            nof_articles += 1

            # Retrieve toponyms for the current article
            article_toponyms = []
            if "GeoCoDe" in directory:
                for model in geocode_articles:
                    if model['article_id'].split()[0] == docid:
                        article_toponyms = model["toponym_data"]
                        break
            else:
                article_toponyms = self.data_handler.get_toponyms_for_article(docid)

            if generation_for_article.fatal_errors:
                nof_articles_with_fatal_errors += 1
                print(f"Fatal error for article {docid}: {generation_for_article.fatal_errors}")
            elif generation_for_article.resolution_fatal_errors:
                nof_articles_with_fatal_errors += 1
                print(f"Fatal error for article {docid}: {generation_for_article.resolution_fatal_errors}")

            if not generation_for_article.resolution_critic_prompt == '':
                critic_intervention += 1
                articles_with_critic.append(generation_for_article)


            selected_candidates = generation_for_article.valid_geocoded_toponyms.copy()
            toponyms_with_candidates = generation_for_article.toponyms_with_candidates.copy()

            correct_toponyms_for_article = []
            error_distances_for_article = []

            for gt_toponym in article_toponyms:
                total_toponyms += 1
                if "GeoCoDe" in directory:
                    correct_latitude = gt_toponym["coordinates"]["latitude"]
                    correct_longitude = gt_toponym["coordinates"]["longitude"]
                    gt_coords = (correct_latitude, correct_longitude)
                    gt_toponym_name = gt_toponym["toponym"]
                else:
                    correct_geonameid = gt_toponym["geonameid"]
                    correct_latitude = gt_toponym["lat"]
                    correct_longitude = gt_toponym["lon"]
                    gt_coords = (correct_latitude, correct_longitude)
                    gt_toponym_name = gt_toponym["phrase"]

                for toponym in selected_candidates:
                    if toponym.toponym.casefold() == gt_toponym_name.casefold():
                        if toponym.selected_candidate_geonameId in [None, 0]:
                            topos_without_geonameid += 1
                            selected_candidates.remove(toponym)
                            break
                        for item in toponyms_with_candidates:
                            if item.toponym_with_search_arguments.toponym.casefold() == toponym.toponym.casefold():
                                # get the candidate with the correct geonameid
                                incorrect_geonameid = True
                                for candidate in item.candidates:
                                    if toponym.selected_candidate_geonameId == candidate["geonameId"]:
                                        generated_coords = (candidate["lat"], candidate["lng"])
                                        error_distance = geodesic(gt_coords, generated_coords).kilometers
                                        error_distances.append(error_distance)
                                        if error_distance <= k:
                                            toponym.coordinates = {"latitude": candidate["lat"],
                                                                   "longitude": candidate["lng"]}
                                            correct_toponyms_for_article.append(toponym)
                                            error_distances_for_article.append(error_distance)
                                        if hasattr(toponym, 'generated_by_retry') and toponym.generated_by_retry:
                                            generated_by_retry += 1
                                        incorrect_geonameid = False
                                        break
                                if incorrect_geonameid:
                                    topos_with_incorrect_geonameid += 1
                                toponyms_with_candidates.remove(item)
                                break
                        selected_candidates.remove(toponym)
                        break
            if len(correct_toponyms_for_article) == len(article_toponyms):
                correct_articles.append(GeoCodedArticle(**generation_for_article.model_dump(),
                                                        toponym_data=article_toponyms,
                                                        correctly_geocoded_toponyms=correct_toponyms_for_article))
                error_distances_for_correct_articles.extend(error_distances_for_article)
            elif len(correct_toponyms_for_article) > 0:
                articles_with_at_least_one_correct_toponym.append(GeoCodedArticle(
                    **generation_for_article.model_dump(),
                    correctly_geocoded_toponyms=correct_toponyms_for_article
                ))
        articles_with_at_least_one_correct_toponym.extend(correct_articles)

        print(f'Generated by retry: {generated_by_retry}')
        print(f'share of articles with critic intervention: {critic_intervention/nof_articles}')

        # Accuracy@k
        within_k = [d <= k for d in error_distances]
        accuracy_at_k = sum(within_k) / len(error_distances)
        print(f"Accuracy@{k}: {accuracy_at_k}")

        strict_accuracy_at_k = sum(within_k) / total_toponyms
        print(f"Strict Accuracy@{k}: {strict_accuracy_at_k}")

        # AUC
        def calculate_auc(sorted_values):
            max_error = 20038  # Earth's circumference in km / 2 (maximum possible distance)
            size = len(sorted_values)
            if size <= 1:
                return 0.0

            h = 1  # step size
            sum = 0.5 * (np.log(1 + sorted_values[0]) / np.log(max_error) + np.log(
                1 + sorted_values[-1]) / np.log(max_error))

            for i in range(1, size - 1):
                sum += np.log(1 + sorted_values[i]) / np.log(max_error)

            auc = sum * h / (size - 1)
            return auc

        sorted_error_distances = sorted(
            error_distances)
        auc = calculate_auc(sorted_error_distances)
        print(f"AUC: {auc}")

        # Mean error distance
        mean_error_distance = np.mean(error_distances)
        print(f"Mean error distance: {mean_error_distance}")

        # Median error distance
        median_error_distance = np.median(error_distances)
        print(f"Median error distance: {median_error_distance}")

        print(f"Number of correctly geocoded articles: {len(correct_articles)}")
        print(f"Number of articles with at least one correctly geocoded toponym: {len(articles_with_at_least_one_correct_toponym)}")
        print(f"Median error distance for correctly geocoded articles: {np.median(error_distances_for_correct_articles)}")
        print(f"Mean error distance for correctly geocoded articles: {np.mean(error_distances_for_correct_articles)}")


        return (accuracy_at_k, strict_accuracy_at_k, auc, mean_error_distance, median_error_distance, correct_articles,
                articles_with_at_least_one_correct_toponym, nof_articles_with_fatal_errors)
