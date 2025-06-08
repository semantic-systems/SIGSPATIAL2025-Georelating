import copy
import json
import re
import itertools
from typing import Any

import pycountry
from langchain_core.messages import AIMessage

from models.candidates import ReflectionPhase, ResolvedToponym, GeoCodingState, ResolvedToponymWithErrors
from models.errors import Error, ExecutionStep
from models.llm_output import ToponymSearchArguments, ToponymSearchArgumentsWithErrors, ValidatedOutput, LLMOutput


def _is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def _is_integer(value):
    try:
        int(value)
        return True
    except ValueError:
        return False


def _is_valid_countrycode(code):
    return pycountry.countries.get(alpha_2=code.upper()) is not None


def _extract_thoughts(text):
    # Regular expression to capture content inside <think>...</think>
    pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)

    thoughts = pattern.findall(text)  # Extract thoughts
    non_thoughts = pattern.sub("", text)  # Remove thoughts from text

    return thoughts, non_thoughts.strip()


class OutputParser:
    def __init__(self, article_id: str = "", toponym_list=None):
        if toponym_list is None:
            toponym_list = [""]
        self.article_id = article_id
        self.toponym_list = toponym_list

    @staticmethod
    def clean_and_parse_json_content(content: str, start_token: str = '[', end_token: str = ']', return_thoughts: bool = False) -> \
    tuple[Any, list[Any] | list[str]] | Any:
        """
        Ensures the content is valid JSON by removing characters before the first '['
        and after the last ']'. Then parses the JSON content.

        Args:
            content (str): The raw content to be cleaned and parsed.
            start_token (str): The starting token to find the beginning of the JSON content.
            end_token (str): The ending token to find the end of the JSON content.
            return_thoughts (bool): If True, returns the chain of thoughts along with the parsed content.

        Returns:
            list: The parsed JSON content as a Python object.

        Raises:
            ValueError: If the content is not valid JSON.
        """
        chain_of_thought, content = _extract_thoughts(content)
        start = content.find("```json") + 7
        end = content.find("```", start)  # Find the next occurrence after start
        if end != -1:  # Ensure closing backticks exist
            content = content[start:end]
        if content[0] != start_token:
            content = content[content.find(start_token):]
        if content[-1] != end_token:
            content = content[:content.rfind(end_token) + 1]
        if return_thoughts:
            return json.loads(content), chain_of_thought
        return json.loads(content)

    @staticmethod
    def handle_parsing_error(e: Exception, step: ExecutionStep):
        """
        Creates an error message for a parsing error.

        Args:
            e (Exception): The exception raised during parsing.
            step (ExecutionStep): The step where the error occurred.

        Returns:
            Error: The constructed error object.
        """
        error_message = f"Error while parsing output: The generated content does not seem to be valid JSON. Error: {e}"
        return Error(execution_step=step, error_message=error_message)

    @staticmethod
    def handle_exceptions(step: ExecutionStep):
        """
        Decorator to handle exceptions during parsing and add an error to the relevant state or output.

        Args:
            step (ExecutionStep): The step where the error occurred.
        """

        def decorator(func):
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error = OutputParser.handle_parsing_error(e, step)
                    if isinstance(args[1], GeoCodingState):  # Handle GeoCodingState
                        args[1].resolution_fatal_errors.append(error)
                        return args[1]
                    elif isinstance(args[1], AIMessage):  # Handle LLMOutput
                        output = LLMOutput(
                            article_id=args[0].article_id,
                            toponyms=args[0].toponym_list,
                            raw_output=args[1].model_copy(),
                            parsed_output=[],
                            fatal_errors=[error]
                        )
                        return output

            return wrapper

        return decorator

    @handle_exceptions(ExecutionStep.SEARCHOUTPUTPARSER)
    def extract_generation_output(self, message: AIMessage) -> LLMOutput:
        parsed_output = [
            ToponymSearchArguments.model_validate(location_mention)
            for location_mention in self.clean_and_parse_json_content(message.content)
        ]
        llm_output = LLMOutput(
            article_id=self.article_id,
            toponyms=self.toponym_list,
            raw_output=message.model_copy(),
            parsed_output=parsed_output
        )
        return llm_output

    @handle_exceptions(ExecutionStep.SEARCHOUTPUTPARSER)
    def extract_resolution_output(self, state: GeoCodingState):
        state.geocoded_toponyms = []
        resolved_toponyms = [
            ResolvedToponym.model_validate(location_mention)
            for location_mention in self.clean_and_parse_json_content(state.resolution_raw_output.content)
        ]
        state.geocoded_toponyms.extend(resolved_toponyms)

        # Since only non-duplicates are provided to the actor LLM, we need to add the duplicates to the list of
        # resolved toponyms manually afterward  in the initial try of the resolution actor
        if state.reflection_phase == ReflectionPhase.RESOLUTION_ACTOR_GENERATION:
            for topo in state.toponyms_with_candidates:
                if topo.toponym_with_search_arguments.duplicate_of:
                    for resolved_toponym in resolved_toponyms:
                        if resolved_toponym.toponym.casefold() == topo.toponym_with_search_arguments.duplicate_of.casefold():
                            adjusted_toponym = copy.copy(resolved_toponym)
                            adjusted_toponym.toponym = topo.toponym_with_search_arguments.toponym
                            state.geocoded_toponyms.append(adjusted_toponym)
                            break
        else:
            for retry_generated_resolution in state.geocoded_toponyms:
                # set the generated_by_retry flag to True for the retry generated resolutions
                retry_generated_resolution.generated_by_retry = True
        return state


class SearchParameterSyntaxValidator:
    def __init__(self):
        self.params = {}
        self.errors = []
        self.required_params = ['q', 'name', 'name_equals']
        self.optional_params = [
            'name_startsWith', 'maxRows', 'startRow', 'country', 'countryBias',
            'continentCode', 'adminCode1', 'adminCode2', 'adminCode3', 'adminCode4',
            'adminCode5', 'featureClass', 'featureCode', 'cities', 'lang', 'type',
            'style', 'isNameRequired', 'tag', 'operator', 'charset', 'fuzzy',
            'east', 'west', 'north', 'south', 'searchlang', 'orderby', 'inclBbox'
        ]
        self.continent_codes = ['AF', 'AS', 'EU', 'NA', 'OC', 'SA', 'AN']

    def validate(self, params) -> tuple[bool, list[str]]:
        """
        Validate the syntax of the GeoNames API parameters
        :param params: dict
        :return: tuple[bool, list[str]] - (is_valid, errors)
        """
        self.params = params
        self.errors = []
        self.validate_required_params()
        self.validate_optional_params()
        self.validate_bounding_box()
        self.validate_fuzzy()
        self.validate_maxRows()
        self.validate_startRow()
        self.validate_country()
        self.validate_featureClass()
        self.validate_featureCode()
        self.validate_cities()
        self.validate_lang()
        self.validate_type()
        self.validate_style()
        self.validate_isNameRequired()
        self.validate_tag()
        self.validate_operator()
        self.validate_charset()
        self.validate_orderby()
        self.validate_inclBbox()

        return len(self.errors) == 0, self.errors

    def validate_required_params(self):
        if not any(param in self.params for param in self.required_params):
            self.errors.append("One of 'q', 'name', or 'name_equals' is required.")
        else:
            for param in self.required_params:
                if param in self.params:
                    if not isinstance(self.params[param], str):
                        self.errors.append(f"{param} must be a string.")

    def validate_optional_params(self):
        for param in self.params:
            if param not in self.required_params + self.optional_params:
                self.errors.append(f"Invalid parameter: {param}")

    def validate_name_startsWith(self):
        if 'name_startsWith' in self.params and not isinstance(self.params['name_startsWith'], str):
            self.errors.append("name_startsWith must be a string.")

    def validate_maxRows(self):
        if 'maxRows' in self.params and not _is_integer(self.params['maxRows']):
            self.errors.append("maxRows must be an integer.")

    def validate_startRow(self):
        if 'startRow' in self.params:
            if not _is_integer(self.params['startRow']):
                self.errors.append("startRow must be an integer.")
            elif not 0 <= int(self.params['startRow']) <= 5000:
                self.errors.append("startRow must be non-negative and maximum 5000.")

    def validate_country(self):
        if 'country' in self.params and not _is_valid_countrycode(self.params['country']):
            self.errors.append("country must be a ISO 3166-1 Alpha-2 Code.")

    def validate_countryBias(self):
        if 'countryBias' in self.params and not _is_valid_countrycode(self.params['countryBias']):
            self.errors.append("countryBias must be a ISO 3166-1 Alpha-2 Code.")

    def validate_continentCode(self):
        if 'continentCode' in self.params and self.params not in self.continent_codes:
            self.errors.append(f"continentCode must be one of {self.continent_codes}.")

    def validate_featureClass(self):
        if 'featureClass' in self.params and not re.match(r'^[AHLPRSTUV]+$', self.params['featureClass']):
            self.errors.append("featureClass must be one or more of A,H,L,P,R,S,T,U,V.")

    def validate_featureCode(self):
        if 'featureCode' in self.params and not isinstance(self.params['featureCode'], str):
            self.errors.append("featureCode must be a string.")

    def validate_cities(self):
        if 'cities' in self.params and self.params['cities'] not in ['cities1000', 'cities5000', 'cities15000']:
            self.errors.append("cities must be one of 'cities1000', 'cities5000', 'cities15000'.")

    def validate_lang(self):
        if 'lang' in self.params:
            self.errors.append("language shall always be English.")

    def validate_type(self):
        if 'type' in self.params and self.params['type'] != 'json':
            self.errors.append("type shall always be json.")

    def validate_style(self):
        if 'style' in self.params and self.params['style'] not in ['SHORT', 'MEDIUM', 'LONG', 'FULL']:
            self.errors.append("style must be one of 'SHORT', 'MEDIUM', 'LONG', 'FULL'.")

    def validate_isNameRequired(self):
        if 'isNameRequired' in self.params and not isinstance(self.params['isNameRequired'], bool):
            self.errors.append("isNameRequired must be boolean.")

    def validate_tag(self):
        if 'tag' in self.params and not isinstance(self.params['tag'], str):
            self.errors.append("tag must be a string.")

    def validate_operator(self):
        if 'operator' in self.params and self.params['operator'] not in ['AND', 'OR']:
            self.errors.append("operator must be one of 'AND', 'OR'.")

    def validate_charset(self):
        if 'charset' in self.params and self.params['charset'] != "UTF8":
            self.errors.append("charset must be UTF-8.")

    def validate_fuzzy(self):
        if 'fuzzy' in self.params and not _is_float(self.params['fuzzy']):
            self.errors.append("fuzzy must be a float between 0 and 1.")

    def validate_bounding_box(self):
        bbox_params = ['east', 'west', 'north', 'south']
        if any(param in self.params for param in bbox_params):
            for param in bbox_params:
                if param in self.params and not _is_float(self.params[param]):
                    self.errors.append(f"{param} must be a float.")

    def validate_orderby(self):
        if 'orderby' in self.params and self.params['orderby'] not in ['population', 'elevation', 'relevance']:
            self.errors.append("orderby must be one of 'population', 'elevation', 'relevance'.")

    def validate_inclBbox(self):
        if 'inclBbox' in self.params and self.params['inclBbox'] != 'true':
            self.errors.append("inclBbox can only be true.")

    def validate_username(self):
        if 'username' in self.params:
            self.errors.append("username shall not be provided.")


class ArticleSyntaxValidator:
    def __init__(self):
        self.geonames_syntax_validator = SearchParameterSyntaxValidator()

    def validate_toponyms_of_article(self, llm_output: LLMOutput) -> ValidatedOutput:
        validated_output = ValidatedOutput(**llm_output.model_dump())
        try:
            # needed in baseline generation chain, not in agent graph
            if validated_output.fatal_errors:
                return validated_output

            temp_gt_toponyms = [toponym.casefold() for toponym in validated_output.toponyms]

            in_reflection_phase = False
            if hasattr(llm_output, 'reflection_phase'):  # needed in candidate generation chain, not in agent graph
                if llm_output.reflection_phase == ReflectionPhase.ACTOR_RETRY_ON_INVALID_TOPONYMS:
                    in_reflection_phase = True
                    for toponym_with_search_params in validated_output.parsed_output:
                        toponym_with_search_params.generated_by_retry = True
                    already_correct_toponyms = [topo.toponym.casefold() for topo in itertools.chain(
                        validated_output.valid_toponyms, validated_output.duplicate_toponyms)]
                    # remove already correct toponyms from the list of toponyms to generate
                    for correct_topo in already_correct_toponyms:
                        temp_gt_toponyms.remove(correct_topo)

            # First, all parsed toponyms can either have valid syntax, invalid syntax, or be duplicates
            for toponym_with_search_params in validated_output.parsed_output:
                if toponym_with_search_params.params:
                    is_valid, errors = self.geonames_syntax_validator.validate(toponym_with_search_params.params)
                    if is_valid:
                        validated_output.valid_toponyms.append(toponym_with_search_params)
                    else:
                        validated_output.invalid_toponyms.append(ToponymSearchArgumentsWithErrors(
                            **toponym_with_search_params.model_dump(),
                            errors_per_toponym=[
                                Error(
                                    execution_step=ExecutionStep.SEARCHPARAMETERVALIDATOR,
                                    error_message=str(errors)
                                )
                            ]
                        ))
                elif toponym_with_search_params.duplicate_of:
                    validated_output.duplicate_toponyms.append(toponym_with_search_params)
                else:
                    validated_output.invalid_toponyms.append(ToponymSearchArgumentsWithErrors(
                        toponym=toponym_with_search_params.toponym,
                        errors_per_toponym=[
                            Error(
                                execution_step=ExecutionStep.SEARCHPARAMETERVALIDATOR,
                                error_message="Toponym without a valid search parameter (params) or duplicate ("
                                              "duplicate_of) key."
                            )
                        ]
                    ))

            # Second, we need to check if the toponyms with valid syntax are actually in the article (both for valid
            # and duplicate toponyms) create local copies of lists to avoid changing the original lists and allow
            # stable iteration

            if not in_reflection_phase:
                valid_toponyms = validated_output.valid_toponyms.copy()
                duplicate_toponyms = validated_output.duplicate_toponyms.copy()
            else:
                valid_toponyms = [valid for valid in validated_output.valid_toponyms if valid.generated_by_retry]
                duplicate_toponyms = [duplicate for duplicate in validated_output.duplicate_toponyms if
                                      duplicate.generated_by_retry]

            for generation in itertools.chain(valid_toponyms, duplicate_toponyms):
                if generation.toponym.casefold() in temp_gt_toponyms:
                    temp_gt_toponyms.remove(generation.toponym.casefold())
                else:
                    validated_output.invalid_toponyms.append(ToponymSearchArgumentsWithErrors(
                        **generation.model_dump(),
                        errors_per_toponym=[
                            Error(
                                execution_step=ExecutionStep.SEARCHPARAMETERVALIDATOR,
                                error_message="Generation does not reference a toponym in the article. Only generate "
                                              "search parameters for the toponyms provided under the toponyms key!"
                            )
                        ]
                    ))
                    if generation in validated_output.valid_toponyms:
                        validated_output.valid_toponyms.remove(generation)
                    else:
                        validated_output.duplicate_toponyms.remove(generation)

            # Third, we need to check if all duplicates reference a valid toponym
            if in_reflection_phase:
                duplicate_toponyms = [duplicate for duplicate in validated_output.duplicate_toponyms if
                                      duplicate.generated_by_retry]
            else:
                duplicate_toponyms = validated_output.duplicate_toponyms.copy()
            for duplicate in duplicate_toponyms:
                valid_duplicate = False
                for valid_toponym in validated_output.valid_toponyms:  # harsh because duplicate could also be correctly referring to an invalid toponym
                    if duplicate.duplicate_of.casefold() == valid_toponym.toponym.casefold():
                        valid_duplicate = True
                        break
                if not valid_duplicate:  # didn't find a valid toponym to reference for the duplicate
                    validated_output.invalid_toponyms.append(ToponymSearchArgumentsWithErrors(
                        **duplicate.model_dump(),
                        errors_per_toponym=[
                            Error(
                                execution_step=ExecutionStep.SEARCHPARAMETERVALIDATOR,
                                error_message="Duplicate toponym was generated correctly, but does not reference a "
                                              "valid generated toponym under the duplicate_of key."
                            )
                        ]
                    ))
                    validated_output.duplicate_toponyms.remove(duplicate)

            if in_reflection_phase:
                # remove the newly generated toponyms from the list of invalid toponyms
                for new_topo in itertools.chain(validated_output.valid_toponyms, validated_output.duplicate_toponyms,
                                                validated_output.invalid_toponyms):
                    if new_topo.generated_by_retry:
                        for invalid_topo in validated_output.invalid_toponyms:
                            if not invalid_topo.generated_by_retry:
                                if new_topo.toponym.casefold() == invalid_topo.toponym.casefold():
                                    validated_output.invalid_toponyms.remove(invalid_topo)
                                    break

            # the sum of all generated toponyms
            temp_toponym_list = [temp_toponym.toponym.casefold() for temp_toponym in (validated_output.valid_toponyms +
                                                                                      validated_output.duplicate_toponyms +
                                                                                      validated_output.invalid_toponyms)]
            if len(temp_toponym_list) < len(validated_output.toponyms):  # too few toponyms generated
                for mention in validated_output.toponyms:
                    if mention.casefold() not in temp_toponym_list:
                        validated_output.invalid_toponyms.append(ToponymSearchArgumentsWithErrors(
                            toponym=mention,
                            errors_per_toponym=[
                                Error(
                                    execution_step=ExecutionStep.SEARCHPARAMETERVALIDATOR,
                                    error_message="No search arguments were generated for this toponym."
                                )
                            ]
                        ))
                    else:
                        temp_toponym_list.remove(mention.casefold())

            nof_valid_invalid_duplicate = len(
                [temp_toponym.toponym for temp_toponym in (validated_output.valid_toponyms +
                                                           validated_output.duplicate_toponyms +
                                                           validated_output.invalid_toponyms)])

            nof_valid_duplicate = len([temp_toponym.toponym for temp_toponym in (validated_output.valid_toponyms +
                                                                                 validated_output.duplicate_toponyms)])

            correct_nof_toponyms = len(validated_output.toponyms)
            if nof_valid_invalid_duplicate < correct_nof_toponyms:  # indicates most probably a coding error
                print("FATAL ERROR: TOO FEW TOPONYMS: VALIDATION ERROR for article ", validated_output.article_id)
            if nof_valid_duplicate > correct_nof_toponyms:  # indicates most probably a coding error
                print("FATAL ERROR: TOO MANY TOPONYMS: VALIDATION ERROR for article ", validated_output.article_id)

            return validated_output
        except Exception as e:
            validated_output.fatal_errors = [Error(execution_step=ExecutionStep.ARTICLESYNTAXVALIDATOR,
                                                   error_message=str(e))]
            return validated_output


class ResolutionSyntaxValidator:

    def validate_resolution(self, state: GeoCodingState) -> GeoCodingState:
        """
        Validate the syntax of the resolved toponyms
        :param state: GeoCodingState - The graph state containing the resolved toponyms
        :return: GeoCodingState - The graph state additionally containing the validated resolved toponyms
        """
        try:
            state.invalid_geocoded_toponyms = []

            # Create a list of toponyms to resolve, excluding already resolved ones
            temp_gt_toponyms = [
                str(toponym.toponym_with_search_arguments.toponym)
                for toponym in state.toponyms_with_candidates
            ]
            if state.valid_geocoded_toponyms:
                for valid_toponym in state.valid_geocoded_toponyms:
                    temp_gt_toponyms.remove(valid_toponym.toponym)

            # Copy resolved toponyms
            resolved_toponyms = state.geocoded_toponyms.copy()

            def _validate_resolved_toponym(resolved_toponym):
                """Validates the resolved toponym and returns errors if any."""
                errors = []
                if not isinstance(resolved_toponym.toponym, str):
                    errors.append("Toponym must be a string.")
                if not isinstance(resolved_toponym.reasoning, str):
                    errors.append("Reasoning must be a string.")
                if resolved_toponym.selected_candidate_geonameId and not _is_integer(
                        resolved_toponym.selected_candidate_geonameId):
                    errors.append("Selected candidate geonameId must be an integer.")
                return errors

            # Validate resolved toponyms
            remaining_toponyms = temp_gt_toponyms[:]
            for resolved_toponym in resolved_toponyms:
                errors = _validate_resolved_toponym(resolved_toponym)

                # Check if the resolved toponym is in the article
                matched = next(
                    (i for i, topo in enumerate(remaining_toponyms) if
                     topo.casefold() == resolved_toponym.toponym.casefold()),
                    None
                )

                if matched is not None:
                    del remaining_toponyms[matched]  # Remove the matched toponym from the remaining list
                else:
                    errors.append("Resolved toponym does not reference a toponym in the article.")

                if errors:
                    state.invalid_geocoded_toponyms.append(ResolvedToponymWithErrors(
                        **resolved_toponym.model_dump(),
                        errors=[Error(
                            execution_step=ExecutionStep.RESOLUTIONSYNTAXVALIDATOR,
                            error_message=", ".join(errors)
                        )]
                    ))
                else:
                    state.valid_geocoded_toponyms.append(resolved_toponym)

            # Add unresolved toponyms to invalid list
            for toponym in remaining_toponyms:
                state.invalid_geocoded_toponyms.append(ResolvedToponymWithErrors(
                    toponym=toponym,
                    reasoning="",
                    selected_candidate_geonameId=0,
                    errors=[Error(
                        execution_step=ExecutionStep.RESOLUTIONSYNTAXVALIDATOR,
                        error_message="LLM resolution step is missing a generation for this toponym."
                    )]
                ))

            return state
        except Exception as e:
            state.resolution_fatal_errors.append(Error(execution_step=ExecutionStep.RESOLUTIONSYNTAXVALIDATOR,
                                                       error_message=str(e)))
            return state
