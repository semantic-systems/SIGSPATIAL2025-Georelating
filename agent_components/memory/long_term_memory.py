import json
import os
import random
import subprocess
from typing import Tuple, List, Dict

from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate

from models.candidates import CandidateGenerationState, GeoCodingState, ReflectionPhase, ToponymWithCandidatesShort, \
    CandidateResolutionInput, ToponymWithCandidates
from models.errors import Error
from helpers.helpers import preprocess_data

string_seperator = "\n\n----------------------------------------\n"

file_path = os.path.dirname(os.path.abspath(__file__))
root_path = subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).strip().decode()
GEONAMES_DOCUMENTATION = os.path.join(file_path, 'external_tool_documentation/geonames_websearch_documentation.md')
FEWSHOT_RESOLUTION_EXAMPLES_PATH = os.path.join(root_path, "data/few_shot_examples_selection_short.json")
class LongTermMemory:
    def __init__(self, documentation_file=GEONAMES_DOCUMENTATION):
        self.documentation_file = documentation_file
        self.documentation = self._load_documentation()
        self.system_instructions_prompt = self._create_system_instructions()
        self.task_instructions_prompt = self._create_task_instructions()
        self.documentation_prompt = self.create_documentation_message()
        self.reflective_actor_system_prompt = (
            "System: \n"
            "You are a reflective actor. Please strictly follow the feedback provided by the critic to generate a "
            "new output which does not lead to the errors your previous generation caused."
        )
        self.reflective_actor_instruction_prompt_text = "Your new output in plain JSON:\n"

    ####################################################################################################################
    # Baseline Candidate Generation
    ####################################################################################################################

    def _load_documentation(self):
        loader = TextLoader(self.documentation_file, encoding='utf-8')
        documentation = loader.load()
        return documentation

    @staticmethod
    def _create_system_instructions():
        return PromptTemplate.from_template(
            "System:\nYou are an expert API search assistant with comprehensive geographical knowledge. Your task "
            "is to create search parameters for the GeoNames Websearch API based on any provided news article. "
            "Ensure the search parameters are formatted strictly in JSON and comply exactly with the GeoNames "
            "Websearch API documentation. Your goal is to be precise and helpful, which you can best accomplish by "
            "following all instructions accurately and without deviation."
        )

    @staticmethod
    def _create_task_instructions():
        template = '''
            Human:
            Please create the search arguments for the GeoNames Websearch API based on the given news article.

            Your Task:

            1. Read the news article under the key 'News Article' to understand its content.
            2. Identify all the toponyms listed under the key 'Toponym List' within the article.
            3. For each toponym in the 'Toponym List,' generate the search arguments for the GeoNames Websearch API in JSON format.
            4. Strictly follow the JSON output format: [{"toponym": "<toponym>", "params": {"<search argument>": "<search_value>"}}].
            5. Ensure that the search arguments comply with the GeoNames Websearch API documentation.
            6. If any toponyms are duplicated based on the context of the news article, use the 'duplicate_of' key in the output JSON to reference the first occurrence of the toponym instead of the 'params' key.
            7. Typically, use the search argument 'q' with the toponym as the value, along with other relevant information such as upper administrative orders.
            8. Set the search argument 'isNameRequired' to 'true' to ensure relevant search results.
            9. Use the 'maxRows' search argument to limit the number of results returned.
            10. Dynamically select additional search arguments based on the context of the news article.
            11. Ensure the search arguments are as specific as possible to return only a few, highly relevant results.'''

        return PromptTemplate(
            template=template,
            template_format="mustache",
            input_variables=[]
        )

    def create_documentation_message(self):
        return PromptTemplate.from_template(
            f"Here is the documentation for the GeoNames Websearch API provided in Markdown:\n"
            f"{self.documentation[0].page_content}"
        )

    ####################################################################################################################
    # Reflective (Critic) Candidate Generation
    ####################################################################################################################

    def _generate_initial_generation_prompt_text(self, state: CandidateGenerationState | GeoCodingState) -> str:
        generated_output = str(state.raw_output.content)

        # exclude documentation from the initial prompt to make it shorter
        docu_string = self.create_documentation_message().format()
        initial_prompt = state.initial_prompt.replace(docu_string, "")

        initial_generation_prompt_text = (
            f"Original Prompt Text: \n {initial_prompt + string_seperator}"
            f"Generated Output: \n {generated_output}")
        return initial_generation_prompt_text

    def generate_critic_prompt_for_fatal_errors(self,
                                                state: CandidateGenerationState | GeoCodingState,
                                                fatal_errors: list[Error],
                                                initial_prompt: str = "") -> str:
        critic_system_prompt = (
            "System: \n"
            "You are a constructive critic for the actor LLM. Please analyze the errors in the generated output "
            "and provide actionable feedback to fix them. Your feedback will be directly used to guide the actor "
            "LLM to generate better outputs in the future. Focus on identifying the specific execution step where "
            "the error occurred and provide feedback ONLY for the cause of this error. Be as concise as possible."
            # Actionable feedback, c.g. Madaan et al. (2023) and Lauscher et al. (2024)
        )
        initial_generation_prompt = initial_prompt if initial_prompt else self._generate_initial_generation_prompt_text(
            state)
        fatal_errors = str([error.model_dump() for error in fatal_errors])
        critic_instruction = "Your feedback:\n"

        critic_prompt_text = (critic_system_prompt + string_seperator +
                              initial_generation_prompt + string_seperator +
                              f"Fatal Errors: \n {fatal_errors}" + string_seperator +
                              critic_instruction)
        prompt = PromptTemplate(
            template=critic_prompt_text,
            template_format="mustache",
            input_variables=[]
        )
        return prompt.format()

    def generate_critic_prompt_for_invalid_toponyms(self, state: CandidateGenerationState | GeoCodingState) -> str:
        critic_system_prompt = (
            "System: \n"
            "You are a constructive critic for the actor LLM which generated the output below. Please analyze the "
            "invalid toponyms and their corresponding errors in the generated output. Provide actionable feedback "
            "to fix these errors. Make sure your feedback adheres closely to the instructions in the original "
            "prompt. Your feedback will be directly used to guide the actor LLM to generate better outputs in the "
            "future. Focus only on the invalid toponyms based on the error message for each of them, using the "
            "valid toponyms as a reference. Be as concise as possible and ensure to include every invalid toponym "
            "provided below."
            # Actionable feedback, c.g. Madaan et al. (2023) and Lauscher et al. (2024)
        )
        initial_generation_prompt = self._generate_initial_generation_prompt_text(state)

        valid_examples, valid_examples_text = [], ""
        if len(state.valid_toponyms) > 2:
            valid_examples = [f"{topo.model_dump_json(indent=4)},\n" for topo in
                              random.sample(state.valid_toponyms, 2)]
        if len(state.duplicate_toponyms) > 2:
            valid_examples.extend(
                [f"{topo.model_dump_json(indent=4)},\n" for topo in random.sample(state.duplicate_toponyms, 2)])
        if valid_examples:
            valid_examples = "".join(valid_examples)
            valid_examples_text = f"Some valid generations for reference: \n [{valid_examples}]"
        invalid_toponyms = [f"{topo.model_dump_json(indent=4)},\n" for topo in state.invalid_toponyms]
        invalid_toponyms = "".join(invalid_toponyms)
        invalid_toponyms_text = f"All incorrect toponyms with errors: \n [{invalid_toponyms}]"
        critic_instruction = "Your feedback:\n"

        critic_prompt_text = (critic_system_prompt + string_seperator +
                              initial_generation_prompt + string_seperator +
                              valid_examples_text + string_seperator +
                              invalid_toponyms_text + string_seperator +
                              critic_instruction)
        prompt = PromptTemplate(
            template=critic_prompt_text,
            template_format="mustache",
            input_variables=[]
        )
        return prompt.format()

    def generate_reflected_actor_prompt(self, state: CandidateGenerationState | GeoCodingState) -> str:
        initial_generation_prompt_text = self._generate_initial_generation_prompt_text(state)
        feedback = f"Actionable feedback by the critic: \n{str(state.critic_feedback.content)}"

        invalid_prompt_part = ""

        if state.reflection_phase == ReflectionPhase.ACTOR_RETRY_ON_INVALID_TOPONYMS:
            invalids = str([topo.toponym for topo in state.invalid_toponyms])
            invalid_prompt_part = (f"Invalid toponyms: \n"
                                   f"{invalids + string_seperator}"
                                   "Now generate the search arguments ONLY for all invalid toponyms, NOT for the "
                                   "valid or duplicate ones. \n")

        reflected_prompt_text = (self.reflective_actor_system_prompt + string_seperator +
                                 initial_generation_prompt_text + string_seperator +
                                 feedback + string_seperator +
                                 invalid_prompt_part +
                                 self.reflective_actor_instruction_prompt_text)
        prompt = PromptTemplate(
            template=reflected_prompt_text,
            template_format="mustache",
            input_variables=[]
        )
        return prompt.format()

    ####################################################################################################################
    # Reflective (Critic) Candidate Resolution
    ####################################################################################################################
    @staticmethod
    def _generate_basic_instructions() -> Tuple[str, str]:
        instructions = ("System: \n"
                        "You are an expert with comprehensive geographical knowledge. Your task is to select the "
                        "correct candidate for each toponym listed under the 'Toponyms with candidates' key. Follow "
                        "these instructions:\n"
                        "1. **Candidate Selection**:\n"
                        " - Use your understanding of geography and the context of the news article to determine the "
                        "most likely candidate for each toponym.\n"
                        " - If no candidates match the toponym, select `null` and provide a detailed explanation.\n"
                        "2. **Reasoning**:\n"
                        " - Base your decision on the context of the news article, prioritizing geographical and "
                        "situational clues.\n"
                        " - Be concise yet precise in your reasoning, focusing on relevant parameters like name, "
                        "location, and type.\n"
                        "3. **Output**:\n"
                        " - Provide the result in JSON format following the example given below, ensuring consistency "
                        "in keys and structure.\n"
                        "4. **Error Handling**:\n"
                        " - If the context is ambiguous or insufficient, state this explicitly in your reasoning.")
        # load example from json file
        with open(FEWSHOT_RESOLUTION_EXAMPLES_PATH, "r") as f:
            example = json.load(f)
        example_text = (f"Example: \n"
                        f"Title: {example['title']}\n"
                        f"News Article: {example['news_article']}\n"
                        f"Toponyms with candidates: \n{json.dumps(example['toponyms_with_candidates'],
                                                                  indent=4)}\n"
                        f"Output in JSON: \n{json.dumps(example['ground_truth'],
                                                        indent=4)}\n")
        return instructions, example_text

    @staticmethod
    def _generate_input_candidates(article_title: str,
                                   article_text: str,
                                   toponyms_with_candidates_list: List[ToponymWithCandidates]) -> str:
        # Required fields for candidates
        CANDIDATE_FIELDS = ["adminName1", "countryName", "fclName", "fcodeName",
                            "geonameId", "lat", "lng", "name", "population", "toponymName"]

        # Processing with a generator for efficiency and clarity
        reduced_toponyms_with_candidates_list = [
            ToponymWithCandidatesShort(
                toponym=topo.toponym_with_search_arguments.toponym,
                candidates=[
                    {field: candidate[field] for field in CANDIDATE_FIELDS if field in candidate}
                    for candidate in topo.candidates
                ]
            )
            for topo in toponyms_with_candidates_list if not topo.toponym_with_search_arguments.duplicate_of
        ]
        toponyms_with_candidates = CandidateResolutionInput(toponyms_with_candidates=reduced_toponyms_with_candidates_list)
        input_text = (f"Ensure your response is structured and clear. Now, it's your turn. Respond only in JSON::\n"
                      f"Title: {article_title}\n"
                      f"News Article: {article_text}\n"
                      f"Toponyms with candidates: \n[{toponyms_with_candidates.model_dump_json(indent=4)}]\n"
                      f"Output in JSON: \n")
        return input_text

    def generate_candidate_resolution_prompt(self, state: GeoCodingState) -> str:
        """
        Resolve the candidates retrieved from the GeoNames API to one correct toponym.
        :param state: The current state of the reflective candidate generation.
        :return: The updated state of the reflective candidate generation.
        """
        if isinstance(state, CandidateGenerationState):
            raw_state = state.model_dump()
            # Preprocess the data to replace None values with model defaults
            preprocessed_state = preprocess_data(raw_state, GeoCodingState)
            # Parse the preprocessed data into the Pydantic model
            state = GeoCodingState(**preprocessed_state)

        instructions, example_text = self._generate_basic_instructions()
        input_text = self._generate_input_candidates(state.article_title,
                                                     state.article_text,
                                                     state.toponyms_with_candidates)

        prompt = PromptTemplate(
            template=instructions + string_seperator + example_text + string_seperator + input_text,
            template_format="mustache",
            input_variables=[]
        )
        return prompt.format()

    def _generate_initial_generation_prompt_for_resolution(self, state: GeoCodingState) -> str:
        # Generate the initial generation prompt text
        instructions, example_text = self._generate_basic_instructions()
        generated_output = str(state.resolution_raw_output.content)

        initial_generation_prompt = (
            f"Original Prompt Text: \n {instructions + example_text + string_seperator}"
            f"Generated Output: \n {generated_output}"
        )
        return initial_generation_prompt

    def generate_resolution_critic_prompt_for_fatal_errors(self, state: GeoCodingState) -> str:
        """
        Generate the prompt for the resolution critic in case of fatal errors.
        :param state: The current state of the reflective candidate resolution.
        :return: The formatted prompt for the resolution critic.
        """
        initial_generation_prompt = self._generate_initial_generation_prompt_for_resolution(state)
        return self.generate_critic_prompt_for_fatal_errors(state=state,
                                                            fatal_errors=state.resolution_fatal_errors,
                                                            initial_prompt=initial_generation_prompt)

    def generate_resolution_critic_prompt_for_invalid_toponyms(self, state: GeoCodingState) -> str:
        """
        Generate the prompt for the resolution critic in case of invalid, but not fatal outputs.
        :param state: The current state of the reflective candidate resolution.
        :return: The formatted prompt for the resolution critic.
        """
        critic_system_prompt = (
            "System: \n"
            "You are a constructive critic for the actor LLM which generated the output below. Please analyze the "
            "invalid toponyms and their corresponding errors in the generated output. Provide actionable feedback "
            "to fix these errors. Make sure your feedback adheres closely to the instructions in the original "
            "prompt. Your feedback will be directly used to guide the actor LLM to generate better outputs in the "
            "future. Focus ONLY on the invalid outputs strictly based on the error message for each of them."
            " Be as concise as possible and ensure to include every invalid output provided below."
            # Actionable feedback, c.g. Madaan et al. (2023) and Lauscher et al. (2024)
        )
        initial_generation_prompt = self._generate_initial_generation_prompt_for_resolution(state)

        invalid_outputs = [f"{topo.model_dump_json(indent=4)},\n" for topo in state.invalid_geocoded_toponyms
                           if not "Resolved toponym does not reference a toponym in the article" in topo.errors[
                -1].error_message]
        invalid_outputs = "".join(invalid_outputs)
        invalid_outputs_text = f"All invalid toponyms with errors: \n {invalid_outputs}"

        critic_instruction = "Your feedback:\n"

        critic_prompt_text = (critic_system_prompt + string_seperator +
                              initial_generation_prompt + string_seperator +
                              invalid_outputs_text + string_seperator +
                              critic_instruction)

        prompt = PromptTemplate(
            template=critic_prompt_text,
            template_format="mustache",
            input_variables=[]
        )
        return prompt.format()

    def generate_resolution_reflected_actor_prompt(self, state: GeoCodingState) -> str:
        """
        Generate the prompt for the reflective actor in the resolution phase.
        :param state: The current state of the reflective candidate resolution.
        :return: The formatted prompt for the reflective actor.
        """
        initial_generation_prompt_text = self._generate_initial_generation_prompt_for_resolution(state)
        feedback = f"Actionable feedback by the critic: \n{str(state.resolution_critic_feedback.content)}"

        invalid_prompt_part = ""

        if state.reflection_phase == ReflectionPhase.RESOLUTION_ACTOR_RETRY_ON_INVALID_RESOLUTIONS:
            invalids = [topo.toponym for topo in state.invalid_geocoded_toponyms
                        if not "Resolved toponym does not reference a toponym in the article" in
                               topo.errors[-1].error_message]
            missing_topos_with_candidates = [topo for topo in state.toponyms_with_candidates if
                                             topo.toponym_with_search_arguments.toponym in invalids]
            input_text = self._generate_input_candidates(state.article_title,
                                                         state.article_text,
                                                         missing_topos_with_candidates)

            invalids = ", ".join(invalids)
            invalid_prompt_part = (f"**Retry for article:** \n"
                                   f"{input_text + string_seperator}"
                                   f"Short List of all missing toponyms: [{invalids}]{string_seperator}"
                                   "Now select the candidates ONLY for all missing toponyms, NOT for the "
                                   "valid ones. \n")

        reflected_prompt_text = (self.reflective_actor_system_prompt + string_seperator +
                                 initial_generation_prompt_text + string_seperator +
                                 feedback + string_seperator +
                                 invalid_prompt_part)
        prompt = PromptTemplate(
            template=reflected_prompt_text,
            template_format="mustache",
            input_variables=[]
        )
        return prompt.format()

    ####################################################################################################################
    # GeoRelating
    ####################################################################################################################

    def _generate_georelating_system_prompt(self):
        prompt = PromptTemplate.from_template(
            "System:\n"
            "You are a helpful assistant which is very confident about it's geographical knowledge. Thus, "
            "you can provide the coordinates for the locations you are asked about. You provide them in decimal "
            "degrees and respond strictly in JSON format!"
        )
        return prompt.format()

    def _generate_georelating_task_instructions(self):
        prompt = PromptTemplate.from_template(
            "Human:\n"
            "You are provided with an article focused on a natural disaster. This article references "
            "several other geographical units (toponyms) for which you have been provided with coordinates. Your task "
            "has two steps:\n"
            "1. **Determine Coordinates**: Use your geographic understanding and the location description within the "
            "article to identify and output the center coordinates of the area affected by the natural disaster "
            "discussed in the article, utilizing the coordinates of the referenced toponyms as a guide.\n"
            "2. **Estimate Area**: Based on information from the article, identify the approximate area of the affected "
            "area in square kilometers (km^2).\n"
            "Please follow these instructions carefully to complete the task accurately."
        )
        return prompt.format()

    def _generate_georelating_example(self, example_path="data/few_shot_example_georelating.json"):
        with open(example_path, "r") as f:
            example = json.load(f)
        template = (
            f"Here is an example:\n{json.dumps(example, indent=4)}\n"
            f"Now it's your turn. Determine the coordinates and estimate the area!"
        )
        prompt = PromptTemplate(
            template=template,
            template_format="mustache",
            input_variables=[]
        )
        return prompt.format()

    def _generate_input_georelating(self,
                                    article_text: str,
                                    mentioned_toponyms: List[Dict],
                                    ) -> str:
        toponyms_with_coordinates = [
            {"toponym": topo["toponym"], "coordinates": topo["coordinates"]}
            for topo in mentioned_toponyms if topo["coordinates"] != {}
        ]
        input = {
            "article_text": article_text,
            "mentioned toponyms with coordinates": toponyms_with_coordinates
        }
        return f"{json.dumps(input, indent=4)}\nOutput in JSON: \n"

    def generate_georelating_prompt(self,
                                    article_text: str,
                                    mentioned_toponyms: List[Dict],
                                    example_path: str = "") -> str:
        system_prompt = self._generate_georelating_system_prompt()
        task_instructions = self._generate_georelating_task_instructions()
        example = self._generate_georelating_example(example_path=example_path) if (
            example_path) else self._generate_georelating_example()
        input = self._generate_input_georelating(article_text=article_text,
                                                 mentioned_toponyms=mentioned_toponyms)

        prompt = (system_prompt + string_seperator +
                  task_instructions + string_seperator +
                  example + string_seperator +
                  input)
        return prompt