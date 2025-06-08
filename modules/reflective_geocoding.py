import json
import pickle
import os
import time

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.constants import END
from langgraph.graph import StateGraph
from openai import APIStatusError
from tqdm import tqdm

from agent_components.environment.external_tools import GeoNamesAPI
from agent_components.environment.internal_tools import ArticleSyntaxValidator, OutputParser, ResolutionSyntaxValidator
from agent_components.llms.api_error_handler import handle_api_errors
from agent_components.llms.chatAI import ChatAIHandler
from agent_components.memory.working_memory import WorkingMemory
from evaluation.geocoding import CandidateGenerationEvaluator
from helpers.helpers import preprocess_data
from models.candidates import CandidateGenerationState, CandidateGenerationOutput, ReflectionPhase, GeoCodingState
from models.errors import Error, ExecutionStep
from models.llm_output import LLMOutput, ValidatedOutput


class ReflectiveGeoCoder:
    def __init__(self,
                 actor_model_name: str = "meta-llama-3.1-8b-instruct",
                 critic_model_name: str = "meta-llama-3.1-8b-instruct",
                 call_times: list = None,
                 skip_few_shot_loader: bool = False,
                 data_set: str = 'LGL'):
        load_dotenv()
        self.working_memory = WorkingMemory(skip_few_shot_loader=skip_few_shot_loader)
        self.data_handler = self.working_memory.few_shot_handler.data_handler
        self.llm_handler = ChatAIHandler()
        self.llm = self.llm_handler.get_model(actor_model_name)
        self.critic_llm = self.llm_handler.get_model(critic_model_name)
        self.validator = ArticleSyntaxValidator()
        self.geonames = GeoNamesAPI()
        self.output_parser = OutputParser()
        self.resolution_validator = ResolutionSyntaxValidator()
        self.call_times = call_times if call_times else []
        self.data_set = data_set

    ####################################################################################################################
    """
    Node functions
    """
    ####################################################################################################################

    ####################################################################################################################
    # Candidate Generation
    ####################################################################################################################

    def create_prompt(self, input_state: CandidateGenerationState) -> CandidateGenerationState:
        if self.data_set != "LGL":
            toponyms = input_state.toponyms
        else:
            toponyms = self.data_handler.get_short_toponyms_for_article(input_state.article_id)
        prompt = self.working_memory.create_final_prompt()
        formatted_prompt = prompt.format(
            input__heading=input_state.article_title,
            input__news_article=input_state.article_text,
            input__toponym_list=str(toponyms)
        )
        # Replace all "&quot;" with double quotes
        formatted_prompt = formatted_prompt.replace("&quot;", "\"")  # remove when fixed in langchain
        return CandidateGenerationState(
            article_id=input_state.article_id,
            article_title=input_state.article_title,
            article_text=input_state.article_text,
            toponyms=toponyms,
            reflection_phase=ReflectionPhase.INITIAL_ACTOR_GENERATION,
            initial_prompt=formatted_prompt
        )

    def call_actor(self, state: CandidateGenerationState) -> LLMOutput:
        # First check in which state of the reflective candidate generation we are
        if state.reflection_phase == ReflectionPhase.INITIAL_ACTOR_GENERATION:
            prompt = state.initial_prompt
        else:
            prompt = state.reflected_prompt

        @handle_api_errors(call_times=self.call_times)
        def _invoke_llm(input_prompt: str):
            return self.llm.invoke(input_prompt)

        try:
            llm_answer = _invoke_llm(prompt)

            if isinstance(llm_answer, APIStatusError):
                return LLMOutput(
                    article_id=state.article_id,
                    toponyms=state.toponyms,
                    fatal_errors=[Error(
                        execution_step=ExecutionStep.ACTOR,
                        error_message=llm_answer.message
                    )]
                )
            else:
                return LLMOutput(
                    article_id=state.article_id,
                    toponyms=state.toponyms,
                    raw_output=AIMessage(**llm_answer.model_dump())
                )
        except Exception as e:
            return LLMOutput(
                article_id=state.article_id,
                toponyms=state.toponyms,
                fatal_errors=[Error(
                    execution_step=ExecutionStep.ACTOR,
                    error_message=str(e)
                )]
            )

    def extract_generation_output(self, state: LLMOutput) -> LLMOutput:
        parser = OutputParser(article_id=state.article_id,
                              toponym_list=state.toponyms)
        parsed_output = parser.extract_generation_output(state.raw_output)
        # Ensure no duplicate keys by unpacking parsed_output selectively
        parsed_data = parsed_output.model_dump()
        parsed_data.update({
            "article_title": state.article_title,
            "article_text": state.article_text,
        })
        return LLMOutput(**parsed_data)

    def validate_output(self, state: LLMOutput) -> ValidatedOutput:
        return self.validator.validate_toponyms_of_article(state)

    def retrieve_candidates(self, state: ValidatedOutput) -> CandidateGenerationOutput:
        return self.geonames.retrieve_candidates(state)

    def criticize(self, state: CandidateGenerationState) -> CandidateGenerationState:

        @handle_api_errors(call_times=self.call_times)
        def _invoke_llm(input_prompt: str):
            return self.critic_llm.invoke(input_prompt)

        prompt_builder = self.working_memory.long_term_memory

        # compile final critic prompt
        if state.fatal_errors:
            state.reflection_phase = ReflectionPhase.CRITIC_GENERATION_FOR_FATAL_ERRORS
            state.critic_prompt = prompt_builder.generate_critic_prompt_for_fatal_errors(state=state,
                                                                                         fatal_errors=state.fatal_errors)
        else:
            state.reflection_phase = ReflectionPhase.CRITIC_GENERATION_FOR_INVALID_TOPONYMS
            state.critic_prompt = prompt_builder.generate_critic_prompt_for_invalid_toponyms(state)

        try:
            # invoke critic and use feedback to generate reflected actor prompt
            critic_feedback = _invoke_llm(state.critic_prompt)
            if not isinstance(critic_feedback, APIStatusError):
                state.critic_feedback = AIMessage(**critic_feedback.model_dump())
            else:
                state.fatal_errors.append(Error(
                    execution_step=ExecutionStep.CRITIC,
                    error_message=critic_feedback.message
                ))
                return state

            # generate the matching reflected actor prompt
            if state.reflection_phase == ReflectionPhase.CRITIC_GENERATION_FOR_FATAL_ERRORS:
                state.reflection_phase = ReflectionPhase.ACTOR_RETRY_AFTER_FATAL_ERROR
            else:
                state.reflection_phase = ReflectionPhase.ACTOR_RETRY_ON_INVALID_TOPONYMS
            state.reflected_prompt = prompt_builder.generate_reflected_actor_prompt(state)

            # Reset fatal errors for the next iteration of the reflective candidate generation
            state.fatal_errors = []
            return state

        except Exception as e:
            state.fatal_errors.append(Error(
                execution_step=ExecutionStep.CRITIC,
                error_message=str(e)
            ))
            return state


    ####################################################################################################################
    # Candidate Resolution
    ####################################################################################################################

    def create_candidate_resolution_prompt(self, state: GeoCodingState) -> GeoCodingState:
        """
        Resolve the candidates retrieved from the GeoNames API to one correct toponym.
        :param state: The current state of the reflective candidate generation.
        :return: The updated state of the reflective candidate generation.
        """
        prompt_builder = self.working_memory.long_term_memory
        state.resolution_initial_prompt = prompt_builder.generate_candidate_resolution_prompt(state)
        state.reflection_phase = ReflectionPhase.RESOLUTION_ACTOR_GENERATION
        return state

    def call_resolution_actor(self, state: GeoCodingState) -> GeoCodingState:
        """
        Call the resolution actor to resolve the candidates retrieved from the GeoNames API.
        :param state: The current state of the reflective candidate generation.
        :return: The updated state of the reflective candidate generation.
        """

        @handle_api_errors(call_times=self.call_times)
        def _invoke_resolution_actor(input_prompt: str):
            return self.llm.invoke(input_prompt)

        if state.reflection_phase == ReflectionPhase.RESOLUTION_ACTOR_GENERATION:
            prompt = state.resolution_initial_prompt
        else:
            prompt = state.resolution_reflected_prompt

        try:
            resolution_output = _invoke_resolution_actor(prompt)
            if isinstance(resolution_output, APIStatusError):
                state.resolution_fatal_errors.append(Error(
                    execution_step=ExecutionStep.RESOLUTION_ACTOR,
                    error_message=resolution_output.message
                ))
            else:
                state.resolution_raw_output = AIMessage(**resolution_output.model_dump())
        except Exception as e:
            state.resolution_fatal_errors.append(Error(
                execution_step=ExecutionStep.RESOLUTION_ACTOR,
                error_message=str(e)
            ))
        return state

    def extract_resolution_output(self, state: GeoCodingState):
        """
        Extract the resolution output from the LLM response.
        :param state: The current state of the reflective candidate generation.
        :return: The updated state of the reflective candidate generation.
        """
        return self.output_parser.extract_resolution_output(state)

    def validate_resolution_output(self, state: GeoCodingState):
        """
        Validate the resolution output.
        :param state: The current state of the reflective candidate generation.
        :return: The updated state of the reflective candidate generation.
        """
        return self.resolution_validator.validate_resolution(state)

    def criticize_resolution(self, state: GeoCodingState):
        """
        Criticize the resolution output.
        :param state: The current state of the reflective candidate generation.
        :return: The updated state of the reflective candidate generation.
        """
        prompt_builder = self.working_memory.long_term_memory
        if state.resolution_fatal_errors:
            state.reflection_phase = ReflectionPhase.RESOLUTION_CRITIC_GENERATION_FOR_FATAL_ERRORS
            state.resolution_critic_prompt = prompt_builder.generate_resolution_critic_prompt_for_fatal_errors(state)
        else:
            state.reflection_phase = ReflectionPhase.RESOLUTION_CRITIC_GENERATION_FOR_INVALID_RESOLUTIONS
            state.resolution_critic_prompt = prompt_builder.generate_resolution_critic_prompt_for_invalid_toponyms(state)

        @handle_api_errors(call_times=self.call_times)
        def _invoke_llm(input_prompt: str):
            return self.critic_llm.invoke(input_prompt)

        try:
            critic_feedback = _invoke_llm(state.resolution_critic_prompt)
            if not isinstance(critic_feedback, APIStatusError):
                state.resolution_critic_feedback = AIMessage(**critic_feedback.model_dump())
            else:
                state.resolution_fatal_errors.append(Error(
                    execution_step=ExecutionStep.CRITIC,
                    error_message=critic_feedback.message
                ))
                return state

            # generate the matching reflected actor prompt
            if state.reflection_phase == ReflectionPhase.RESOLUTION_CRITIC_GENERATION_FOR_FATAL_ERRORS:
                state.reflection_phase = ReflectionPhase.RESOLUTION_ACTOR_RETRY_AFTER_FATAL_ERROR
            else:
                state.reflection_phase = ReflectionPhase.RESOLUTION_ACTOR_RETRY_ON_INVALID_RESOLUTIONS
            state.resolution_reflected_prompt = prompt_builder.generate_resolution_reflected_actor_prompt(state)

            # Reset fatal errors for the next iteration of the reflective candidate generation
            state.resolution_fatal_errors = []
            return state
        except Exception as e:
            state.resolution_fatal_errors.append(Error(
                execution_step=ExecutionStep.RESOLUTIONCRITIC,
                error_message=str(e)
            ))
            return state


    ####################################################################################################################
    """
    Routing functions
    """
    ####################################################################################################################

    ####################################################################################################################
    # Candidate Generation
    ####################################################################################################################

    @staticmethod
    def has_fatal_errors( state: CandidateGenerationState) -> str:
        if state.fatal_errors:
            if state.reflection_phase == ReflectionPhase.ACTOR_RETRY_AFTER_FATAL_ERROR or \
                    state.reflection_phase == ReflectionPhase.ACTOR_RETRY_ON_INVALID_TOPONYMS:
                return "critique_did_not_solve_fatal_errors"
            else:
                return "has_fatal_errors"
        else:
            return "successful"

    @staticmethod
    def has_invalid_toponyms(state: CandidateGenerationState) -> str:
        if state.fatal_errors:
            if state.reflection_phase == ReflectionPhase.ACTOR_RETRY_AFTER_FATAL_ERROR:
                return "critique_did_not_solve_fatal_errors_or_invalid_toponyms"
            else:
                return "has_fatal_errors_or_invalid_toponyms"
        else:
            if state.invalid_toponyms:
                if state.reflection_phase == ReflectionPhase.ACTOR_RETRY_ON_INVALID_TOPONYMS:
                    return "critique_did_not_solve_fatal_errors_or_invalid_toponyms"
                else:
                    return "has_fatal_errors_or_invalid_toponyms"
            else:
                return "everything_valid"

    @staticmethod
    def critique_was_generated(state: CandidateGenerationState) -> str:
        if state.fatal_errors:
            return "critique_failed"
        else:
            return "critique_generated"

    # Necessary to include as a node function to be able to change the state of the graph
    @staticmethod
    def add_critique_error(
            state: CandidateGenerationState | GeoCodingState) -> CandidateGenerationState | GeoCodingState:
        if isinstance(state, GeoCodingState):
            if state.resolution_fatal_errors:
                state.resolution_fatal_errors.append(Error(
                    execution_step=ExecutionStep.RESOLUTIONCRITIC,
                    error_message="Resolution critique did not solve fatal errors. Exiting."
                ))
                return state

        if state.fatal_errors:
            state.fatal_errors.append(Error(
                execution_step=ExecutionStep.CRITIC,
                error_message="Critique did not solve fatal errors. Exiting."
            ))
        return state

    ####################################################################################################################
    # Candidate Resolution
    ####################################################################################################################

    @staticmethod
    def has_fatal_errors_resolution(state: GeoCodingState) -> str:
        if state.resolution_fatal_errors:
            if state.reflection_phase == ReflectionPhase.RESOLUTION_ACTOR_RETRY_AFTER_FATAL_ERROR or \
                    state.reflection_phase == ReflectionPhase.RESOLUTION_ACTOR_RETRY_ON_INVALID_RESOLUTIONS:
                return "critique_did_not_solve_fatal_errors"
            else:
                return "has_fatal_errors"
        else:
            return "successful"

    @staticmethod
    def has_invalid_resolutions(state: GeoCodingState) -> str:
        if state.resolution_fatal_errors:
            if state.reflection_phase == ReflectionPhase.RESOLUTION_ACTOR_RETRY_AFTER_FATAL_ERROR:
                return "critique_did_not_solve_fatal_errors_or_invalid_resolutions"
            else:
                return "has_fatal_errors_or_invalid_resolutions"
        else:
            if state.invalid_geocoded_toponyms:
                if state.reflection_phase == ReflectionPhase.RESOLUTION_ACTOR_RETRY_ON_INVALID_RESOLUTIONS:
                    return "critique_did_not_solve_fatal_errors_or_invalid_resolutions"
                else:
                    return "has_fatal_errors_or_invalid_resolutions"
            else:
                return "everything_valid"

    @staticmethod
    def critique_was_generated_resolution(state: GeoCodingState) -> str:
        if state.resolution_fatal_errors:
            return "critique_failed"
        else:
            return "critique_generated"


    ####################################################################################################################
    """
    Grap Structure
    """
    ####################################################################################################################

    def build_graph(self):
        graph_builder = StateGraph(CandidateGenerationState)

        # Add nodes to the graph
        graph_builder.add_node("create_prompt", self.create_prompt)
        graph_builder.add_node("call_actor", self.call_actor)
        graph_builder.add_node("extract_output", self.extract_generation_output)
        graph_builder.add_node("validate_output", self.validate_output)
        graph_builder.add_node("retrieve_candidates", self.retrieve_candidates)
        graph_builder.add_node("criticize", self.criticize)
        graph_builder.add_node("add_critique_error", self.add_critique_error)


        # Set the entry point and define edges
        graph_builder.set_entry_point("create_prompt")
        graph_builder.add_edge("create_prompt", "call_actor")
        graph_builder.add_conditional_edges("call_actor",
                                            self.has_fatal_errors,
                                            {
                                                "has_fatal_errors": "criticize",
                                                "successful": "extract_output",
                                                "critique_did_not_solve_fatal_errors": "add_critique_error"
                                            })
        graph_builder.add_conditional_edges("extract_output",
                                            self.has_fatal_errors,
                                            {
                                                "has_fatal_errors": "criticize",
                                                "successful": "validate_output",
                                                "critique_did_not_solve_fatal_errors": "add_critique_error"
                                            })
        graph_builder.add_conditional_edges("validate_output",
                                            self.has_fatal_errors,
                                            {
                                                "has_fatal_errors": "criticize",
                                                "successful": "retrieve_candidates",
                                                "critique_did_not_solve_fatal_errors": "add_critique_error"
                                            })
        graph_builder.add_conditional_edges("retrieve_candidates",
                                            self.has_invalid_toponyms,
                                            {
                                                "has_fatal_errors_or_invalid_toponyms": "criticize",
                                                "everything_valid": END,
                                                "critique_did_not_solve_fatal_errors_or_invalid_toponyms": "add_critique_error"
                                            })
        graph_builder.add_conditional_edges("criticize",
                                            self.critique_was_generated,
                                            {
                                                "critique_generated": "call_actor",
                                                "critique_failed": "add_critique_error"
                                            })
        graph_builder.add_edge("add_critique_error", END)
        return graph_builder

    def build_resolution_graph(self):
        graph_builder = StateGraph(GeoCodingState)

        # Add nodes to the graph
        graph_builder.add_node("create_candidate_resolution_prompt", self.create_candidate_resolution_prompt)
        graph_builder.add_node("call_resolution_actor", self.call_resolution_actor)
        graph_builder.add_node("extract_resolution_output", self.extract_resolution_output)
        graph_builder.add_node("validate_resolution_output", self.validate_resolution_output)
        graph_builder.add_node("criticize_resolution", self.criticize_resolution)
        graph_builder.add_node("add_critique_error", self.add_critique_error)

        # Set the entry point and define edges
        graph_builder.set_entry_point("create_candidate_resolution_prompt")
        graph_builder.add_edge("create_candidate_resolution_prompt", "call_resolution_actor")
        graph_builder.add_conditional_edges("call_resolution_actor",
                                            self.has_fatal_errors_resolution,
                                            {
                                                "has_fatal_errors": "criticize_resolution",
                                                "successful": "extract_resolution_output",
                                                "critique_did_not_solve_fatal_errors": "add_critique_error"
                                            })
        graph_builder.add_conditional_edges("extract_resolution_output",
                                            self.has_fatal_errors_resolution,
                                            {
                                                "has_fatal_errors": "criticize_resolution",
                                                "successful": "validate_resolution_output",
                                                "critique_did_not_solve_fatal_errors": "add_critique_error"
                                            })
        graph_builder.add_conditional_edges("validate_resolution_output",
                                            self.has_invalid_resolutions,
                                            {
                                                "has_fatal_errors_or_invalid_resolutions": "criticize_resolution",
                                                "everything_valid": END,
                                                "critique_did_not_solve_fatal_errors_or_invalid_resolutions": "add_critique_error"
                                            })
        graph_builder.add_conditional_edges("criticize_resolution",
                                            self.critique_was_generated_resolution,
                                            {
                                                "critique_generated": "call_resolution_actor",
                                                "critique_failed": "add_critique_error"
                                            })
        graph_builder.add_edge("add_critique_error", END)
        return graph_builder


    ####################################################################################################################
    """
    Compile and run the graph for entire datasets
    """
    ####################################################################################################################

    def run_generation_graph(self, article: pd.Series | dict):
        graph_builder = self.build_graph()
        agent_graph = graph_builder.compile()

        if self.data_set == "GeoCoDe":
            input_state = {
                "article_id": article['article_id'],
                "article_title": "Not provided",
                "article_text": article['article_text'],
                "toponyms": article['toponyms']
            }
        else:
            input_state = {
                "article_id": article['docid'],
                "article_title": article['title'],
                "article_text": article['text']
            }

        # Invoke the graph
        agent_graph_answer = agent_graph.invoke(input_state)

        return agent_graph_answer

    def reflectively_generate_candidates_for_evaluation(self,
                                                        seed: int = 42,
                                                        nof_articles: int = 100,
                                                        output_dir: str = 'output/'):
        def _modify_geocode_article(article):
            # Find the position of " is a " in the article_text
            split_marker = " is a "
            if split_marker in article["article_text"]:
                split_index = article["article_text"].index(split_marker)

                # Extract the article_about part
                article["article_about"] = article["article_text"][:split_index]

                # Remove the article_about part from article_text
                article["article_text"] = article["article_text"][split_index + len(split_marker):]

            return article

        def _generate_and_save_candidates_for_article(article):
            agent_graph_answer = self.run_generation_graph(article)

            # Parse dictionary (default langgraph output) to CandidateGenerationState
            agent_graph_answer = CandidateGenerationState(**agent_graph_answer)

            with open(os.path.join(output_dir, f'{article_id}.pkl'), 'wb') as f:
                # noinspection PyTypeChecker
                pickle.dump(agent_graph_answer, f)


        os.makedirs(output_dir, exist_ok=True)

        if self.data_set == "GeoCoDe":
            with open("data/processed_GeoCoDe_test.json", "r", encoding="utf-8") as f:
                articles = json.load(f)

            # Modify the articles to exclude the explicit toponym the article is about from the text
            articles = [_modify_geocode_article(article) for article in articles if " is a " in article["article_text"]]

            # Limit the number of articles to process
            articles = articles[:nof_articles]

            for article in tqdm(articles, total=len(articles), desc="Processing articles"):
                article_id = article['article_id'].split()[0]

                # Skip processing if the article already exists
                if os.path.exists(os.path.join(output_dir, f'{article_id}.pkl')):
                    continue

                _generate_and_save_candidates_for_article(article)

        else:
            articles_df = self.data_handler.get_random_articles_for_evaluation(
                seed=seed, n=nof_articles
            )
            for index, article in tqdm(articles_df.iterrows(), total=len(articles_df), desc="Processing articles"):
                # First check if the article has already been processed
                article_id = article['docid']

                if os.path.exists(os.path.join(output_dir, f'{article_id}.pkl')):
                    continue

                _generate_and_save_candidates_for_article(article)


    def generate_graph_image(self, output_file_path: str = "graph_layout_image.png", candidate_resolution: bool = False):
        if candidate_resolution:
            graph_builder = self.build_resolution_graph()
        else:
            graph_builder = self.build_graph()
        agent_graph = graph_builder.compile()
        image_data = agent_graph.get_graph().draw_mermaid_png(output_file_path=output_file_path)
        # Open the image
        os.system(output_file_path)
        return image_data

    def run_resolution_graph(self,
                             input_directory: str = "",
                             output_directory: str = ""):
        if not os.path.exists(input_directory):
            raise FileNotFoundError(f"Input directory {input_directory} does not exist.")
        evaluator = CandidateGenerationEvaluator(output_directory=input_directory,
                                                 data_directory='data/')
        os.makedirs(output_directory, exist_ok=True)
        for candidate_file in tqdm(os.listdir(input_directory), desc="Processing candidate files"):
            if not candidate_file.endswith(".pkl"):
                continue
            article_id = candidate_file.replace(".pkl", "")
            if os.path.exists(os.path.join(output_directory, f'{article_id}.pkl')):
                continue
            candidate_generation = evaluator.load_generation_for_article(article_id)
            candidate_generation_dict = preprocess_data(candidate_generation.model_dump(), GeoCodingState)
            graph_builder = self.build_resolution_graph()
            agent_graph = graph_builder.compile()
            agent_graph_answer = agent_graph.invoke(candidate_generation_dict)
            # Parse dictionary (default langgraph output) to GeoCodingState
            agent_graph_answer = GeoCodingState(**agent_graph_answer)
            with open(os.path.join(output_directory, f'{article_id}.pkl'), 'wb') as f:
                # noinspection PyTypeChecker
                pickle.dump(agent_graph_answer, f)
