from enum import Enum

from pydantic import BaseModel, Field


class ExecutionStep(Enum):
    SEARCHOUTPUTPARSER = "search_output_parser"
    SEARCHPARAMETERVALIDATOR = "search_parameter_validator"
    ARTICLESYNTAXVALIDATOR = "article_syntax_validator"
    GEOAPI = "geonames_search_api"
    ACTOR = "actor_llm_generation"
    CRITIC = "critic_llm_generation"
    RESOLUTION_ACTOR = "resolution_actor"
    RESOLUTIONSYNTAXVALIDATOR = "resolution_syntax_validator"
    RESOLUTIONCRITIC = "resolution_critic"

class Error(BaseModel):
    execution_step: ExecutionStep | str = Field(description="The step in which the error occurred")
    error_message: str = Field(description="The error message")