import os
import subprocess

from langchain_core.prompts import PipelinePromptTemplate, PromptTemplate

from agent_components.memory.episodic_memory import EpisodicMemory
from agent_components.memory.long_term_memory import LongTermMemory


class WorkingMemory:
    def __init__(self, skip_few_shot_loader: bool = False):
        self.few_shot_handler = EpisodicMemory(data_directory=os.path.join(
            subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).strip().decode(), 'data/'),
                                               xml_file='LGL_test.xml',
                                               skip_few_shot_loader=skip_few_shot_loader)
        self.long_term_memory = LongTermMemory()

    def create_final_prompt(self) -> PipelinePromptTemplate:
        final_template = "{system_instructions}\n{task_instructions}\n{documentation}\n{few_shot_examples}"
        final_prompt = PromptTemplate.from_template(final_template)
        pipeline_prompts = [
            ("system_instructions", self.long_term_memory.system_instructions_prompt),
            ("task_instructions", self.long_term_memory.task_instructions_prompt),
            ("documentation", self.long_term_memory.documentation_prompt),
            ("few_shot_examples", self.few_shot_handler.few_shot_template)
        ]
        return PipelinePromptTemplate(
            final_prompt=final_prompt,
            pipeline_prompts=pipeline_prompts,
            input_variables=self.few_shot_handler.few_shot_template.input_variables.extend(
                self.long_term_memory.task_instructions_prompt.input_variables
            )
        )
