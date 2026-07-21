import os
import subprocess

from agent_components.memory.episodic_memory import EpisodicMemory
from agent_components.memory.long_term_memory import LongTermMemory


class _PipelinePromptAdapter:
    """Replacement for the removed langchain_core PipelinePromptTemplate.

    Pre-formats static sub-prompts at construction time and delegates
    dynamic formatting to the few-shot template.
    """
    def __init__(self, static_prefix: str, few_shot_template):
        self._prefix = static_prefix
        self._few_shot = few_shot_template

    def format(self, **kwargs) -> str:
        return self._prefix + "\n" + self._few_shot.format(**kwargs)


class WorkingMemory:
    def __init__(self, skip_few_shot_loader: bool = False):
        self.few_shot_handler = EpisodicMemory(data_directory=os.path.join(
            subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).strip().decode(), 'data/'),
                                               xml_file='LGL_test.xml',
                                               skip_few_shot_loader=skip_few_shot_loader)
        self.long_term_memory = LongTermMemory()

    def create_final_prompt(self) -> _PipelinePromptAdapter:
        static_prefix = "\n".join([
            self.long_term_memory.system_instructions_prompt.format(),
            self.long_term_memory.task_instructions_prompt.format(),
            self.long_term_memory.documentation_prompt.format(),
        ])
        return _PipelinePromptAdapter(static_prefix, self.few_shot_handler.few_shot_template)
