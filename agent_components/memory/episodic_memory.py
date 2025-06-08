from typing import List, Dict
import subprocess
import os

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.few_shot_with_templates import FewShotPromptWithTemplates
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import JSONLoader
from data_handler.xml_parsing import XMLDataHandler

class EpisodicMemory:
    FEW_SHOT_EXAMPLE_PATH = os.path.join(subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).strip().decode(), 'data/few_shot_examples_generation.json')

    def __init__(self, data_directory: str, xml_file: str, skip_few_shot_loader: bool = False):
        self.data_handler = XMLDataHandler(data_directory)
        self.data_handler.parse_xml(xml_file)
        self.examples = self._load_examples()
        if skip_few_shot_loader:
            self.few_shot_template = None
        else:
            self.few_shot_template = self.create_few_shot_template()

    def _load_ground_truth(self, few_shot_example_path: str = FEW_SHOT_EXAMPLE_PATH) -> List[Document]:

        def _metadata_func(record: Dict, metadata: dict) -> dict:
            metadata['docid'] = record.get('docid')
            return metadata

        loader = JSONLoader(
            file_path=few_shot_example_path,
            jq_schema='.[]',
            content_key='ground_truth',
            metadata_func=_metadata_func,
            text_content=False
        )
        fs_examples = loader.load()
        return fs_examples

    def _load_examples(self) -> List[Dict]:
        examples = []
        for gt_example in self._load_ground_truth():
            _id = gt_example.metadata['docid']
            _article = self.data_handler.get_article_for_prompting(_id)
            toponym_list = self.data_handler.get_short_toponyms_for_article(_id)

            # Assuming _article is a DataFrame
            example = {
                "input__heading": _article.get('title'),
                "input__news_article": _article.get('text'),
                "input__toponym_list": str(toponym_list),
                "output": gt_example.page_content
            }
            examples.append(example)
        return examples

    def create_example_selector(self, k: int = 2) -> SemanticSimilarityExampleSelector:
        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        return SemanticSimilarityExampleSelector.from_examples(
            examples=self.examples,
            embeddings=embeddings,  #could also use new embedding model by ChatAI
            vectorstore_cls=Chroma,
            k=k
        )

    def create_few_shot_template(self) -> FewShotPromptWithTemplates:
        example_selector = self.create_example_selector()
        example_template =  ("Input:\n"
                            "News Article Heading: {{input__heading}}\n"
                            "News Article: {{input__news_article}}\n"
                            "Toponym List: {{input__toponym_list}}\n\n"                      
                            "Output:\n")

        example_prompt = PromptTemplate(
            input_variables=["input__heading", "input__news_article", "input__toponym_list", "output"],
            template=example_template + "{{output}}",
            template_format="mustache"  # Requires langchain-core 0.3.15, langchain 0.3.7 bc later, mustache wrongly handles double quotes
        )

        prefix = PromptTemplate.from_template(
            "Here are a few examples on how to provide the search arguments for a given news article:"
        )

        suffix = PromptTemplate(
            input_variables=["input__heading", "input__news_article", "input__toponym_list"],
            template="\n\nNow consider all that you've been instructed on and provide the search arguments for this news article strictly in JSON without any additional text:\n\n" + example_template,
            template_format="mustache"
        )
        return FewShotPromptWithTemplates(
            example_selector=example_selector,
            example_prompt=example_prompt,
            prefix=prefix,
            suffix=suffix,
            example_separator="\n\n-----\n",
            input_variables=["input__heading", "input__news_article", "input__toponym_list"],
            template_format="mustache"
        )
