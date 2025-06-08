import os

import httpx
from dotenv import load_dotenv
import json
from pandas.io.common import file_exists

import requests
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate


# API configuration
BASE_URL = "https://chat-ai.academiccloud.de/v1"
load_dotenv()

class ChatAIHandler:
    def __init__(self, base_url=BASE_URL):
        self.api_key = os.getenv('SAIA_API_KEY')
        self.base_url = base_url
        self.available_models = self._load_available_models()

    def _load_available_models(self):
        if file_exists("available_models.json"):
            with open("available_models.json", "r") as f:
                return json.load(f)
        else:
            return self.list_all_models()

    def list_all_models(self):
        """
        :return: list: returns all available model keys
        """
        url = self.base_url + "/models"
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }

        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            models = response.json()
            all_models = [model_key["id"] for model_key in models["data"]]
            with open("available_models.json", "w") as f:
                json.dump(all_models, f)
            print("Available models are ", all_models)
            return all_models
        else:
            response.raise_for_status()

    def get_model(self, model_name, temperature=0.0) -> ChatOpenAI:
        """
        Get ChatAI model if available
        :param model_name: str: model name according to the GDWG ChatAI API
        :param temperature: float
        :return: ChatOpenAI: returns the LangChain LLM model
        """
        if model_name not in self.available_models:
            raise ValueError(f"Model {model_name} is not available. \n"
                             f"Please choose from the following models: {self.available_models}")
        else:
            return ChatOpenAI(api_key=self.api_key,
                              base_url=self.base_url,
                              max_retries=1,
                              timeout=httpx.Timeout(500.0),
                              model_name=temperature,
                              temperature=0)
                              #top_p=top_p)

    def get_embedding_model(self, model_name="e5-mistral-7b-instruct") -> OpenAIEmbeddings:
        """
        Get ChatAI embedding model
        :param model_name: str: model name according to the GDWG ChatAI API
        :return: ChatOpenAI: returns the LangChain embedding model
        """
        return OpenAIEmbeddings(api_key=self.api_key,
                                base_url=self.base_url,
                                model=model_name)

    def call_llm(self, model: ChatOpenAI, question: str) -> str:
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a helpful assistant which is very confident about it's geographical knowledge. "
             "Since you were trained on coordinates, you can provide the coordinates strictly in JSON format "
             "in response to a location specified to you!"),
            ("user", "{question}")
        ])

        chain = prompt | model
        llm_answer = chain.invoke(question)
        return llm_answer.content