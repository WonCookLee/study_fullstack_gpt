from typing import Any, Dict, List
from langchain.chat_models import ChatOpenAI
from langchain.prompts import example_selector
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts.example_selector.base import BaseExampleSelector
from langchain.prompts.example_selector import LengthBasedExampleSelector
from langchain.prompts import load_prompt

chat = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[
        StreamingStdOutCallbackHandler(),
    ],
)

prompt = load_prompt(path="./prompt.yaml")
#prompt = load_prompt(path="./prompt.json")
prompt.format(country="한국")

