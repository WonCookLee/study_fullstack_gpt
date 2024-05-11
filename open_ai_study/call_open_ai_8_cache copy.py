from typing import Any, Dict, List
from langchain.chat_models import ChatOpenAI
from langchain.prompts import example_selector
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts.example_selector.base import BaseExampleSelector
from langchain.prompts.example_selector import LengthBasedExampleSelector
from langchain.prompts import load_prompt
from langchain.prompts.pipeline import PipelinePromptTemplate
from langchain.cache import InMemoryCache, SQLiteCache
from langchain.globals import set_llm_cache, set_debug
from langchain.callbacks import get_openai_callback

#set_llm_cache(InMemoryCache())
set_llm_cache(SQLiteCache("cash.db"))
set_debug(True)

chat = ChatOpenAI(
    temperature=0.1,
    # streaming=True,
    # callbacks=[
    #     StreamingStdOutCallbackHandler(),
    # ],
)
chat.predict("이탈리안 파스타 어떻게 만들어?")

