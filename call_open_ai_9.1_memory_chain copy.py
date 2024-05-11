from langchain.chat_models import ChatOpenAI
from langchain.llms.openai import OpenAI
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

llm = ChatOpenAI(
    temperature=0.1,
    # streaming=True,
    # callbacks=[
    #     StreamingStdOutCallbackHandler(),
    # ],
)

memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=120,
    memory_key="chat_history"
    )

template = """
    너는 사람에게 도움을 주는 AI야, 물어본 언어로 대답해
    
    {chat_history}
    사람: {question}
    AI: 
"""
chain = LLMChain(
    llm=llm,
    memory=memory,
    prompt=PromptTemplate.from_template(template),
    verbose=True
)

chain.predict(question="내 이름은 원국")


chain.predict(question="난 강남에 살아")

#이름 기억하는지 테스트
chain.predict(question="내 이름은?")