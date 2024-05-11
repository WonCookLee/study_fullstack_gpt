from langchain.chat_models import ChatOpenAI
from langchain.llms.openai import OpenAI
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder

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
    memory_key="chat_history",
    return_messages=True
    )

prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 사람에게 도움을 주는 AI야, 물어본 언어로 대답해"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

template = """
    너는 사람에게 도움을 주는 AI야, 물어본 언어로 대답해
    
    {chat_history}
    사람: {question}
    AI: 
"""
chain = LLMChain(
    llm=llm,
    memory=memory,
    prompt=prompt,
    verbose=True
)

chain.predict(question="내 이름은 원국")
chain.predict(question="난 강남에 살아")
chain.predict(question="자곡동 알아 ? 강남같지 않아 시골 같아")
chain.predict(question="내 이름 기억하고 있니?")

