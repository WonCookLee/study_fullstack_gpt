from langchain.chat_models import ChatOpenAI
from langchain.llms.openai import OpenAI
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema.runnable import RunnablePassthrough
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
    return_messages=True
    )

prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 사람에게 도움을 주는 AI야, 물어본 언어로 대답해"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

def load_memory(_):
    return memory.load_memory_variables({})["history"]

chain = RunnablePassthrough.assign(history=load_memory) | prompt | llm

def invoke_chain(question):
    result = chain.invoke({"question": question})
    memory.save_context(
        {"input": question},
        {"output": result.content},
    )
    print(result)

invoke_chain("내 이름은 원국")
invoke_chain("난 강남에 살아")
invoke_chain(question="자곡동 알아 ? 강남같지 않아 시골 같아")
invoke_chain(question="내 이름 기억하고 있니? 어디살아?")