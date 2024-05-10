from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage, SystemMessage

from langchain.schema import BaseOutputParser

class CommaOutputParser(BaseOutputParser):
    def parse(self, text):
        items = text.split(",")
        return list(map(str.strip, items))
        

#temperature 란?
#사용할 샘플링 온도는 0에서 2 사이입니다. 0.8과 같이 값이 높을수록 출력이 더 무작위로 만들어지고, 0.2와 같이 값이 낮을수록 더 집중적이고 결정적이게 됩니다.
chat = ChatOpenAI(
    temperature=0.1 
    )

template = ChatPromptTemplate.from_messages([
    ("system",
    "you are a list generating machine. Everything you are asked will be answered with a comma separated of max {max_items}. Do Not reply with anything else."),
    ("human", "{question}"),
])

chain = template | chat | CommaOutputParser()

result = chain.invoke({
    "max_items":"10",
    "question":"What are you pokenmons?",
})

print(result)