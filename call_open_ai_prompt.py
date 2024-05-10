from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage, SystemMessage

#temperature 란?
#사용할 샘플링 온도는 0에서 2 사이입니다. 0.8과 같이 값이 높을수록 출력이 더 무작위로 만들어지고, 0.2와 같이 값이 낮을수록 더 집중적이고 결정적이게 됩니다.
chat = ChatOpenAI(
    temperature=0.1 
    )

#temperature 란?
#사용할 샘플링 온도는 0에서 2 사이입니다. 0.8과 같이 값이 높을수록 출력이 더 무작위로 만들어지고, 0.2와 같이 값이 낮을수록 더 집중적이고 결정적이게 됩니다.
chat = ChatOpenAI(
    temperature=0.1 
    )

template = ChatPromptTemplate.from_messages([
    ("system", "you are a geography expert. And you only reply in {language}."),
    ("ai", "Ciao, mi chiamo {name}!"),
    ("human", "What is the distance between {country_a} and {country_b}. Also, what is your name?"),
])

prompt = template.format_messages(
    language="Korean",
    name = "Socrates",
    country_a = "Mexico",
    country_b = "Thailand",
    )

message = chat.predict_messages(prompt)
print(message)