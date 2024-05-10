from langchain.llms.openai import OpenAI
from langchain.chat_models import ChatOpenAI

llm = OpenAI(model_name="gpt-3.5-turbo-1106")
chat = ChatOpenAI(model_name="gpt-4-turbo")

a = llm.predict("open ai api 자바로 호출하는 코드 알아?")
b = chat.predict("open ai api 자바로 호출하는 코드 알아?")

print("gpt-3.5-turbo-1106 : ", a)
print("gpt-4-turbo : ", b)
