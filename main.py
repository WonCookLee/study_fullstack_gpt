from langchain.llms.openai import OpenAI
from langchain.chat_models import ChatOpenAI

llm = OpenAI(model_name="gpt-3.5-turbo")
chat = ChatOpenAI()

a = llm.predict("How many planets art there?")
b = chat.predict("How many planets art there?")

a, b

print(a,b)