from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.callbacks import StreamingStdOutCallbackHandler

chat = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[
        StreamingStdOutCallbackHandler(),
    ],
)


with get_openai_callback() as usage:
    a = chat.predict("What is the recipe for soju")
    b = chat.predict("What is the recipe for bread")
    print(a, "\n")
    print(b, "\n")
    print(usage)