from langchain.chat_models import ChatOpenAI
from langchain.llms.openai import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferWindowMemory

# 대화 전체를 저장하는 메모리
# 대화 길수록 비효율
memory = ConversationBufferMemory(return_messages=True)
memory.save_context({"input":"hi"}, {"output":"How are you"})
memory.load_memory_variables({})


#최근 대화를 저장하는 메모리
#4개까지만 저장
memory = ConversationBufferWindowMemory(
    return_messages=True,
    k=4
)

def add_message(input, output):
    memory.save_context({"input":input}, {"output":output})

add_message("테스트인풋1", "테스트아웃풋1")
add_message("테스트인풋2", "테스트아웃풋2")
add_message("테스트인풋3", "테스트아웃풋3")
add_message("테스트인풋4", "테스트아웃풋4")
add_message("테스트인풋5", "테스트아웃풋5")
add_message("테스트인풋6", "테스트아웃풋6")

memory.load_memory_variables({})


from langchain.memory import ConversationSummaryMemory
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[
        StreamingStdOutCallbackHandler(),
    ],
)

#대화 내용 요약해서 저장하는 메모리
#대화가 짧을수록 손해, 길수록 유리
memory = ConversationSummaryMemory(llm=llm)

def add_message(input, output):
    memory.save_context({"input":input}, {"output":output})
    
def get_history():
    return memory.load_memory_variables({})
    
add_message("난 이원국이고, 대한민국 서울에 살아", "와우! 멋져!")
add_message("한국은 매우 아름다워", "나도 가보고 싶어")


from langchain.memory import ConversationSummaryBufferMemory

#특정 limit 넘으면 요약하기 시작
memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=150,
    return_messages=True
    )

def add_message(input, output):
    memory.save_context({"input":input}, {"output":output})
    
def get_history():
    return memory.load_memory_variables({})

add_message("난 이원국이고, 대한민국 서울에 살아", "와우! 멋져!")
add_message("한국은 매우 아름다워", "나도 가보고 싶어")
add_message("난 치즈라는 고양이를 키워 고양이에 대해 설명해주고 수명도 알려줘 우리집 고양이는 14년 살았어 얼마나 남았을까?", """치즈라는 고양이를 키우고 계시군요! 고양이는 사랑스럽고 호기심 많은 동물로, 많은 사람들이 그들과 함께 삶을 나누고 있습니다. 고양이의 수명은 여러 요소에 따라 달라질 수 있습니다. 평균적으로 실외에서 살면 10~15년, 반면에 실내에서 살면 15년 이상의 수명을 가질 수 있습니다.

14년이라는 나이로 보면 이미 치즈라는 고양이는 상당히 잘 살고 계시는 것 같아요! 고양이의 건강과 행복을 유지하기 위해 꾸준한 동물 병원 방문과 건강한 식사, 적절한 운동과 함께 정서적으로도 케어를 해주시는 것이 중요합니다. 현재 치즈라는 고양이가 얼마나 더 살 수 있을지 정확히 알기는 어렵지만, 건강하고 행복한 일년이 많이 남았으면 좋겠네요!""")

get_history()

from langchain.memory import ConversationKGMemory
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(temperature=0.1)
#중요한거만 뽑음, 대화에서 entity 추출
memory = ConversationKGMemory(
    llm=llm,
    return_messages=True,
)


def add_message(input, output):
    memory.save_context({"input": input}, {"output": output})


add_message("Hi I'm Nicolas, I live in South Korea", "Wow that is so cool!")

memory.load_memory_variables({"input": "who is Nicolas"})
add_message("Nicolas likes kimchi", "Wow that is so cool!")
memory.load_memory_variables({"inputs": "what does nicolas like"})

#{'history': [SystemMessage(content='On Nicolas: Nicolas lives in South Korea. Nicolas likes kimchi.')]}


