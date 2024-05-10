
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.callbacks import StreamingStdOutCallbackHandler

#temperature 란?
#사용할 샘플링 온도는 0에서 2 사이입니다. 0.8과 같이 값이 높을수록 출력이 더 무작위로 만들어지고, 0.2와 같이 값이 낮을수록 더 집중적이고 결정적이게 됩니다.
chat = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
    )

chef_prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 월드 클래스 국제 쉐프 입니다. 당신은 어떤 종류의 요리든 쉽게 구할 수 있는 재료로 따라하기 쉬운 레시피를 전달합니다."),
    ("human", "난 {cook} 요리를 원해"),
])

chef_chain = chef_prompt | chat

veg_chef_prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 채식주의자를 위한 쉐프 입니다. 전통적인 채식주의자용 레시피에 특화되어 있습니다. 레시피를 입력받아 그걸 채식주의자 레시피로 변환해서 전달합니다. 기존 레시피를 너무 많이 변경해선 안됩니다. 만약 대체품이 없다면 그냥 레시피를 모른다고 말하세요."),
    ("human", "{recipe}"),
])

veg_chain = veg_chef_prompt | chat

final_chain = {"recipe": chef_chain} | veg_chain

result = final_chain.invoke({
    "cook": "치킨 커리"
})

#print(result)