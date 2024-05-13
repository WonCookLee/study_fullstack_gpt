from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseOutputParser
import json


class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)


output_parser = JsonOutputParser()


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-3.5-turbo-0125",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)

questions_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            당신은 교사 역할을 하는 유용한 보조자입니다.

            다음 문맥에 기초하여 10개의 질문을 만들어 텍스트에 대한 사용자의 지식을 테스트하십시오.
            각 질문에는 4개의 답이 있어야 하며, 그 중 3개는 틀리고 1개는 맞아야 합니다.

            (o)를 사용하여 정답을 표시하세요.

            질문 예:

            질문: 바다의 색깔은 무엇입니까?
            답: 빨간색|노란색|녹색|파란색(o)

            질문: 수도 또는 조지아는 무엇입니까?
            답변: Baku|Tbilisi(o)|Manila|베이루트

            질문: 아바타는 언제 출시되었나요?
            답변: 2007|2001|2009(o)|1998

            질문: 율리우스 카이사르는 누구였나요?
            답변: 로마 황제(o)|화가|배우|모델

            네 차례야!

            Context: {context}
            """,
        )
    ]
)

formatting_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a powerful formatting algorithm.

    You format exam questions into JSON format.
    Answers with (o) are the correct ones.

    Example Input:

    질문: 바다의 색깔은 무엇입니까?
    답: 빨간색|노란색|녹색|파란색(o)

    질문: 수도 또는 조지아는 무엇입니까?
    답변: Baku|Tbilisi(o)|Manila|베이루트

    질문: 아바타는 언제 출시되었나요?
    답변: 2007|2001|2009(o)|1998

    질문: 율리우스 카이사르는 누구였나요?
    답변: 로마 황제(o)|화가|배우|모델


    Example Output:

    ```json
    {{ "questions": [
            {{
                "question": "바다의 색깔은 무엇입니까?",
                "key" : 1,
                "answers": [
                        {{
                            "answer": "빨간색",
                            "correct": false
                        }},
                        {{
                            "answer": "노란색",
                            "correct": false
                        }},
                        {{
                            "answer": "녹색",
                            "correct": false
                        }},
                        {{
                            "answer": "파란색",
                            "correct": true
                        }},
                ]
            }},
                        {{
                "question": "수도 또는 조지아는 무엇입니까?",
                "key" : 2,
                "answers": [
                        {{
                            "answer": "Baku",
                            "correct": false
                        }},
                        {{
                            "answer": "Tbilisi",
                            "correct": true
                        }},
                        {{
                            "answer": "Manila",
                            "correct": false
                        }},
                        {{
                            "answer": "Beirut",
                            "correct": false
                        }},
                ]
            }},
                        {{
                "question": "아바타는 언제 출시되었나요?",
                "key" : 3,
                "answers": [
                        {{
                            "answer": "2007",
                            "correct": false
                        }},
                        {{
                            "answer": "2001",
                            "correct": false
                        }},
                        {{
                            "answer": "2009",
                            "correct": true
                        }},
                        {{
                            "answer": "1998",
                            "correct": false
                        }},
                ]
            }},
            {{
                "question": "율리우스 카이사르는 누구였나요?",
                "key" : 4,
                "answers": [
                        {{
                            "answer": "로마 황제",
                            "correct": true
                        }},
                        {{
                            "answer": "화가",
                            "correct": false
                        }},
                        {{
                            "answer": "배우",
                            "correct": false
                        }},
                        {{
                            "answer": "모델",
                            "correct": false
                        }},
                ]
            }}
        ]
    }}
    ```
    Your turn!

    Questions: {context}

""",
        )
    ]
)

questions_chain = {"context": format_docs} | questions_prompt | llm

formatting_chain = formatting_prompt | llm

st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("QuizGPT")


@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, topic):
    chain = {"context": questions_chain} | formatting_chain | output_parser
    return chain.invoke(_docs)


@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5, lang="ko")
    docs = retriever.get_relevant_documents(term)
    return docs


with st.sidebar:
    docs = None
    choice = st.selectbox(
        "Choose what you want to use.",
        (
            "File",
            "Wikipedia Article",
        ),
    )
    if choice == "File":
        file = st.file_uploader(
            "Upload a .docx , .txt or .pdf file",
            type=["pdf", "txt", "docx"],
        )
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("Search Wikipedia...")
        if topic:
            docs = wiki_search(topic)
            st.write(docs)

if not docs:
    st.markdown(
        """
    QuizGPT에 오신 것을 환영합니다.

    여러분의 지식을 테스트하고 공부에 도움이 되도록 Wikipedia 기사나 여러분이 업로드한 파일로 퀴즈를 만들겠습니다.

    파일을 업로드하거나 사이드바에서 Wikipedia를 검색하여 시작하세요.
    """
    )
else:

    response = run_quiz_chain(docs, topic if topic else file.name)
    with st.form("questions_form"):
        for idx, question in enumerate(response["questions"]):
            st.write(question["question"])
            value = st.radio(
                "Select an option.",
                [answer["answer"] for answer in question["answers"]],
                index=None,
                key=f"{idx}_radio"
            )
            if {"answer": value, "correct": True} in question["answers"]:
                st.success("Correct!")
            elif value is not None:
                st.error("Wrong!")
        button = st.form_submit_button()
