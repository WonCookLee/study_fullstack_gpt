from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


llm = ChatOpenAI(temperature=0.1)

# loader = TextLoader("./rag/file/현진건-운수_좋은_날+B3356-개벽.txt", encoding="utf-8")
# loader = PyPDFLoader("./rag/file/현진건-운수_좋은_날+B3356-개벽.pdf")
#UnstructuredFileLoader 만능 로더
loader = UnstructuredFileLoader("./rag/file/현진건-운수_좋은_날+B3356-개벽.pdf")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=600, #분단 사이즈
    chunk_overlap=100, #문단 겹치는 정도
    separators="\n\n", #분할자
)

result = loader.load_and_split(text_splitter=splitter)

print(result)
