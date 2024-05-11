from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.storage import LocalFileStore
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS


llm = ChatOpenAI(
    temperature=0.1,
)


cache_dir = LocalFileStore("./.cache/")

splitter = CharacterTextSplitter.from_tiktoken_encoder(
    separator="\n",
    chunk_size=600,
    chunk_overlap=100,
)
loader = UnstructuredFileLoader("./rag/file/현진건-운수_좋은_날+B3356-개벽.txt")

docs = loader.load_and_split(text_splitter=splitter)

embeddings = OpenAIEmbeddings()

cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

vector_store = FAISS.from_documents(docs, cached_embeddings)


chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="refine",
    retriever=vector_store.as_retriever(),
)
chain.run("주인공에 대해 요약해줘")
