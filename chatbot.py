from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

load_dotenv()

# Carregar PDF
loader = PyPDFLoader("https://raw.githubusercontent.com/sebavassou/chatbot_fila_into/main/dados/Cartilha-Funcionamento-da-Lista-de-Espera-do-Into_web2.pdf")
documents = loader.load()

# Processar documentos
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = splitter.split_documents(documents)

# Criar vetores
embeddings = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
vector_store = FAISS.from_documents(docs, embeddings)

# Configurar LLM
llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)
