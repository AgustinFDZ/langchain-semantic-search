from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
import os
import getpass

file_path = "./assets/Perros_Raza_Boxer.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)

all_splits = text_splitter.split_documents(docs)

if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your API Key for Google Gemini: ")

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

vector_1 = embeddings.embed_query(all_splits[0].page_content)
vector_2 = embeddings.embed_query(all_splits[1].page_content)
assert len(vector_1) == len(vector_2)

vector_store = InMemoryVectorStore(embeddings)

ids = vector_store.add_documents(all_splits)

results = vector_store.similarity_search("Dieta ideal para un perro Boxer")

print(results[0])




