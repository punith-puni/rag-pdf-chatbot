from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# READ PDF
reader = PdfReader("sample.pdf")

text = ""

for page in reader.pages:
    text += page.extract_text()

# SPLIT TEXT INTO CHUNKS
splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=300,
    chunk_overlap=50,
    length_function=len
)

chunks = splitter.split_text(text)

print("Total Chunks:", len(chunks))

# LOAD EMBEDDING MODEL
model = SentenceTransformer('all-MiniLM-L6-v2')

# CREATE EMBEDDINGS
embeddings = model.encode(chunks)

print("Embedding Shape:", embeddings.shape)

# CREATE FAISS INDEX
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

# STORE EMBEDDINGS
index.add(np.array(embeddings))

print("FAISS vector store created successfully!")