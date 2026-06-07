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

# CHUNKING
splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=300,
    chunk_overlap=50,
    length_function=len
)

chunks = splitter.split_text(text)

# LOAD MODEL
model = SentenceTransformer('all-MiniLM-L6-v2')

# CREATE EMBEDDINGS
embeddings = model.encode(chunks)

# CREATE FAISS INDEX
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings))

# USER QUESTION
question = input("Ask a question: ")

# CONVERT QUESTION TO VECTOR
question_embedding = model.encode([question])

# SEARCH SIMILAR CHUNKS
k = 2

distances, indices = index.search(
    np.array(question_embedding),
    k
)

print("\nMost Relevant Chunks:\n")

for i in indices[0]:
    print(chunks[i])
    print("\n-----------------\n")