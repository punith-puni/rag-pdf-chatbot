from langchain_text_splitters import CharacterTextSplitter
from PyPDF2 import PdfReader

reader = PdfReader("sample.pdf")

text = ""

for page in reader.pages:
    text += page.extract_text()

splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=300,
    chunk_overlap=50,
    length_function=len
)

chunks = splitter.split_text(text)

print(chunks)