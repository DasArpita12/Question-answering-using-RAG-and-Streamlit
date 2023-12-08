from langchain.document_loaders import DirectoryLoader,TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


#documents loader
DATA_PATH = r'data/files'


## Data Processing
def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.txt",loader_cls=TextLoader)
    documents = loader.load()
    # print(documents[0].page_content)
    # print(documents[1].page_content)
    # print(documents[2].page_content)
    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    #print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    #print(type(chunks))
    #document = chunks[50]
    #print(document.page_content)
    #print(document.metadata)
    print(len(chunks))
    return chunks

split_text(load_documents())