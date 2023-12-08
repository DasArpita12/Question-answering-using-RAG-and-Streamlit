from document_processing import load_documents,split_text
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Weaviate
import os                    
from langchain.docstore.document import Document
from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain import OpenAI
from langchain.pydantic_v1 import BaseModel, Field
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers.openai_functions import PydanticOutputFunctionsParser
from langchain.prompts import PromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
from langchain.schema.prompt_template import format_document
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from langchain.schema import Document
from chains import map_rerank_chain
import streamlit as st
from langchain.document_loaders import TextLoader

load_dotenv()
openai_api_key = os.getenv("openai_api_key")
#print(openai_api_key)
#llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
# documents = load_documents()
# chunks = split_text(documents)
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
# loader = TextLoader()
# loader.load()


def app():
    st.set_page_config(page_title="Chat with multiple documents",
                       page_icon=":books:")
    #db = Weaviate.from_documents(chunks, embeddings, weaviate_url='http://localhost:8080', by_text=False)


    
    with st.sidebar:
        st.subheader("Your documents")
        docs = st.file_uploader(
            "Upload your data and click on 'Process'", accept_multiple_files=True)
        #print(docs)
    if docs is not None:  
        all_docs = []
        for file in docs:
            file = file.read().decode("utf-8")
            doc =  Document(page_content=file, metadata={"source": "local"})
            all_docs.append(doc)
    query = st.text_input("Ask a question about your documents:")
    chunks = split_text(all_docs)
    db = Weaviate.from_documents(chunks, embeddings, weaviate_url='http://localhost:8080', by_text=False)

    #Finding out responce based on query using similarity search on vectors which are saved in db.
    results = db.similarity_search_with_relevance_scores(query=query,k=10)

    final_results = []
    for i in results:
                if i[1]>0.7:
                    final_results.append(i)
                else:
                    pass
    print(len(results),len(final_results))
                
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in final_results])
    print((context_text))

    if len(context_text) != 0:
            
        docs = [
            Document(
                page_content=split,
            )
            for split in context_text.split("\n\n")
        ]

        #   st.markdown(docs[0].page_content)


        st.markdown(
            map_rerank_chain.invoke({"docs": docs, "question": query})
        )
    else:
        st.markdown (
            'Information is not avilable in the given document'
        )
app()