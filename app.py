from document_processing import load_documents,split_text
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Weaviate
import os                    
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("openai_api_key")
#print(openai_api_key)

PROMPT_TEMPLATE = """
    Answer the question based only on the following context,keep the answer detailed,

    {context}

    ---

    Answer the question based on the above context: {question}
    """

documents = load_documents()
chunks = split_text(documents)

#openai_api_key="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"
def main(query):
    # Embedding the text data and saving into Waeviate db
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = Weaviate.from_documents(chunks, embeddings, weaviate_url='http://localhost:8080', by_text=False)


    #query = query


    #Finding out responce based on query using similarity search on vectors which are saved in db.
    results = db.similarity_search_with_relevance_scores(query=query,k=3)
    #results format -------> [(document_info,score)]
    #print((results[0][0].page_content))
    if len(results) == 0 or results[0][1] < 0.5:
            return (f"Unable to find matching results.")
    #print(results)
    scores = []
    for i,j in results:
          scores.append(j)
    #print(scores)
    #Featching the overall context from the query result.
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    print(context_text)


    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query)
    #print(prompt)

    model = ChatOpenAI(openai_api_key=openai_api_key)
    response_text = model.predict(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}\nScores:{scores}"
    return formatted_response

print(main(query = 'what are the types of machine learning?'))