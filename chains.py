from langchain.prompts import PromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
from langchain.output_parsers.openai_functions import PydanticOutputFunctionsParser
from langchain.chat_models import ChatOpenAI
from langchain.schema.prompt_template import format_document
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from dotenv import load_dotenv
import os
load_dotenv()
openai_api_key = os.getenv("openai_api_key")

map_prompt = PromptTemplate.from_template(
    "Answer the user question using the context"
    "\n\nContext:\n\n{context}\n\nQuestion: {question}"
)

class AnswerAndScore(BaseModel):
    """Return the answer to the question and a relevance score."""

    answer: str = Field(
        description="Answer the user question using the context."
    )
    score: float = Field(
        decsription="A 0.0-1.0 relevance score, where 1.0 indicates the provided context answers the question completely and 0.0 indicates the provided context does not answer the question at all."
    )


function = convert_pydantic_to_openai_function(AnswerAndScore)
map_chain = (
    map_prompt
    | ChatOpenAI().bind(
        temperature=0, functions=[function], function_call={"name": "AnswerAndScore"}
    )
    | PydanticOutputFunctionsParser(pydantic_schema=AnswerAndScore)
).with_config(run_name="Map")

# Final chain, which after answer and scoring based on
# each doc return the answer with the highest score.


def top_answer(scored_answers):
    return max(scored_answers, key=lambda x: x.score).answer


document_prompt = PromptTemplate.from_template("{page_content}")
map_rerank_chain = (
    (
        lambda x: [
            {
                "context": format_document(doc, document_prompt),
                "question": x["question"],
            }
            for doc in x["docs"]
        ]
    )
    | map_chain.map()
    | top_answer
).with_config(run_name="Map rerank")
