import os

from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi import FastAPI

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_ibm import WatsonxLLM

load_dotenv()

credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "apikey": os.environ['WATSONX_API_KEY'],
    "project_id": os.environ['WATSON_ML_PROJECT']
}

model_param = {
    "decoding_method": "greedy",
    "temperature": 0,
    "min_new_tokens": 5,
    "max_new_tokens": 500
    }

app = FastAPI()

class Query(BaseModel):
    question: str

class Query_Response(BaseModel):
    result : str

@app.post("/question")
async def question(q: Query) -> Query_Response:
    question = q.question

    print("Questions")
    print(question)

    db = SQLDatabase.from_uri(os.environ['DB_URL'])
    print(db.dialect)
    print(db.get_usable_table_names())

    # model_id = "meta-llama/llama-3-405b-instruct"
    model_id = "meta-llama/llama-3-2-90b-vision-instruct"

    llm = WatsonxLLM(
        model_id=model_id,
        url=credentials.get("url"),
        apikey=credentials.get("apikey"),
        project_id=credentials.get("project_id"),
        params=model_param
    )

    agent_executor = create_sql_agent(llm, db=db, verbose=True, handle_parsing_errors=True)

    final_state = agent_executor.invoke(question)
    res = final_state["output"]

    query_response = Query_Response(result=res)

    return query_response

