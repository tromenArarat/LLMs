#!/usr/bin/env python
from typing import List

from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_cohere import CohereLanguageModel
from langserve import add_routes
import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("COHERE_API_KEY")

# 1. Create prompt template
system_template = "Translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '{text}')
])


# 2. Create model
model = model = CohereLanguageModel(api_key=api_key)

# 3. Create parser
parser = StrOutputParser()

# 4. Create chain
chain = prompt_template | model | parser

# 4. App definition
app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple API server using LangChain's Runnable interfaces",
)

# 5. Adding chain route

add_routes(
    app,
    chain,
    path="/chain",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)

    from langserve import RemoteRunnable

# remote_chain = RemoteRunnable("http://localhost:8000/chain/")
# remote_chain.invoke({"language": "italian", "text": "hi"})