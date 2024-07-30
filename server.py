#!/usr/bin/env python
from fastapi import FastAPI, HTTPException
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
import cohere
import os
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

# Load Cohere API key from environment variable
api_key = os.getenv("COHERE_API_KEY")
if not api_key:
    raise ValueError("No API key found. Please set the COHERE_API_KEY environment variable.")

# Define a Cohere model class
class CohereModel:
    def __init__(self, api_key):
        self.client = cohere.Client(api_key)

    def generate(self, prompt):
        response = self.client.generate(
            model='command-xlarge-nightly',
            prompt=prompt,
            max_tokens=450,
            temperature=0.7
        )
        return response.generations[0].text

# 1. Create prompt template
system_template = "Translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '{text}')
])

# 2. Create model
model = CohereModel(api_key=api_key)

# 3. Create parser
parser = StrOutputParser()

# 4. Create Runnable chain
class TranslationChain(Runnable):
    def __init__(self, model, prompt_template, parser):
        self.model = model
        self.prompt_template = prompt_template
        self.parser = parser

    def invoke(self, input_text):
        prompt = self.prompt_template.format(language=input_text['language'], text=input_text['text'])
        generated_text = self.model.generate(prompt)
        return self.parser.parse(generated_text)

# Instantiate the chain
chain = TranslationChain(model, prompt_template, parser)

# 4. App definition
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using LangChain's Runnable interfaces",
)

# Define the input model for the request
class TranslationRequest(BaseModel):
    language: str
    text: str

# 5. Adding chain route
@app.post("/chain")
async def run_chain(request: TranslationRequest):
    try:
        result = chain.invoke(request.dict())
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)

