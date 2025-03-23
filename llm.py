# pip install langchain
# pip install openai
# pip install langchain-openai
# pip install python-dotenv

from langchain_openai import OpenAI
from dotenv import load_dotenv
 
load_dotenv()
 
llm = OpenAI(model='gpt-3.5-turbo-instruct', temperature=0.6) 

result = llm.invoke("I want to open a restatunt for Indin food. Suggest a fancy name for this")

print(result)


# python llm.py