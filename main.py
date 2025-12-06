from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
import os
from tools import search_tool, wiki_tool,save_tool, penn_state_tool, get_tools_for_query
# Load environment variables
load_dotenv()

class ResearchResponse(BaseModel): # response model/how the LLM response format
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]


# Verify the API key is loaded
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("Error: OPENAI_API_KEY not found in .env file")
    exit()

print("API key loaded successfully!")

llm = ChatOpenAI(model="gpt-4o-mini") #output_model

parser = PydanticOutputParser(pydantic_object=ResearchResponse)#parser
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",#system message
            """
            You are an expert research assistant that will help generate a reasearch paper.
            Answer the user query and use neccessary tools.
            Wrap the output in strictly in this format and provide no other text 
            Do not deviate from the format!
            \n{format_instructions} """ #ResearchResponse format ouput
        ),
        
        ("placeholder", "{chat_history}"), #automatically filled by agentExecutor
        ("human", "{query}"), #user input (query)
        ("placeholder","{agent_scratchpad}"), #automatically filled by agentExecutor
        
    ]
).partial(format_instructions=parser.get_format_instructions())#partial fill in to format string output
tools = [] #list of tools
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools, #add tools here
)
query = input("What can i help you with today? ")
# Filter tools dynamically
tools_for_use = get_tools_for_query(query)

agent_executor = AgentExecutor(agent=agent, tools=tools_for_use, verbose=True)#verbose = True to see the thought process of the agent

raw_response = agent_executor.invoke({"query": query})

import json

try:
    output_str = raw_response.get("output")
    structured_response = parser.parse(output_str)

    print(structured_response)

except Exception as e:
    print("Error parsing response", e, "Raw Response - ", raw_response)

