# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 14:30:01 2023

@author: o_hol
"""

#AUTOGEN REPORT CHAT


from dataclasses import dataclass
import autogen
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent

from autogen_generic import Config
from autogen_generic import AutoAgents
import openai

apibase = "http://localhost:4200/v1"
openai.api_base = apibase
openai.api_key = "NULL"


config = Config(base_url = apibase,seed = 1,request_timeout = 6000,retry_wait_time = 10)

llm_config = config.llm

assistant = RetrieveAssistantAgent(
    name="assistant",
    system_message="You are a helpful assistant.",
    llm_config=llm_config,
)

ragproxyagent = RetrieveUserProxyAgent(
    name="ragproxyagent",
    retrieve_config={
        "task": "qa",
        "docs_path": "https://raw.githubusercontent.com/microsoft/autogen/main/README.md",
    },
)




if __name__ == "__main__":
    
    modelpath = r"C:\Users\o_hol\.cache\lm-studio\models\TheBloke\Mistral-7B-Instruct-v0.1-GGUF\mistral-7b-instruct-v0.1.Q5_0.gguf"

    assistant.reset()
    ragproxyagent.initiate_chat(assistant,problem = "What is autogen?")
    



@dataclass
class RAGchat():
    pass


