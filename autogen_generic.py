# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 18:11:04 2023

@author: o_hol
"""
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

from dataclasses import dataclass
from typing import List
import pandas as pd
import openai
import time
import os
import autogen
import json

##MEMGPT IMPORTS
memgptimports = True
try:
    import memgpt.autogen.memgpt_agent as memgpt_autogen
    import memgpt.autogen.interface as autogen_interface
    import memgpt.agent as agent
    import memgpt.system as system
    import memgpt.utils as utils
    import memgpt.presets as presets
    import memgpt.constants as constants
    import memgpt.personas.personas as personas
    import memgpt.humans.humans as humans

    from memgpt.autogen.memgpt_agent import create_autogen_memgpt_agent, create_memgpt_autogen_agent_from_config
    from memgpt.persistence_manager import InMemoryStateManager, InMemoryStateManagerWithPreloadedArchivalMemory, InMemoryStateManagerWithFaiss
except:
    memgptimports = False
    pass

##



@dataclass
class Config:
    api_type: str = "open_ai"
    base_url: str = "http://localhost:1234/v1" #prev api_base
    api_key: str = "NULL"
    seed : int() = 47
    cache : bool = True
    temperature : int() = 0.5
    max_tokens : int() = -1
    request_timeout: int() = 6000
    retry_wait_time : int() = 30
    
    #for memgpt
    if memgptimports:
        interface = autogen_interface.AutoGenInterface()
        persistence_manager = InMemoryStateManager()
    
    
    @property
    def api(self):
        configdict = {
            #"api_type": self.api_type,
            "organization": self.api_type,
            "base_url": self.base_url, #this comes from the opensource LLM STUDIO #prev api_base
            "api_key": self.api_key
        }
        
        return configdict
    @property
    def llm(self,functions = None):
        config_list = [self.api]
        llm_config = {
            "functions": functions,
            "config_list": config_list,
            "seed": self.seed,
            #"use_cache" : self.cache,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            #"request_timeout": self.request_timeout,
            "timeout": self.request_timeout,
            #"retry_wait_time" : self.retry_wait_time,
            "max_retries":5
        }
        return llm_config
    
    @property
    def llmuser(self,functions = None):
        config_list = [self.api]
        llm_config = {
            "config_list": config_list,
            "seed": self.seed,
            "use_cache" : self.cache,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "request_timeout": self.request_timeout,
            "timeout": self.request_timeout,
            "retry_wait_time": self.retry_wait_time,
            "max_retries":5
        }
        return llm_config
    
    def memgpt(self,model=''):
        openai.base_url = self.base_url#openai.api_base = self.api_base
        openai.api_key = self.api_key
        config_list= [self.api]
        llm_config_memgpt = {"config_list": config_list,"seed":self.seed}
        
        return llm_config_memgpt
              
            
                

    
    def groupchat(self,autoagents:dict ,messages=[]):
        
        
        groupchat = GroupChat(agents=list(autoagents.values()), messages=[])
        manager = GroupChatManager(groupchat=groupchat, llm_config=self.llm)
        return manager





# groupchat = GroupChat(
#     agents=[user_proxy, content_creator, script_writer, researcher, reviewer], messages=[]
# )
# manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)

# user_proxy.initiate_chat(manager, message="I need to create a Social media post in english and spanish that talks about the latest scienctific news in cancer research.")






@dataclass
class AutoAgents:
    agents:list()
    messages : list()
    goal : str
    config : object
    auto_reply:str = "You are going to figure all out by your own. Work by yourself the user wont reply unitl you output TERMINATE to end the conversation."
    human_input_mode:str  = "TERMINATE"
    workdir:str = "groupchat"
    clear_history : bool = True
    
    @staticmethod
    def retry(fun,max_retries = 5,interval = 10):
        def retry_wrapper(*args,**kwargs):
            attempt = 0
            while attempt < max_retries:
                try:
                    result= fun(*args,**kwargs)
                    return result
                except Exception as e:
                    print(e)
                    time.sleep(interval)
                    print(f'retrying attempt {attempt+1} out of {max_retries}')
                    #self.driver.quit()
                    attempt+=1
                    result = 'ERROR' 
            return result        
        return retry_wrapper  
    

    
    
    def buildagents(self):
        agentsdict = dict(zip(self.agents,self.messages))
        self.user_proxy = UserProxyAgent(
            name="User_proxy",
            system_message="A human admin.",
            code_execution_config = {"last_n_messages":2, "work_dir":self.workdir, "use_docker": "python:3"},
            max_consecutive_auto_reply=10,
            llm_config=self.config.llm,
            human_input_mode = self.human_input_mode,
            default_auto_reply = json.dumps(self.auto_reply),
            is_termination_msg = lambda x: (
            False if x.get("content") is None else x.get("content", "").rstrip().endswith("TERMINATE") 
            if isinstance(x.get("content"), dict) else False
            ),

            
            #human_input_mode="NEVER"
        )#ALWAYS A HUMAN USER PROXY as admin
        
        autoAgents = {}
        autoAgents["user_proxy"] = self.user_proxy
        for agent in self.agents:
            if 'memgpt' in agent.lower():
                
                persona = agentsdict[agent]
                human = 'Im a team manager'
                memgpt_agent = presets.use_preset(presets.DEFAULT_PRESET, model = "gpt-4",persona = persona, human = human, interface = self.config.interface, persistence_manager = self.config.persistence_manager,agent_config = self.config.llm)
                autoAgents[agent] = memgpt_autogen.MemGPTAgent(name = agent, agent = memgpt_agent,default_auto_reply = json.dumps("Nothing to reply now lets move on"))
            else:    
                autoAgents[agent] = AssistantAgent(name=agent, system_message=agentsdict[agent], llm_config=self.config.llm, max_consecutive_auto_reply = 10)
            
        return autoAgents
    
    
    @retry         
    def groupchat_start(self):
        autoagents = self.buildagents()
        print(f'we have {len(autoagents)} num of agents')
        if len(autoagents)<=2:
            assistag = list(autoagents.keys())[-1]
            return self.user_proxy.initiate_chat(autoagents[assistag], message=self.goal)
        else:
            manager =self.config.groupchat(autoagents)
        
            print(manager)
            self.user_proxy.initiate_chat(manager, message=self.goal,clear_history=self.clear_history) #add clear_history=False to continue unfinished conversation

    

    
    
    
agents = ["content_creator","Script_Writer mempgt","Translator","Researcher","Reviewer"]


messages = ["I am a content creator that talks about exciting technologies about Science.  I want to create exciting divulgative content for my audience that is about the wonders of science.  I want to provide in-depth details of the latest Science news.",
            "I am a script writer for the Content Creator.  This should be an eloquently written script so the Content Creator can talk to the audience about Science.",
            "I am the translator for the script writer.  The translation should be as good as the original text written by the script writer but in spanish, make sure to use latin american spanish style.",
            "I am the researcher for the Content Creator and look up the latest scientific developments.  Make sure to include the paper Title Author and Year it was introduced to the Script_Writer.",
            "I am the reviewer for the Content Creator, Script Writer, and Researcher once they are done and have come up with a script.  I will double check the script and provide feedback."]
   
goal = "I need to create a Social media post in english and spanish that talks about the latest scienctific news in cancer research."





if __name__ == "__main__":
    
    
    config = Config() #pass the necessary configs
    agi = AutoAgents(agents, messages, goal,config) #initialize autonomous agents
#     # autoags = agi.buildagents()
#     # print(f'we have {len(autoags)} num of agents')
#     # manager = agi.config.groupchat(autoags)
#     # print(manager)
#     # print(autoags['user_proxy'])
#     # print(autoags)
#     #autoags['user_proxy'].initiate_chat(manager, message=agi.goal)
#     #agi.user_proxy.initiate_chat(manager, message=agi.goal)
    agi.groupchat_start() # start 
    
    
    # groupchat = GroupChat(
    #     agents=[user_proxy, content_creator, script_writer, researcher, reviewer], messages=[]
    # )
    
    # manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)
    
    
    # user_proxy.initiate_chat(manager, message="I need to create a Social media post in english and spanish that talks about the latest scienctific news in cancer research.")
        


# # API Configurations
# api_config = APIConfig(api_type="open_ai", api_base="http://localhost:1234/v1", api_key="NULL")

# # LLM Configurations
# llm_config = LLMConfig(config_list=[api_config], seed=47, temperature=0.5, max_tokens=-1, request_timeout=6000)

# # Agent Configurations
# user_proxy = AgentConfig(name="user_proxy", system_message="A human admin.", llm_config=llm_config)
# content_creator = AgentConfig(name="content_creator", system_message="...", llm_config=llm_config)
# script_writer = AgentConfig(name="Script_Writer", system_message="...", llm_config=llm_config)
# translator = AgentConfig(name="Translator", system_message="...", llm_config=llm_config)
# researcher = AgentConfig(name="Researcher", system_message="...", llm_config=llm_config)
# reviewer = AgentConfig(name="Reviewer", system_message="...", llm_config=llm_config)

# # GroupChat Configuration
# groupchat_config = GroupChatConfig(agents=[user_proxy, content_creator, script_writer, researcher, reviewer], messages=[])

# # Initialize the GroupChat
# groupchat = GroupChat(**groupchat_config.__dict__)

# # Use groupchat and llm_config as needed
