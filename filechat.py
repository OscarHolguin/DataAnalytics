# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 11:50:17 2023

@author: o_hol
"""

import streamlit as st
import openai
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory



def get_pdf_prompts(qa):
    prompt = "Based on my pdf text give me 5 suggested prompts to get insights I can analyze from my text, be brief only 1 sentence per prompt"
    result =  qa({"question": prompt}).get('answer')
    return result



def get_pdf_agent(texts,model="gpt-3.5-turbo-0613",temperature=0.0 ,max_tokens=1048 ,top_p=0.5):
    
    openai.api_key = openai_api_key = st.secrets["openai_key"]
    
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    
    db = Chroma.from_documents(texts, embeddings, persist_directory="filesdb")

    
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        openai_api_key = st.secrets["openai_key"]
    
    )
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        verbose=False,
    )





    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(llm, db.as_retriever(), memory=memory)
    
    return qa



def generate_responsepdf(qa,query):

    result = qa({"question": query}).get('answer')
    return result
# query = ''
# while query!='exit':

#     query =  input('Hi there ask me something when you want to finish only write exit ' )
#     if query =='':
#         print('EMPTY QUERY RUN AGAIN')
#         break
#     print('Working on it please wait')
#     result = 
#     print(result["answer"])



   # texts = text_splitter.create_documents(documents)
   #      # Select embeddings
   #      embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
   #      # Create a vectorstore from documents
   #      db = Chroma.from_documents(texts, embeddings)
   #      # Create retriever interface
   #      retriever = db.as_retriever()
   #      # Create QA chain
   #      qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key), chain_type='stuff', retriever=retriever)
   #      return qa.run(query_text)