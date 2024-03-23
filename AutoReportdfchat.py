# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 14:55:27 2023

@author: o_hol


CHAT ENABLED PDF VERSION
"""



import asyncio

import nltk
nltk.download('punkt')
import plotly_express as px
import streamlit as st
import pandas as pd
from langchain.document_loaders import PyPDFLoader, UnstructuredPDFLoader, OnlinePDFLoader
# from langchain.embeddings import HuggingFaceEmbeddings
import pyodbc
from langchain.vectorstores import Chroma
import urllib
import numpy as np
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
import plotly.io as pio
pio.templates.default = 'plotly'
# pio.renderers.default = 'iframe'
import matplotlib
import codecs
import os
import pygwalker as pyg
import base64
import streamlit.components.v1 as components

from streamlit_d3graph import d3graph
import d3graph
import networkx as nx
from dataclasses import dataclass
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
import uuid
import spacy
import pathlib
import time
import glob

############################
#
from datachat import generate_response,write_response,generate_insights_one,generate_trends_and_patterns_one,aggregate_data,get_agent, get_insight_prompts

import hmac


def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the passward is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False


#if not check_password():
#    st.stop()  # Do not continue if check_password is not True.


pagetitle = "Chat with Data"


st.set_page_config(page_title = pagetitle,
                   page_icon ="👋" ,#"https://www.thinkdatadynamics.com/dark/assets/imgs/logo-light.png",#"https://static.wixstatic.com/media/d75272_496750661bda495c8beaea93cf6dfcab~mv2.png/v1/fill/w_230,h_60,al_c,q_85,usm_0.66_1.00_0.01,enc_auto/Asset%201Logo.png",
                   layout="wide",
                   initial_sidebar_state="expanded", 
    )



st.markdown(
    """
    <style>
    button[kind="primary"] {
        background: 5px navy;
        border: none;
        padding: 0!important;
        color: white !important;
        text-decoration: none;
        cursor: pointer;
        border: none !important;
    }
    button[kind="primary"]:hover {
        text-decoration: click over me;
        color: orange !important;
    }
    button[kind="primary"]:focus {
        outline: none !important;
        box-shadow: none !important;
        color: blue !important;
    }
    /* Add this selector to create a box around the button with kind=primary */
    div.stButton > button[kind=primary] {
        border: 2px navy;
        border-radius: 10px;
        box-shadow: 0 0 5px black;
        background-color: navy;
        margin: 10px;
        padding: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)



if "app_stopped" not in st.session_state:
    st.session_state["app_stopped"] = False 
elif st.session_state["app_stopped"]:
    st.session_state["app_stopped"] = False


def Running():
    with st.spinner("running"):
        time.sleep(60)

def stopRunning():
    try:
        st.session_state["app_stopped"] = True
    except:
        pass
def reset_conversation():
    try:
        st.session_state.conversation = None
        st.session_state.chat_history = None
        vecrag,llmrca,loaded_embeddings,rag2 = importinits()
        st.session_state.messages = [{"role": "assistant", "content": "Hi this is your Copilot! How can I help you?"}]

        #st.cache_data.clear
        st.empty()
    except:
        pass
def continue_generating():
    st.session_state.messages.append({"role": "user", "content": "Continue Generating"})
    with st.chat_message("assistant"):
        response =  stream_response(get_copilot_response("Continue Generating"))
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message) 

if st.session_state["app_stopped"]:
    print("stopping this NOW")
    st.stop()

st.title(":bar_chart: :clipboard:")
st.image('https://www.thinkdatadynamics.com/dark/assets/imgs/logo-light.png',width=350)


hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

st.header("",divider="rainbow")


left,mid,right = st.columns([1,3,1],gap='large')


#if not st.toggle("From url"):
data_files = st.file_uploader("Choose from your files :file_folder:",type=['csv','xlsx'],accept_multiple_files=True)
file_ext = option_chosen = "null"
if data_files is not None:
    file_names = []
    file_exts = []
    for data_file in data_files:
        file_name = data_file.name
        file_names.append(file_name)
        file_ext = file_name.split(".")[-1]
        file_exts.append(file_ext)
        urlflag = urlfile = False
else:
    #add clear session state here  this gets the last file modified from excel or csv in memory
    st.session_state.conversation = []
    st.session_state.chat_history = []

    try:
        all_files = glob.glob(os.path.join('*.csv')) + glob.glob(os.path.join( '*.xlsx'))
        data_file = max(all_files, key=os.path.getctime)
        file_name= data_file.split(".")[0]
        file_ext = "csv"
        urlflag = urlfile = False
    #file_ext = file_name.split(".")[-1]
    #urlflag = urlfile = False
    except:
        pass

# else:    
#         urlfile = st.text_input("Provide your CSV or Excel from a valid url")
#         urlflag = True
#         st.write("You added ",urlfile)
#         file_ext = urlfile.split(".")[-1]
#         file_name = urlfile
#         data_files = [file_name]
#         file_names = [file_name]
#         file_exts = [file_ext]
#elif from database
        


@st.cache_resource
def generate_chat_response(df,prompt,openail=True):
    return generate_response(df,prompt,openail=True)
    
    
#@st.cache_resource
def write_suggestions(suggestions):
    from streamlit_pills import pills
    selections={}
    selected = pills("Suggestions for chat",suggestions,)
    st.write(selected)
    return selected

        

# Define a callback function that takes the selected suggestion as an argument
def on_submit(suggestion):
    # Do something with the suggestion, such as sending it to a chatbot
    st.write(f"You have selected: {suggestion}")

def stream_response(response,speech=False):
    progress_text = "Operation in progress. Please wait."
    my_bar = st.sidebar.progress(0, text=progress_text)
    for n,word in enumerate(response):
        yield word +""
        my_bar.progress(n/len(response)) #progress bar
        time.sleep(0.02)
    if speech:
        speech_response(response)

    my_bar.empty()



#multiple dfs example
dfs = []
for n in range(len(data_files)):
    if file_exts[n] =='csv' or file_exts[n] =='xlsx':
        #tmppath = os.path.join("\tmp", data_file.name)
        if 'toggle' not in st.session_state:
            st.session_state.Chat = False
        tmppath = file_name[n]
        response_history = st.session_state.get("response_history", [])

        if file_ext =='csv':
            df= pd.read_csv(data_files[n],encoding ="utf-8")
        elif file_ext =='xlsx':
            df= pd.read_excel(data_files[n],engine = "openpyxl")


        df.to_csv(tmppath)
        dfs.append(df)
        st.dataframe(df.head())
        #menu = ["Home","Report","Retro_Report","Create"]

if len(dfs)>1:
    st.session_state.dfs = dfs
elif len(dfs)>0:
     st.session_state.dfs = dfs[0]
else:
     st.session_state.dfs = None




    
col1, col2, col3  = st.sidebar.columns(3)

#STOP
col1.button("Stop 🛑",on_click=stopRunning)

#Continue running
#col2.button("Run ▶️", on_click=Running)


#RESET CONVERSATION (CLEAR IT)  
col3.button('Reset', on_click=reset_conversation)
    
import streamlit.components.v1 as components 
    ##chat section
if st.session_state.dfs is not None:
    print("THIS IS MY DATAFRAME LIST: NOWWW ",st.session_state.dfs)

    #SUGGESTIONS PART TO BE UNCOMMENTED
    pdfagent1 = get_agent(st.session_state.dfs) #multiple dfs can be passed
    if sinsights:= st.sidebar.toggle("Suggest insights :bulb:"):
        st.sidebar.subheader("Suggested Inisghts ::bar_chart: :chart_with_downwards_trend:")
        suggestions = get_insight_prompts(pdfagent1)
        suggestions_s = [n for n in nltk.sent_tokenize(' '.join([x for x in nltk.word_tokenize(suggestions)]))]
        suggestions_s = [x for x in suggestions_s if x not in [str(n)+' .' for n in list(range(1,6))]]
        # print(suggestions_s[0])
        clickables = {}

        with st.sidebar: 
            for sug in suggestions_s:
                clickables[sug] = st.button(sug + " ➤ ", type="primary")                    
                              # st.write(response)
                              # message = {"role": "assistant", "content": response}
                              # st.session_state.messages.append(message) 

    # # suggestionins = ["Give the best insight for this data","plot the best insight","Calculate the best metric","Provide an in detail analysis for a stakeholder"]
    st.session_state.prompt_history = []

    if "messages" in st.session_state:
        print('messages found')
        
    else:
        if "messages" not in st.session_state.keys():
            st.session_state.messages = [{"role": "assistant", "content": "Hi! How can I help you with your data?"}]
    
    if "messages" in st.session_state:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                try:
                    exec(message["content"])
                except:
                    st.write(message["content"])
        
        if prompt := st.chat_input():

            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

        
        if st.session_state.messages[-1]["role"] != "assistant":
            if not prompt and sinsights:
                for sug in suggestions_s:
                    if clickables[sug]:
                        # with st.chat_message("user"):
                        #     st.write(clickables[sug])
                        # # st.session_state.messages.append({"role": "user", "content": sug})
                        # with st.chat_message("assistant"):
                        #     with st.spinner("Thinking..."):
                        #         response =  generate_chat_response(df,sug,openail=True)
                        #         st.write(response)
                        prompt = clickables[sug]
            if prompt:
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        #response =  generate_responsedf(df,prompt)
                        response =  generate_chat_response(dfs,prompt,openail=True)
                        try:
                            #try executing the code first
                            exec(response)
                        except Exception as e:
                            print(e)
                            st.write_stream(stream_response(e))

                        message = {"role": "assistant", "content": response}
                        st.session_state.messages.append(message) 
                        if "insights2" in prompt.lower():
                            insights = generate_insights_one(st.session_state.dfs)
                            st.write(insights)
                        elif "trends2" in prompt.lower() or "patterns" in prompt.lower():
                            trends_and_patterns = generate_trends_and_patterns_one(st.session_state.dfs)
                            for fig in trends_and_patterns:
                                if fig is not None:
                                    st.pyplot(fig)
                        elif "aggregate" in prompt.lower():
                            columns = prompt.lower().split("aggregate ")[1].split(" and ")
                            aggregated_data = aggregate_data(st.session_state.dfs, columns)
                            st.subheader("Aggregated Data:")
                            st.write(aggregated_data)                        
                #To generate images if needed
                fig = plt.gcf()
                fig, ax = plt.subplots(figsize=(10, 6))
                plt.tight_layout()
#                    if fig.get_axes() and fig is not None:
#                      st.pyplot(fig)
#                      fig.savefig("plot.png")
#                    st.write(response)
                st.session_state.prompt_history.append(prompt)
                response_history.append(response)
                st.session_state.response_history = response_history
        else:
            if sinsights:
                for sug in suggestions_s:
                    if clickables[sug]:
                        with st.chat_message("user"):
                            st.session_state.messages.append({"role": "user", "content": sug})
                            st.write(sug)
                        promptinsight = sug
                        with st.chat_message("assistant"):
                            with st.spinner("Thinking..."):
                                response =  generate_chat_response(st.session_state.dfs,promptinsight,openail=True)
                                st.write(response)
                                message = {"role": "assistant", "content": response}
                                st.session_state.messages.append(message)
                                
                                st.session_state.prompt_history.append(prompt)
                                response_history.append(response)
                                st.session_state.response_history = response_history





#Langserve bot will be reading wither the uploaded or latest files