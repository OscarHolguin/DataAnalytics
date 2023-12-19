# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 20:18:25 2023

@author: o_hol
"""




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import json
# from pandasai import SmartDataframe
# from pandasai.llm import HuggingFace
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent


###
import os
from langchain import PromptTemplate, HuggingFaceHub, LLMChain, OpenAI, SQLDatabase
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_experimental.agents import create_csv_agent, create_pandas_dataframe_agent

# from langchain.chains.sql_database.base import SQLDatabaseChain
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
import transformers
from transformers import pipeline



def generate_prompt(question):
    promptquery = (
       """
          For the following query, if it requires drawing a table, reply as follows:
           {{"table": {{"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}}

           If the query requires creating a bar chart, reply as follows:
           {{"bar": {{"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}}

           If the query requires creating a line chart, reply as follows:
           {{"line": {{"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}}

           There can only be two types of chart, "bar" and "line".

           If it is just asking a question that requires neither, reply as follows:
           {{"answer": "answer"}}
           Example:
           {{"answer": "The title with the highest rating is 'Gilead'"}}

           If you do not know the answer, reply as follows:
           {{"answer": "I do not know."}}

           Return all output as a string.

           All strings in "columns" list and data list should be in double quotes,

           For example: {{"columns": ["title", "ratings_count"], "data": [["Gilead", 361], ["Spider's Web", 5164]]}}

           Lets think step by step.

           Below is the query.
           Query: 
           
       
       """+str(question)
   )
    
    
    return promptquery




def write_response(response_dict: dict):
    """
    Write a response from an agent to a Streamlit app.

    Args:
        response_dict: The response from the agent.

    Returns:
        None.
    """

    # Check if the response is an answer.
    if "answer" in response_dict:
        st.write(response_dict)

    # Check if the response is a bar chart.
    if "bar" in response_dict:
        data = response_dict["bar"]
        df = pd.DataFrame(data)
        df.set_index("columns", inplace=True)
        st.bar_chart(df)

    # Check if the response is a line chart.
    if "line" in response_dict:
        data = response_dict["line"]
        df = pd.DataFrame(data)
        df.set_index("columns", inplace=True)
        st.line_chart(df)

    # Check if the response is a table.
    if "table" in response_dict:
        data = response_dict["table"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        st.table(df)





model_id = 'google/flan-t5-base'#'-xxl'
def generate_response(df,prompt,model_id=model_id,openai=False):
    if not openai:
        config = AutoConfig.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id, config=config)
        pipe = pipeline('text2text-generation',
                    model=model,
                    tokenizer=tokenizer,
                    max_length = 512
                    )
        local_llm = HuggingFacePipeline(pipeline = pipe)

    
        prompt2 = generate_prompt(prompt)
        
        agent =  create_pandas_dataframe_agent(llm = local_llm,df=df ,verbose=True)
        response = agent.run(prompt2)
    else:
        from langchain.chat_models import ChatOpenAI
        from langchain import OpenAI

        llm = OpenAI(openai_api_key=st.secrets["openai_key"])

        agent = create_pandas_dataframe_agent(llm, df, verbose=True)
        #agent = create_pandas_dataframe_agent(ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
        #    ,df,verbose=True,agent_type=AgentType.OPENAI_FUNCTIONS)
        prompt2 = generate_prompt(prompt)

        response = agent.run(prompt2)
        try:
            response = json.loads(response)
        except:
            response = response
    return response
    
    
    
#    try:
#        result = agent.run(prompt2)
#        result = result.__str__()
#    except Exception as e:
#        result = str(e)
#        if result.startswith("Could not parse LLM output: `"):
#             result = result.removeprefix("Could not parse LLM output: `").removesuffix("`")
#        result = result.__str__()
#    return result

  
def generate_responsedf(df,prompt):
    print("USING PANDASAI")
    # #NEW VERSION
    # from pandasai import SmartDataframe
    # # from pandasai.llm import HuggingFace
    # from pandasai.llm import Starcoder, Falcon
    # api_token='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'
    
    # llm = Starcoder()
    # from pandasai.responses.streamlit_response import StreamlitResponse
    # aidf = SmartDataframe(df,  config={"llm": llm, "verbose": True, "response_parser": StreamlitResponse})
    
    # #prev version
    from pandasai import PandasAI
    from pandasai.llm.starcoder import Starcoder
    from pandasai.llm.openai import OpenAI
    os.environ['HUGGINGFACE_API_KEY'] = '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'
    openaikey = st.secrets["openai_key"]
    llm = OpenAI(api_token = openaikey) #llm = Starcoder()
    pandas_ai = PandasAI(llm)
    
    return pandas_ai.run(df, prompt=prompt)
    # return aidf.chat(prompt)  


    



def generate_insights(df):
    insights = []
    
    # Summary statistics
    summary_stats = df.describe()
    insights.append("Summary Statistics:\n" + summary_stats.to_string())

    # Missing values
    missing_values_count = df.isnull().sum()
    missing_values_percent = (missing_values_count / len(df)) * 100
    missing_values_summary = pd.DataFrame({
        "Missing Values": missing_values_count,
        "Missing Values %": missing_values_percent
    })
    insights.append("Missing Values Summary:\n" + missing_values_summary.to_string())

    # Correlation matrix
    correlation_matrix = df.corr()
    insights.append("Correlation Matrix:\n" + correlation_matrix.to_string())

    return "\n\n".join(insights)


def generate_trends_and_patterns(df):
    trends_and_patterns = []

    # Distribution of numerical columns
    numerical_cols = df.select_dtypes(include=["float64", "int64"]).columns
    for col in numerical_cols:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(df[col], ax=ax)
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.title("Distribution of " + col)
        plt.tight_layout()
        trends_and_patterns.append(fig)

    return trends_and_patterns

def aggregate_data(df, columns):
    aggregated_data = df.groupby(columns).size().reset_index(name='Count')
    return aggregated_data

def generate_insights_one(df):
    insights = []

    # Summary statistics
    insights.append("Summary Statistics:")
    insights.append(df.describe().to_string()) #

    # Missing values
    missing_values = df.isnull().sum()
    insights.append("Missing Values:")
    insights.append(missing_values.to_string())
    
    # Missing values summary
    missing_values_count = df.isnull().sum()
    missing_values_percent = (missing_values_count / len(df)) * 100
    missing_values_summary = pd.DataFrame({
        "Missing Values": missing_values_count,
        "Missing Values %": missing_values_percent
    })
    insights.append("Missing Values Summary:\n")
    insights.append(missing_values_summary.to_string())
    
    # Data types
    data_types = df.dtypes
    insights.append("Data Types:")
    insights.append(data_types.to_string())

    # Unique values
    unique_values = df.nunique()
    insights.append("Unique Values:")
    insights.append(unique_values.to_string())

    # Correlation matrix
    correlation_matrix = df.corr()
    insights.append("Correlation Matrix:")
    insights.append(correlation_matrix.to_string())

    return "\n\n".join(insights)

def generate_trends_and_patterns_one(df):
    trends_and_patterns = []

    # Distribution of numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(df[col], ax=ax)
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        trends_and_patterns.append(fig)

    # Pairwise scatter plots
    sns.set(style="ticks")
    scatter_matrix = sns.pairplot(df, diag_kind="kde")
    trends_and_patterns.append(scatter_matrix.fig)

    return trends_and_patterns

