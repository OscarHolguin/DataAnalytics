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
import requests
import json
from io import BytesIO
import time
# from pandasai import SmartDataframe
# from pandasai.llm import HuggingFace
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent


###
import os
from langchain import PromptTemplate, HuggingFaceHub, LLMChain, OpenAI
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_experimental.agents import create_csv_agent, create_pandas_dataframe_agent

# from langchain.chains.sql_database.base import SQLDatabaseChain
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
import transformers
from transformers import pipeline

import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend (for non-interactive use)
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings("ignore", message="Starting a Matplotlib GUI outside of the main thread will likely fail.")


def speech_to_text(audiopath):
    from openai import OpenAI
    client = OpenAI(api_key= st.secrets["openai_key"])
    audio_file = open(audiopath, "rb")
    transcription = client.audio.transcriptions.create(
    model="whisper-1",
    file=audio_file
    )
    return transcription.text



def stream_response_(response,speech=False):
    progress_text = "Operation in progress. Please wait."
    my_bar = st.sidebar.progress(0, text=progress_text)
    for n,word in enumerate(response):
        yield word +""
        my_bar.progress(n/len(response)) #progress bar
        time.sleep(0.02)
    if speech:
        speech_response(response)

    my_bar.empty()


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
        try:
            data = response_dict["bar"]
        except:
            data = response_dict
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






#CODE FOR LANGCHAIN PANDAS DATAFRAME AGENT
def extract_python_code(text):
    import re
    pattern = r'```python\s(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        return None
    else:
        return matches[0]


def get_insight_prompts(agent):
    prompt = "Based on the given dataframe give me 5 prompts of inisghts I can analyze from my data. be brief only 62 charactes max prompt"
    result = agent(prompt)
    return result.get('output')
    


def handle_error(error):
    import ast
    
    error2 = ast.literal_eval(error).get("arguments")
    
    if error2:
        figidx = error2.find("fig.show()") 
        res = error2[:figidx] if figidx!=-1 else error2
        return res
    else:
        return str(error)
    

def get_agent(df,model="gpt-3.5-turbo", temperature=0.0, max_tokens=2500, top_p=0.5):
    from langchain.chat_models import ChatOpenAI
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        #top_p=top_p,
        openai_api_key = st.secrets["openai_key"]

    )

    pandas_df_agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        return_intermediate_steps=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        handle_parsing_errors=handle_error,

    )
    
    return pandas_df_agent



#def get_agent2(df,model="gpt-3.5-turbo", temperature=0.0, max_tokens=2500, top_p=0.5):
from typing import Any, List, Optional
from langchain.agents.agent import AgentExecutor
# from langchain.agents.agent_toolkits.pandas.prompt import PREFIX, SUFFIX
from langchain.agents import ZeroShotAgent,AgentType
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains.llm import LLMChain
from langchain.llms.base import BaseLLM
#from langchain.tools.python.tool import PythonAstREPLTool
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key="chat_history")



def create_pandas_dataframe_agent2(
        df: Any,model="gpt-3.5-turbo", temperature=0.0, max_tokens=2500, top_p=0.5,
        callback_manager: Optional[BaseCallbackManager] = None,
        #prefix: str = PREFIX,
        #suffix: str = SUFFIX,
        input_variables: Optional[List[str]] = None,
        verbose: bool = True,
        handle_parsing_errors : bool = True,
        return_intermediate_steps: bool = False,
        max_iterations: Optional[int] = 15,
        max_execution_time: Optional[float] = None,
        early_stopping_method: str = "force",
        **kwargs: Any,
    ) -> AgentExecutor:
        """Construct a pandas agent from an LLM and dataframe."""
        import pandas as pd

        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Expected pandas object, got {type(df)}")
        if input_variables is None:
            input_variables = ["df", "input", "agent_scratchpad"]
        tools = [PythonAstREPLTool(locals={"df": df})]
        
        PREFIX = """
                You are working with a pandas dataframe in Python. The name of the dataframe is `df`.
                You should use the tools below to answer the question posed of you:"""
        
        SUFFIX = """
                This is the result of `print(df.head())`:
                {df}
                Begin!
                {chat_history}
                Question: {input}
                {agent_scratchpad}"""

        prompt = ZeroShotAgent.create_prompt(
            tools, 
            prefix=PREFIX,
            suffix=SUFFIX, 
            input_variables=["df", "input", "chat_history", "agent_scratchpad"]
        )
        
        print(prompt)
        
        partial_prompt = prompt.partial(df=str(df.head()))

        from langchain.chat_models import ChatOpenAI
        llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        #top_p=top_p,
        openai_api_key = st.secrets["openai_key"])


        
        llm_chain = LLMChain(
            llm=llm,
            prompt=partial_prompt,
            callback_manager=callback_manager,
        )
        
        tool_names = [tool.name for tool in tools]
        
        agent = ZeroShotAgent(
            llm_chain=llm_chain,
            allowed_tools=tool_names,
            callback_manager=callback_manager,
            **kwargs,
        )
        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            tools=tools,
            verbose=verbose,
            handle_parsing_errors = handle_parsing_errors,
            return_intermediate_steps=return_intermediate_steps,
            max_iterations=max_iterations,
            max_execution_time=max_execution_time,
            early_stopping_method=early_stopping_method,
            callback_manager=callback_manager,
            memory = memory
        )




#If the code has matplotlib The solution should be given using plotly and only plotly so chage it. Do not use matplotlib

def intermediate_response(answer):
    if answer["intermediate_steps"]:
        action = answer["intermediate_steps"][-1][0].tool_input["query"]
        # st.write(f"Executed the code ```{action}```")
    return answer["output"]



## streamlit run 
def generate_response(dfs, vprompt,model="gpt-3.5-turbo", temperature=0.0, max_tokens=2500, top_p=0.5,openail=True):
    import openai
    from langchain.chat_models import ChatOpenAI
    from langchain.schema.output_parser import OutputParserException
    openai.api_key = st.secrets["openai_key"]

    # prompt_temp1 = lambda x: "You are an expert in python and data analysis help with: "+x + """  If you encounter any parsing errors try to solve them on your own, as a tip check json formatting from your answets. If your response includes any plotting or code in matplotlib please execute it but using plotly instead, don't use matplotlib at all. 
    # If the response has the word fig.show remove it and append this to the code to be executed: st.plotly_chart(fig, theme='streamlit', use_container_width=True) if you do this execute the generated code with exec(code).
    # Remember all of your answers should be based on the provided dataframe list {} """.format(df)
    
    prompt_temp1 =  lambda x: x + "Remember i need you to base your answers on the provided dataframe list {}.Dont generate data on your own and dont make up anything get data only from the provided dataframe list".format(dfs)
    
    
    if not openai:
        pass
    
    """
    A function that answers data questions from a dataframe.
    """
    plot_words = ["plot", "graph", "chart", "diagram", "figure","grafica","gráfica","histogram","graficame"]
    #if "plot" in st.session_state.messages[-1]["content"].lower() or "graph" in st.session_state.messages[-1]["content"].lower():
    if any(word in st.session_state.messages[-1]["content"].lower() for word in plot_words):
        code_prompt = f"""
            Generate the code <code> for plotting the previous data from the dfs {dfs} in plotly,
            in the format requested. Note this is a list of dataframes so you need to find the relevant dataframes within the list (called dfs) intelligently and choose the appropiate one(s). 
            An example of this will be, if the relevant dataframe is the first one, you will choose it as df = dfs[0]   
            The solution should be given using plotly
            and only plotly. Do not use matplotlib.
            Return the code <code> in the following
            format ```python <code>```
        """
        

        st.session_state.messages.append({
            "role": "assistant",
            "content": vprompt +" "+code_prompt
        })
        response = openai.ChatCompletion.create(
            model=model,
            messages=st.session_state.messages,
            temperature=temperature,
            max_tokens=max_tokens,
            # top_p=top_p,
        )
        code = extract_python_code(response["choices"][0]["message"]["content"])
        if code is None:
            st.warning(
                "Couldn't find data to plot in the chat. "
                "Check if the number of tokens is too low for the data at hand. "
                "I.e. if the generated code is cut off, this might be the case.",
                icon="🚨"
            )
            return "Couldn't plot the data"
        else:
            code = code.replace("fig.show()", "")#"fig.write_html(r'plot1.html',full_html = True)")
            code.replace("""st.plotly_chart(fig, theme='plotly', use_container_width=True)""","")
            if """st.plotly_chart(""" not in code:
                code+="""st.plotly_chart(fig, theme='streamlit', use_container_width=True)""" 
            #st.write_(f"```{code}") #WRITE IT HERE?
            st.write_stream(stream_response_(response["choices"][0]["message"]["content"]))
            #imagecode = """fig.write_image(r"plot1.html",format='html',engine='kaleido')"""
            #code= code + "/n"+ imagecode
            #exec(code)

            del st.session_state.messages[len(st.session_state.messages)-1]
            return code #response["choices"][0]["message"]["content"]

    else:
        
        pandas_df_agent = get_agent(dfs)
        try:
            print("THIS IS THE PROMPT", prompt_temp1(vprompt))
            answer = pandas_df_agent(prompt_temp1(vprompt)) #pandas_df_agent(st.session_state.messages)
            if answer["intermediate_steps"]:
                return intermediate_response(answer)
            else:
                return answer['output']
        except ValueError as ve:
            import re
            pattern = r"\{.*?\}"
            match = re.search(pattern, str(ve))
            if match:
                result = match.group()
                prompt_parsed = prompt_temp1(handle_error(result))
                if "plot" or "matplotlib" in prompt_parsed:
                    answer2 = pandas_df_agent((f"Get the result of {vprompt} by using tool python_repl_ast and executing the following code : "+ prompt_parsed+
                                               "Note: dont use matplotlib use plotly and display it with this st.plotly_chart(fig, theme='streamlit', use_container_width=True)"))
                else:
                    answer2 = pandas_df_agent((f"Get the result of {vprompt} by using tool python_repl_ast and executing the following code : "+ prompt_parsed))
                if answer2["intermediate_steps"]:
                    return intermediate_response(answer2)  


def generate_response_tg(df, vprompt,tkn,model="gpt-3.5-turbo", temperature=0.0, max_tokens=2500, top_p=0.5,openail=True):
    TELEGRAM_BOT_TOKEN = tkn
    import openai
    from langchain.chat_models import ChatOpenAI
    from langchain.schema.output_parser import OutputParserException
    openai.api_key = st.secrets["openai_key"]
    prompt_temp1 =  lambda x: x + "Remember I need you to base your answers on the provided dataframe {} Dont generate data on your own and dont make up anything get data only from the provided dataframe".format(df)
    """
    A function that answers data questions from a dataframe.
    """
    plot_words = ["plot", "graph", "chart", "diagram", "figure","grafica","gráfica","histogram","graficame"]
    if any(word in prompt_temp1(vprompt).lower() for word in plot_words):
        print("FOUND PLOT WORD ")
        code_prompt = f"""
            Generate the code <code> for plotting the previous data from the dataframe {df} in plotly,
            in the format requested. The solution should be given using plotly
            and only plotly use dark background template and the plot colors should have different colors! Do not use matplotlib.
            Return the code <code> in the following
            format ```python <code>```
        """

        from openai import OpenAI
        client = OpenAI(api_key = st.secrets["openai_key"])
        response = client.chat.completions.create(model=model,
                                                  messages =[{"role": "assistant", "content": vprompt +" "+code_prompt}],
                                                  temperature = temperature,
                                                  max_tokens = max_tokens)
        print(response)

        code = extract_python_code(response.choices[0].message.content)
        print(code)
        if code is None:
            return "Couldn't plot the data 🚨 Check if the number of tokens is too low for the data at hand. I.e. if the generated code is cut off, this might be the case."
        else:
            code = code.replace("fig.show()", "") #no need to show the image 
            pathimg = str(os.getcwd()+"\\fig1.png")
            print(pathimg)
            imagecode = """fig.write_image(r"plot1.png",format='png',engine='kaleido')"""
            code = code + imagecode
            #f"fig.write_image(r{pathimg})"
            print(code)
            exec(code)
            with open("plot1.png","rb") as imageplot:
                return imageplot

    else:
        
        pandas_df_agent = get_agent(df)#create_pandas_dataframe_agent2(df=df)#
        try:
            print("THIS IS THE PROMPT", prompt_temp1(vprompt))

            answer = pandas_df_agent("Use python_repl_ast tool when needed" +prompt_temp1(vprompt)) #pandas_df_agent(st.session_state.messages)
            if answer.get("intermediate_steps"):
                #print("I will give you INTERMEDIATE STEEPS")
                import ast
                return (intermediate_response(answer))
            else:
                return answer['output']
        except ValueError as ve:
            print(ve)
            import re
            pattern = r"\{.*?\}"
            match = re.search(pattern, str(ve))
            if match:
                result = match.group()
                prompt_parsed = prompt_temp1(handle_error(result))
                if "plot" or "matplotlib" in prompt_parsed:
                    answer2 = pandas_df_agent((f"Get the result of {vprompt} by using tool python_repl_ast and executing the following code : "+ prompt_parsed+
                                               "Note: dont use matplotlib use plotly and display it with this st.plotly_chart(fig, theme='streamlit', use_container_width=True)"))
                else:
                    answer2 = pandas_df_agent((f"Get the result of {vprompt} by using tool python_repl_ast and executing the following code : "+ prompt_parsed))
                if answer2["intermediate_steps"]:
                    return intermediate_response(answer2)   
                        
                        



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





    
    