# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 14:55:27 2023

@author: o_hol


CHAT ENABLED PDF VERSION
"""



import asyncio



import plotly_express as px
import streamlit as st
import pandas as pd
from langchain.document_loaders import PyPDFLoader, UnstructuredPDFLoader, OnlinePDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
import pyodbc
from langchain.vectorstores import Chroma
import urllib
import pylab
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
from ParsingPDF import preprocess_text, PDFParser,get_pdf_url
from InformationExtractionPDF import extract_info, dfText2DfNE,networkgraph
import base64
import streamlit.components.v1 as components


from streamlit_d3graph import d3graph
#import d3graph

import networkx as nx


from dataclasses import dataclass
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
import uuid
import spacy


#######################
#THIS IS FOR TESTING
sqlServer = {'sqlServerName': 'euwdsrg03rsql01.database.windows.net',
 'sqlDatabase': 'EUWDSRG03RRSG02ADB01_Copy',
 'userName': 'dbWSS',
 'password': 'braf0wNVtixu3?IhU=hmrCeLzmzX>Wlo'}




# PROD_MxD_PDM_DeviceFailureV2DataTable
sqlServerName,sqlDatabase,userName,password = sqlServer.get('sqlServerName'),sqlServer.get('sqlDatabase'),sqlServer.get('userName'), sqlServer.get('password')



os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'


############################
from reports_template import Reports
#
from datachat import generate_response,generate_responsedf,generate_insights_one,generate_trends_and_patterns_one,aggregate_data

reports = Reports()

import hmac


# def check_password():
#     """Returns `True` if the user had the correct password."""

#     def password_entered():
#         """Checks whether a password entered by the user is correct."""
#         if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
#             st.session_state["password_correct"] = True
#             del st.session_state["password"]  # Don't store the password.
#         else:
#             st.session_state["password_correct"] = False

#     # Return True if the passward is validated.
#     if st.session_state.get("password_correct", False):
#         return True

#     # Show input for password.
#     st.text_input(
#         "Password", type="password", on_change=password_entered, key="password"
#     )
#     if "password_correct" in st.session_state:
#         st.error("😕 Password incorrect")
#     return False


# if not check_password():
#     st.stop()  # Do not continue if check_password is not True.


company = "CompanyA"
pagetitle = "Auto Analytics "


st.set_page_config(page_title = pagetitle,
                   #page_icon = "https://djnsalesianos.mx/wp-content/uploads/2019/09/salesians-100x100.jpg",
                   layout="wide",
                   initial_sidebar_state="expanded", 
    )





st.title('Data Analysis')
#st.image('https://djnsalesianos.mx/wp-content/uploads/2019/04/logodjnnuevo.png',width=800)


#@st.cache_data(ttl=10)
@st.cache_resource
def read_sql(sqlServerName ,sqlDatabase,userName,password,tablename,sqlPort = 1433,query =None,pandas=False)->pd.DataFrame:
    query = query if query else f"(SELECT * FROM {tablename})"# AS subquery"
    try:
        if pandas:
            cnxn = pyodbc.connect(DRIVER="{ODBC Driver 17 for SQL Server}", SERVER=sqlServerName, DATABASE=sqlDatabase, UID=userName, PWD=password, STORE_DRVRESULTS=0)
            df = pd.read_sql(query,cnxn)
            return df
        #df.write.jdbc(sqlServerUrl, "PDM_AD_PredictionTable", write_mode, connectionProperties)
        else:
            connstr = f"DRIVER=ODBC Driver 17 for SQL Server, SERVER={sqlServerName}, DATABASE={sqlDatabase}, UID={userName}, PWD={password}, STORE_DRVRESULTS=0"
            connection_string = urllib.parse.quote_plus(connstr)
            connection_string = "mssql+pyodbc:///?odbc_connect=%s" % connection_string
            return connection_string
    except Exception as e:
        return str(e)


sqldf = lambda table: read_sql(sqlServerName ,sqlDatabase,userName,password,table,sqlPort = 1433,pandas = True)
tables = ["PROD_MxD_PDM_DeviceFailureV2DataTable","PROD_MxD_PDM_DeviceFailureV2PredictionTable","PROD_MxD_DDM_AssetDataTable","PROD_MxD_DDM_DowntimeDataTable"]

# dfs=[sqldf(table) for table in tables]



left,mid,right = st.columns([1,3,1],gap='large')




# st.sidebar.header('Seleccion de datos')

@st.cache_resource
def st_display_sweetviz(report_html,width=2000,height=2000):
	report_file = codecs.open(report_html,'r')
	page = report_file.read()
    #components.html(page,scrolling =True)
	components.html(page,width=width,height=height,scrolling=True)
    

# #default df but maybe load file(?)
#df = dfs[0][["WindowTimeStamp_Start","WindowTimeStamp_End","DeviceName","Vibration","Voltage","AirPressure","Current_amps"]]
#@st.cache_resource


data_file = st.file_uploader("Choose between CSV or PDF",type=['csv','pdf'])
file_ext = option_chosen = "null"
if data_file is not None:
    file_name = data_file.name
    file_ext = file_name.split(".")[-1]





@st.cache_resource
def get_pyg_renderer(df) -> "StreamlitRenderer":
    # When you need to publish your app to the public, you should set the debug parameter to False to prevent other users from writing to your chart configuration file.
    return StreamlitRenderer(df,debug=False, use_kernel_calc=True,dark='dark',height = 1000)#, spec="./gw_config.json", debug=False)

@st.cache_data(ttl=30)
def generate_ner(_documents,ntype):
        df = extract_info(_documents)
        if ntype == 'huggingface':
            dfner = dfText2DfNE(df,huggingface=True)
        elif ntype =='spacy':
            dfner = dfText2DfNE(df,spacyf=True)
        dfne = dfner.groupby(['name','entity']).agg({"count":"sum","chunk_id":",".join}).reset_index()
        dfne.sort_values(by="count",ascending=False).reset_index()
        return dfne



@st.cache_resource()
def get_network(df,outname = "graph.html"):
    import networkx as nx
    from pyvis.network import Network
    g = nx.from_pandas_edgelist(df,source="name",target="entity")
    
    #net = #Network(notebook = False, cdn_resources = "in_line", height = "900ptx", width = "100%", select_menu = True, filter_menu = False)
    net =  Network(
                   height='3000px',
                   width='100%',
                   bgcolor='#222222',
                   font_color='white',
                   #select_menu = True,
                   #filter_menu = True
                  )
    net.from_nx(g)
    net.repulsion(
        node_distance=420,
        central_gravity=0.33,
        spring_length=110,
        spring_strength=0.10,
        damping=0.95
                       )
    net.force_atlas_2based(central_gravity =  0.015,gravity = -31)
    
    net.save_graph(outname)


@st.cache_resource(experimental_allow_widgets=True)
def profile_report(df,**kwargs):
    r2 = reports.create_profiling_report(df, explorative=True,progress_bar = True,correlations = correlations,title="REPORT",)# config_file = 'config.yaml')
    st_profile_report(r2, navbar=True,key = "Report")

@st.cache_resource
def generate_wordcloud(text):
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    text = preprocess_text(text)
    wordcloud = WordCloud().generate(text)
    return wordcloud





    


if file_ext =='csv':
    tmppath = os.path.join("/tmp", data_file.name)
    response_history = st.session_state.get("response_history", [])

    st.sidebar.header('Select automatic report style')

    df= pd.read_csv(data_file,encoding ="utf-8")
    st.session_state.df = df
    st.dataframe(df.head())
    menu = ["Home","Report","Retro_Report","Create"]
    chat_toggle = st.sidebar.toggle("Chat")
    index = 0 if chat_toggle else None # Set index to 0 if toggle is True, otherwise None
    
    option_chosen = st.sidebar.selectbox("Report Style:", menu,index)
    option_chosen = "Home" if option_chosen == None else option_chosen

    
    import streamlit.components.v1 as components 
    from streamlit_ydata_profiling import st_profile_report
    
    if option_chosen.lower()=='retro_report':
        chat_toggle = False

        r1 = reports.retro_report(df)
        r1.show_html(filepath='./EDA.html', open_browser=False, layout='vertical')#, scale=1.0)
        st_display_sweetviz("EDA.html")
        #components.iframe(src='./EDA.html', scrolling=True) #width=1100, height=1200,
    
        #r1.show_html()
        
    
    elif option_chosen.lower()=='report':
            chat_toggle = False

            correlations={
            "auto": {"calculate": True},
            "pearson": {"calculate": True},
            "spearman": {"calculate": True},
            "kendall": {"calculate": True},
            "phi_k": {"calculate": True},
            "cramers": {"calculate": True},
        }
            
            
            checkbox_labels =[d for d in correlations.keys()]
            checkbox_values = [False]*len(checkbox_labels)
            st.sidebar.header("Select correlations:")

            # Create a loop to create the checkboxes in the sidebar
            for i, label in enumerate(checkbox_labels):
            # Create a checkbox with the label and value
                checkbox_values[i] = st.sidebar.checkbox(label, value=checkbox_values[i], key=i)
                correlations[label]["calculate"] = checkbox_values[i]
                
            profile_report(df,correlations = correlations)

    
    
    
    elif option_chosen.lower() == "create":
        chat_toggle = False

        from pygwalker.api.streamlit import StreamlitRenderer, init_streamlit_comm
        

        pyg_html = pyg.to_html(df) 
        # # Embed the HTML into the Streamlit app
        components.html(pyg_html, height=1000, scrolling=True)
    
    ##chat section
    
    if st.session_state.df is not None and chat_toggle:
        st.session_state.prompt_history = []

        if "messages" in st.session_state:
            print('messages found')
            
        else:
           if "messages" not in st.session_state.keys():
               st.session_state.messages = [{"role": "assistant", "content": "Hi! How can I help you with your data?"}]
        
        if "messages" in st.session_state:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
            if prompt := st.chat_input():
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.write(prompt)

            
            if st.session_state.messages[-1]["role"] != "assistant":
                if prompt:
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            response =  generate_responsedf(df,prompt)
                            try:
                                st.pyplot(response)
                            except:
                                st.write(response)
                            message = {"role": "assistant", "content": response}
                            st.session_state.messages.append(message) 
                            if "insights" in prompt.lower():
                                insights = generate_insights_one(st.session_state.df)
                                st.write(insights)
                            elif "trends" in prompt.lower() or "patterns" in prompt.lower():
                                trends_and_patterns = generate_trends_and_patterns_one(st.session_state.df)
                                for fig in trends_and_patterns:
                                    if fig is not None:
                                        st.pyplot(fig)
                            elif "aggregate" in prompt.lower():
                                columns = prompt.lower().split("aggregate ")[1].split(" and ")
                                aggregated_data = aggregate_data(st.session_state.df, columns)
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
                    
        st.sidebar.subheader("Prompt history:")
        st.sidebar.write(st.session_state.prompt_history)
        
        #st.sidebar.subheader("Prompt response:")
        #for response in response_history:
        #    st.write(response)
 
        if st.sidebar.button("Clear"):
            st.session_state.prompt_history = []
            st.session_state.response_history = []
            st.session_state.df = None
        
        if st.sidebar.button("Save Results", key=0):
            with open("historical_data.txt", "w") as f:
                for response in response_history:
                    f.write(response + "\n")
            if fig is not None:
                fig.savefig("plot.png")  
                    
   

      #PANDAS AI CHAT  
      # st.subheader("Peek into the uploaded dataframe:")
      # st.write(st.session_state.df.head(2))
    
      # with st.form("Question"):
      #     question = st.text_area("Question", value="", help="Enter your queries here")
      #     answer = st.text_area("Answer", value="")
      #     submitted = st.form_submit_button("Submit")
      #     if submitted:
      #         
      # with st.spinner():
      #   llm = OpenAI(api_token=st.session_state.openai_key)
      #   pandas_ai = PandasAI(llm)
      #   x = pandas_ai.run(st.session_state.df, prompt=question)
      # fig = plt.gcf()
      # fig, ax = plt.subplots(figsize=(10, 6))
      # plt.tight_layout()
      # if fig.get_axes() and fig is not None:
      #   st.pyplot(fig)
      #   fig.savefig("plot.png")
#      st.write(x)
#      st.session_state.prompt_history.append(question)
#      response_history.append(x) # Append the response to the list
#      st.session_state.response_history = response_history


elif file_ext =='pdf':
    st.sidebar.header("PDF options")
    extractimgs = st.sidebar.checkbox("extract_images")
    imgs = False
    if extractimgs:
        imgs = True
    bytes_data = data_file.read()  # read the content of the file in binary
    tmppath = os.path.join("/tmp", data_file.name)
    with open(tmppath, "wb") as f:
        f.write(bytes_data)
    parser = PDFParser(tmppath,extract_images= imgs, max_tokens = 1048, chunk_overlap=64)
    documents = parser.parse_pdf()
    alltext = [d.page_content for d in documents]
    alltext2 = preprocess_text(' '.join(alltext))
    wc = generate_wordcloud(' '.join(alltext))
    fig, ax = plt.subplots(figsize = (12, 8))
    ax.imshow(wc)
    plt.axis("off")
    st.sidebar.pyplot(fig,use_container_width=True)
    # st.image(wc, use_column_width=True)

    #st.pyplot()
    
    
    if st.sidebar.toggle("File preview"):
        #for doc in documents:
        pdf_url = get_pdf_url(data_file)
        embed_code = f'<iframe src="{pdf_url}" width="80%" height="1000px"></iframe>'
        st.markdown(embed_code, unsafe_allow_html=True)
    
    if st.sidebar.toggle("Generate NER"):
        
        nertype = st.radio(
            "Select NER extraction type",
            ["spacy","huggingface"],
        captions = ["spacy model for NER","uses distil bert ner"])
    
        dfne = generate_ner(documents,nertype)
        dfne10 = dfne.sample(10)
        st.dataframe(dfne10)
        
        #network graphs
        outname = "entity.html"
        get_network(dfne,outname = outname)
        HtmlFile = open("entity.html", 'r', encoding='utf-8')
        
        components.html(HtmlFile.read(), height=1000)
        
        


