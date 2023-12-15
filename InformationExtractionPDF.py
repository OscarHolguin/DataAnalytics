# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 08:17:46 2023

@author: o_hol
"""
from ParsingPDF import preprocess_text

import pandas as pd
import uuid
import spacy
import numpy as np
nlp = spacy.load("en_core_web_sm")

def extract_info(chunks):
    rows = []
    for page in chunks:
        row = {"text": preprocess_text(page.page_content),**page.metadata,"chunk_id": uuid.uuid4().hex}
        rows+= [row]
    
    return pd.DataFrame(rows)


def dfText2DfNE(dataframe,huggingface=False,openai=False,spacyf=False):
    
    if spacyf:
        nlp = spacy.load("en_core_web_sm")

        entities_list = []
        for doc, chunk_id in zip(nlp.pipe(dataframe.text),dataframe['chunk_id']):
            metadata = {"chunk_id": chunk_id}
            for ent in doc.ents:
                entities_list.append({"name":ent.text,"entity":ent.label_,**metadata})
        
            
        entities_df = pd.DataFrame(entities_list).replace(' ',np.nan)
        entities_df = entities_df.dropna(subset=['entity'])
        entities_df = entities_df.groupby(['name','entity','chunk_id']).size().reset_index(name="count")
        
        
        return entities_df
        

    
    elif huggingface:
        from transformers import pipeline
        ner = pipeline("token-classification", model = "dslim/bert-large-NER", aggregation_strategy = "simple")
        
        def row2NER(row):
            ner_results = ner(row['text'])
            metadata = {"chunk_id": row["chunk_id"]}
            entities = []
            for result in ner_results:
                entities = entities+[{"name": result["word"], "entity": result["entity_group"], **metadata}]
            return entities
        
        
        results = dataframe.apply(row2NER,axis = 1)
        
        entities_list = np.concatenate(results).ravel().tolist()
        entities_df = pd.DataFrame(entities_list).replace(' ',np.nan)
        entities_df = entities_df.dropna(subset=['entity'])
        entities_df = entities_df.groupby(['name','entity','chunk_id']).size().reset_index(name="count")
        
        
        return entities_df
        
        
        
    elif openai:
        raise 'Not implemented'
    else:
        raise "ERROR invalid argument"
        
    # entities_df = pd.DataFrame(entities_list).replace(' ',np.nan)
    # entities_df = entities_df.dropna(subset=['entity'])
    # entities_df = entities_df.groupby(['name','entity','chunk_id']).size().reset_index(name="count")
    
    
    # return entities_df
    
    
    
def networkgraph(df,**kwargs):
    import networkx as nx
    from pyvis.network import Network
    g = nx.from_pandas_edgelist(df,source="name",target="entity")
    # d = df.groupby("entity")["count"].sum().to_dict()
    
    # for node in g.nodes:
    #     d.setdefault(node,1)
    # nodes,values = zip(*d.items())
    
    outdir = "graph.html"
    net = Network(notebook = False, cdn_resources = "in_line", height = "900ptx", width = "100%", select_menu = True, filter_menu = False, **kwargs)
    net.from_nx(g)
    net.repulsion(
        node_distance=420,
        central_gravity=0.33,
        spring_length=110,
        spring_strength=0.10,
        damping=0.95
                       )
    net.force_atlas_2based(central_gravity =  0.015,gravity = -31)
    
    net.save_graph('entities_graph.html')
        
    
    # net.show_buttons(filter_ = ["physics"])
    # net.show(outdir,notebook=False)
    return net