# Import the necessary libraries.
import streamlit as st
import timeit
import yaml
import argparse
from dotenv import find_dotenv, load_dotenv
from src.utils import setup_dbqa, build_retrieval_qa, set_mistral_prompt
import sys
import pandas as pd
from src.utils import set_qa_prompt, setup_dbqa, build_retrieval_qa, set_qa_emmistral_prompt
from src.llm import build_llm
from langchain.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyMuPDFLoader
import torch
import time
from langchain.callbacks import StreamlitCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
import langchain
import box

#torch.mps.empty_cache()

# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))

# App title
st.set_page_config(page_title="🦙💬 Llama 2 Chatbot")
#https://raw.githubusercontent.com/weissenbacherpwc/icons/main/logo.png

def check_strategy(): 
    strategy=""
    if choose_strategy == "Basic":
        strategy = "basic"
        return strategy
    if choose_strategy == "Compressor":
        strategy = "compressor"
        return strategy
    if choose_strategy == "Parent Document Retriever":
        strategy = "parent"
        return strategy
    if choose_strategy == "Ensemble Retriever":
        strategy == "ensemble"
        return strategy
    else:
        strategy == "basic"
    #return strategy

def check_model_path():
    model_path=""
    if model_name == "EM Mistral 7B":
        model_path = "static/em_german_mistral_v01.Q5_K_M.gguf"
        return model_path
    if model_name == "Mixtral 8x7B (MoE)":
        model_path = "/Users/mweissenba001/Documents/rag_example/Modelle/mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf"
        return model_path


col1, col2, col3 = st.columns(3)

with col1:
    st.write(' ')

with col2:
    st.title("RAG Demo")

with col3:
    st.write(' ')
#st.markdown("<h1 style='text-align: center; color: #2d2d2d;'>RAG Demo</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.image("static/PwC_fl_c.png", width=200)
    st.sidebar.write("Please Clear Conversation before changing the Model.")
    model_name = st.sidebar.radio("Choose a model:", ("EM Mistral 7B", "Mixtral 8x7B (MoE)"), index=None)
    choose_strategy = ""
    choose_data=""
    if model_name:
        choose_strategy = st.sidebar.radio("Choose a RAG strategy:", ("Basic", "Compressor", "Parent Document Retriever", "Ensemble Retriever"),index=None)
        choose_data = st.sidebar.radio("Do you want to upload a document by your own or do you want to use the Confluence data?", ("Own Document", "Confluence data", "No Rag - LLM Chat"), index=None)
        strategy = check_strategy()
        model_path = check_model_path()
    
st.sidebar.write(f"You chose Model: {model_name} and Strategy: {choose_strategy}")
    
# # # Allow the user to upload a file with supported extensions.
# if choose_data == "Own Document":
#     uploaded_file = st.file_uploader("Upload an article (max 20 pages)", type=("pdf"))
#     if uploaded_file:
#         # Save the uploaded file locally.
#         with open(uploaded_file.name, "wb") as f:
#             f.write(uploaded_file.getbuffer())
#         #st.write(uploaded_file)
#         #st.write(uploaded_file.type)
#         st.write("Creating Vectordatabase...")
#         loader = PyMuPDFLoader(uploaded_file.name)
#         data = loader.load()
#         embeddings = HuggingFaceEmbeddings(model_name=cfg.EMBEDDING_MODEL_NAME,
#                                         model_kwargs={'device': 'mps'})

#         vectorstore = FAISS.from_documents(data, embeddings) 
#         if vectorstore:
#             st.write("Vector Database created!")
#         else:
#             st.write("Vector Database building...")
#     else:
#         st.write("Please choose a file!")

# # Store LLM generated responses
# if "messages" not in st.session_state.keys():
#     st.session_state.messages = [{"role": "assistant", "content": "Hallo, ich kann dir Fragen zum IPv6 Confluence beantworten."}]

# # Display or clear chat messages   
# def clear_chat_history():
#     torch.mps.empty_cache()
#     st.session_state.messages = [{"role": "assistant", "content": "Hallo, ich kann dir Fragen zum IPv6 Confluence beantworten."}]
# st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# for message in st.session_state.messages:
#     if message["role"] == "user":
#         avatar = "static/Person_Fill_Black_RGB.svg"
#     if message["role"] == "assistant":
#         avatar = "static/Bot_Fill_DigitalRose_RGB.svg"
#     with st.chat_message(message["role"], avatar=avatar):
#         st.write(message["content"])
        
# if prompt := st.chat_input(disabled=not model_name): 
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user", avatar="static/Person_Fill_Black_RGB.svg"):
#         st.write(prompt)

# @st.cache_resource
# def setup_model_dbqa():
#     dbqa = setup_dbqa(strategy=strategy, model_path=model_path)
#     return dbqa
    
# @st.cache_resource
# def setup_llm():
#     llm = build_llm(model_path=model_path)
#     return llm
                   

# # Generate a new response if last message is not from assistant
# if st.session_state.messages[-1]["role"] != "assistant":
#     # um zu verhindern, dass OOM Fehler kommt erst definieren
#     with st.chat_message("assistant", avatar="static/Bot_Fill_DigitalRose_RGB.svg"):
#         with st.spinner("Thinking..."):
#             if choose_data == "Confluence data":
#                 st.write(model_path)
#                 placeholder_stream = st.empty()
#                 st_callback = StreamlitCallbackHandler(placeholder_stream)
#                 #langchain.debug = True
#                 start_time = time.time()
#                 dbqa = setup_model_dbqa()
#                 answer = dbqa(prompt, callbacks=[st_callback])
#                 print(f"Chain History: {dbqa.combine_documents_chain.memory}")
#                 end_time = time.time()
#                 time_ = end_time - start_time
#                 st.write(f"Response Time: {time_}")
#                 answer_text = answer["result"]
#                 placeholder_stream = placeholder_stream.empty()
#                 placeholder = st.empty()
#                 placeholder.markdown(answer_text)
#                 source_docs = answer['source_documents']
#                 for i, doc in enumerate(source_docs):
#                     st.write(f'\nSource Document {i+1}\n')
#                     st.write(f'Document Name: {doc.metadata["source"]}')
#                     st.write('='* 60)
#                 torch.mps.empty_cache()
                    
                    
#             if choose_data == "Own Document":
#                 if model_name == "Mixtral 8x7B (MoE)":
#                     placeholder_stream = st.empty()
#                     st_callback = StreamlitCallbackHandler(placeholder_stream)
#                     qa_prompt = set_mistral_prompt()
#                     llm = setup_llm()
#                     mixtral_qa = build_retrieval_qa(llm=llm,prompt=qa_prompt,vectordb=vectorstore)
#                     answer = mixtral_qa(prompt, callbacks=[st_callback])
#                     answer_text = answer["result"]
#                     placeholder_stream = placeholder_stream.empty()
#                     placeholder = st.empty()
#                     placeholder.markdown(answer_text)
#                     source_docs = answer['source_documents']
#                     for i, doc in enumerate(source_docs):
#                         st.write(f'\nSource Document {i+1}\n')
#                         st.write(f'Document Name: {doc.metadata["source"]}')
#                         st.write('='* 60)
                        
                        
#             if choose_data == "No Rag - LLM Chat":
#                 # Um hier Memory zu etablieren, muss ich eine eigene Chain aufsetzten, mit einem eigenen Prompt.
#                 start_time = time.time()
#                 llm = setup_llm()
#                 placeholder = st.empty()                
#                 st.write(model_path)
#                 st_callback = StreamlitCallbackHandler(placeholder)
#                 answer_text = llm(prompt, callbacks=[st_callback])
#                 end_time = time.time()
#                 time_ = end_time - start_time
#                 st.write(time_)
#                 placeholder.markdown(answer_text)
                
                
                
#     message = {"role": "assistant", "content": answer_text}

#     st.session_state.messages.append(message)
    
    
#     # Ganzes Document hinzufügen, also Vectordb erstellen etc.
#     # Antworte sehr knapp: Was ist die Hauptstadt von Österreich?
#     # 
