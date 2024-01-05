'''
===========================================
        Module: Util functions
===========================================
'''
import box
import yaml
import torch
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from src.prompts import qa_template, em_mistral_template, mistral_prompt
from src.llm import build_llm
from langchain.memory import ConversationBufferMemory
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import BM25Retriever, EnsembleRetriever
import pickle
from langchain.storage._lc_store import create_kv_docstore
from langchain.storage import LocalFileStore
from langchain.chains.conversation.memory import ConversationSummaryMemory
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.prompts import ChatMessagePromptTemplate, PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.load import dumps, loads
from langchain.schema.runnable import RunnablePassthrough
from src.text_preprocess import clean_document, clean_document_confluence
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.document_transformers import (
    LongContextReorder,
    EmbeddingsClusteringFilter
)
from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.text_splitter import CharacterTextSplitter
from langchain.retrievers.merger_retriever import MergerRetriever

# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


def set_qa_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=qa_template,
                            input_variables=['context','chat_history', 'question'])
    return prompt

def set_qa_emmistral_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=em_mistral_template,
                            input_variables=['context','chat_history', 'question'])
    return prompt

def set_mistral_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=mistral_prompt,
                            input_variables=['chat_history', 'context', 'question'])
    return prompt


def build_retrieval_qa(llm, prompt, vectordb, chain_type="RetrievalQA"):
    chain_type_kwargs={
        #"verbose": True,
        "prompt": prompt,
        "memory": ConversationBufferWindowMemory(
            memory_key="chat_history",
            input_key="question",
            #output_key="answer",
            k=8,
            return_messages=True),
        "verbose": False
        }
    
    if chain_type == "RetrievalQA":
        dbqa = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type='stuff',
                                        retriever=vectordb.as_retriever(search_kwargs={'k': cfg.VECTOR_COUNT, 'score_treshold': cfg.SCORE_TRESHOLD}, search_type="similarity"), # search_type="mmr" for Similarity AND diversity
                                        return_source_documents=cfg.RETURN_SOURCE_DOCUMENTS,
                                        chain_type_kwargs=chain_type_kwargs,
                                        verbose=False
                                        )
    return dbqa

def build_compression_retrieval_qa(llm, prompt, vectordb):
    chain_type_kwargs={
        #"verbose": True,
        "prompt": prompt,
        "memory": ConversationBufferMemory(
            memory_key="chat_history",
            input_key="question",
            return_messages=True)}
    
    # Approach with ConversationSummaryMemory
    # chain_type_kwargs={
    #     #"verbose": True,
    #     "prompt": prompt,
    #     "memory": ConversationSummaryMemory(
    #         llm=llm,
    #         memory_key="history",
    #         input_key="question")}
    
    dbqa = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=vectordb,
                                       return_source_documents=cfg.RETURN_SOURCE_DOCUMENTS,
                                       chain_type_kwargs=chain_type_kwargs,
                                       verbose=False
                                       )
    return dbqa


def big_chunk_retriever_build():
    #loader = DirectoryLoader(cfg.DATA_PATH,
    #                         glob='*.pdf',
    #                         loader_cls=PyPDFLoader)
    #documents = loader.load()
    with open(f'confluence_docs.pkl', 'rb') as documents:
        documents = pickle.load(documents)
    
    # Preprocess the text before chunking
    documents = clean_document_confluence(documents)
    embeddings = HuggingFaceEmbeddings(model_name=cfg.EMBEDDING_MODEL_NAME,
                                       model_kwargs={'device': 'mps'}, encode_kwargs={'device': 'mps', 'batch_size': 32})
    
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
    #store = InMemoryStore()
    fs = LocalFileStore("vectorstore/parent_retriever/") #here are the parent chunks
    store = create_kv_docstore(fs)
    vectorstore = Chroma(collection_name="split_parents", embedding_function=embeddings,
                         persist_directory="vectorstore/chroma_db/") #here are the child chunks
    big_chunks_retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    big_chunks_retriever.add_documents(documents)
    
    return big_chunks_retriever

    

def setup_dbqa(strategy, model_path): #model name added for evaluation
    print(f"Strategy: {strategy}")
    device = torch.device('mps')
    embeddings = HuggingFaceEmbeddings(model_name=cfg.EMBEDDING_MODEL_NAME,
                                        model_kwargs={'device': device},
                                        encode_kwargs={'normalize_embeddings': True})
    vectordb = FAISS.load_local(cfg.DB_FAISS_PATH, embeddings)
    llm = build_llm(model_path)
    if model_path == "/Users/mweissenba001/Documents/rag_example/Modelle/mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf":
        qa_prompt = set_mistral_prompt()
    else:
        qa_prompt = set_qa_emmistral_prompt()
    #qa_prompt = set_qa_prompt()
    
    if strategy == "basic":
        dbqa = build_retrieval_qa(llm, qa_prompt, vectordb)
        return dbqa
        
    if strategy == "compressor":
        # Works better with larger Chunks e.g. 1500
        retriever= vectordb.as_retriever(search_kwargs={'k': cfg.VECTOR_COUNT, 'score_treshold': cfg.SCORE_TRESHOLD})
        compressor = LLMChainExtractor.from_llm(llm)
        compression_retriever = ContextualCompressionRetriever(base_compressor=compressor,base_retriever=retriever)
        dbqa = build_compression_retrieval_qa(llm, qa_prompt, compression_retriever)
        return dbqa
        
    if strategy == "ensemble":
        #to read bm25 object
        with open(f'{cfg.BM25_PATH}/bm25.pkl', 'rb') as bm25result_file:
            bm25_retriever = pickle.load(bm25result_file)
        bm25_retriever.k = 2
        faiss_retriever=vectordb.as_retriever(search_kwargs={'k': cfg.VECTOR_COUNT, 'score_treshold': cfg.SCORE_TRESHOLD})
        # initialize the ensemble retriever
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
        )
        dbqa = build_compression_retrieval_qa(llm, qa_prompt, ensemble_retriever)
        return dbqa
        
    if strategy == "parent":
        retriever = big_chunk_retriever_build()
        dbqa = build_compression_retrieval_qa(llm, qa_prompt, retriever)
        return dbqa
    
    if strategy == "lotr":
        retriever= vectordb.as_retriever(search_type="similarity",search_kwargs={'k': 2, 'score_treshold': 0.7})
        retriever_mmr = vectordb.as_retriever(search_type="mmr",search_kwargs={'k': 2, 'score_treshold': 0.7})
        # The Lord of the Retrievers will hold the output of both retrievers and can be used as any other
        # retriever on different types of chains.
        lotr = MergerRetriever(retrievers=[retriever, retriever_mmr])

        reordering = LongContextReorder()
        redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
        relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.7)
        pipeline_compressor = DocumentCompressorPipeline(     
                        transformers=[
                            redundant_filter,
                            relevant_filter,
                            reordering
                        ]
                    )
        compression_retriever = ContextualCompressionRetriever(base_compressor=pipeline_compressor,base_retriever=lotr)
        dbqa = build_compression_retrieval_qa(llm, qa_prompt, compression_retriever)
        return dbqa
                       