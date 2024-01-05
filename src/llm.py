'''
===========================================
        Module: Open-source LLM Setup
===========================================
'''
#from langchain.llms import CTransformers
from dotenv import find_dotenv, load_dotenv
import box
import yaml
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


def build_llm(model_path):
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        
        n_gpu_layers = 1 # Metal set to 1 is enough. # ausprobiert mit mehreren
        n_batch = 1024 # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
        
        if model_path == "/Users/mweissenba001/Documents/llama2/llama.cpp/models/7B/ggml-model-q4_0.bin":
                context_size = 4000
        elif model_path == "/Users/mweissenba001/Documents/rag_example/Modelle/llama-2-13b-german-assistant-v2.Q5_K_M.gguf":
                context_size = 4000
        elif model_path == "/Users/mweissenba001/Documents/rag_example/Modelle/mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf":
                context_size = 28000
        elif model_path =="/Users/mweissenba001/Documents/rag_example/Modelle/mixtral-8x7b-instruct-v0.1.Q5_0.gguf":
                context_size = 28000
        elif model_path == "/Users/mweissenba001/Documents/rag_example/Modelle/mixtral-8x7b-instruct-v0.1.Q6_K.gguf":
                context_size = 28000
        else:
                context_size = 7000
                
        print(f"Selected Context: {context_size}")

        llm = LlamaCpp(
                max_tokens =cfg.MAX_TOKENS,
                n_threads = 7,# für performance
                model_path=model_path,
                temperature=cfg.TEMPERATURE,
                f16_kv=True,
                n_ctx=context_size, # 8k aber mann muss Platz lassen für Instruction, History etc. 
                n_gpu_layers=n_gpu_layers,
                n_batch=n_batch,
                callback_manager=callback_manager, 
                verbose=True, # Verbose is required to pass to the callback manager
                top_p=0.75,
                top_k=40,
                repeat_penalty = 1.1,
                streaming=True,
                #stream=True, # stream oder streaming
                model_kwargs={
                        #'repetition_penalty': 1.1,
                        'mirostat': 2,
                },
        )
        
        return llm