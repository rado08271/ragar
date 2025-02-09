import json
import os
from rich import print

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages import trim_messages

from langchain.text_splitter import TokenTextSplitter

with open("ragar.config", "r") as f:
    config = json.load(f)


api_key = config.get("model", {}).get("api_key", "")
model_name = config.get("model", {}).get("name", "openai")
temperature = config.get("model", {}).get("temperature", 0.0)

# -------------------------------------------------------------------------------------------------
# 1. Setup Model
# -------------------------------------------------------------------------------------------------

# 1.1. Setup Model and temperature
model = ChatOpenAI(api_key=api_key, model="gpt-4o-mini", temperature=temperature)


has_context = config.get("context", {}).get("enabled", False)
has_memory = config.get("memory", {}).get("enabled", False)
has_debug = config.get("debug", {}).get("enabled", False)


# -------------------------------------------------------------------------------------------------
# 2. Setup Context Retrieval    
# -------------------------------------------------------------------------------------------------
retriever = None
if (has_context):
    file_path = config.get("context", {}).get("file", "")
    chunk_size = config.get("context", {}).get("tokens_chunk_size", 512)
    chunk_overlap = config.get("context", {}).get("overlap", 50)
    embedding = config.get("context", {}).get("embedding", "openai")
    retrieval_k = config.get("context", {}).get("retrieval_k", 5)

    # 2.1. Setup Context Retrieval
    loader = PyPDFLoader(file_path)
    # if we want to split the documents into chunks using default behavior, we can use loader.load_and_split()
    docs = loader.load()
    
    # 2.2. Setup Context Reduction
    text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(docs)

    # 2.3. Setup Context Embedding
    embeddings = OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-small")
    # 2.4. Setup Context Vector Store
    vector_store = Chroma(
        collection_name="ragar",
        embedding_function=embeddings,
        persist_directory="./ragar.db"
    )
    vector_store.add_documents(docs)

    # 2.5. Setup Context Retrieval
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": retrieval_k})

else:
    print("[red]Context ingestion retrieval is disabled.[/red]")

# -------------------------------------------------------------------------------------------------
# 3. Setup Memory
# -------------------------------------------------------------------------------------------------
def summarize_messages(messages):
    summary_prompt = (
        "Write a concise summary of the following:"
        "{messages}"
        "Summary:"
    )
    summary_prompt_fmt = summary_prompt.format(messages=messages)
    return model.invoke(summary_prompt_fmt)

def truncate_messages(messages):
    trimmer = trim_messages(strategy="last", max_tokens=160, token_counter=len)
    return trimmer.invoke(messages)


memory = None
chat_history = []
if (has_memory):
    # 3.1. Setup Memory
    reduction_method = config.get("memory", {}).get("reduction_method", "truncation")
    if (reduction_method == 'truncation'):
        memory = truncate_messages
    elif (reduction_method == 'summarization'):
        memory = summarize_messages

else:
    print("[red]Memory is disabled.[/red]")


# -------------------------------------------------------------------------------------------------
# 4. Setup Debug
# -------------------------------------------------------------------------------------------------
if (has_debug):
    # 4.1. Setup Debug
    debug_api_key = config.get("debug", {}).get("api_key", "")
    project_name = config.get("debug", {}).get("project_name", "default")
    os.environ["LANGSMITH_API_KEY"] = debug_api_key
    os.environ["LANGSMITH_PROJECT"] = project_name
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
else:
    print("[red]Debug is disabled.[/red]")


system_prompt = (
    "You are a helpful assistant for questions answering."
)
if (has_context and not has_memory):
    system_prompt = (
        "You are a helpful assistant that can answer questions about the context provided."
        "You have the following context:"
        "{context}"
    )
elif (has_memory and not has_context): 
    system_prompt = (
        "You are a helpful assistant that can answer questions about the chat history provided."
        "You have the following chat history:"
        "{chat_history}"
    )
elif (has_context and has_memory):
    system_prompt = (
        "You are a helpful assistant that can answer questions about the context and chat history provided."
        "You have the following context:"
        "{context}"
        "You have the following chat history:"
        "{chat_history}"
    )

system_prompt += (
    "If you don't know the answer, just say that you don't know. Don't make up an answer. You have the following user prompt:"
    "{user_prompt}"
    ""
    "Answer:"
)

# def summarize_history(state: MessagesState):
def chat_prompt(input: str):
    
    context_text = ""
    if (has_context):
        context = retriever.invoke(input)
        context_text = "\n".join([doc.page_content for doc in context])

    history_text = ""
    if (has_memory):
        if (len(chat_history) > 0):
            history = "\n".join([message.content for message in chat_history])
            history_text = memory(history)
    
    system_prompt_fmt = system_prompt.format(user_prompt=input, context=context_text, chat_history=history_text)

    output_message = model.invoke([SystemMessage(content=system_prompt_fmt), HumanMessage(content=input)])
    
    chat_history.append(HumanMessage(content=input))
    chat_history.append(output_message)

    return (output_message.content, output_message.response_metadata["token_usage"]["total_tokens"])

