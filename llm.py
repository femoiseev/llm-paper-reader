from operator import itemgetter
import sys
import os

from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms.llamacpp import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from langchain.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser

paper_url = "https://arxiv.org/pdf/2306.02516.pdf"
emb_model_name = "all-MiniLM-L6-v2"
llm_path = "./models/mistral-7b-instruct-v0.2.Q4_0.gguf"
system_prompt = "You are AI assistant who helps humans understand scientific papers. With each request, you receive few relevant chunks from the paper (each of them is starts with 'Chunk \{i\}', where i is the chunk number), and then user question prepended by 'User question: '. Your task is to give a clear and concise answer to the user question based on provided chunks from the paper."


def format_chunks(chunks):
    return "\n".join([f"Chunk {i + 1}: {d.page_content}" for i, d in enumerate(chunks)])


def create_db(paper_url: str, emb_model_name: str):
    loader = PyPDFLoader(paper_url)
    splitter = SentenceTransformersTokenTextSplitter(
        chunk_overlap=50, model_name=emb_model_name
    )

    sys.stderr = open(os.devnull, "w")
    chunks = loader.load_and_split(text_splitter=splitter)
    sys.stderr = sys.__stderr__

    emb_model = HuggingFaceEmbeddings(model_name=emb_model_name)
    return FAISS.from_documents(chunks, emb_model)


def create_llm(llm_path):
    return LlamaCpp(
        model_path=llm_path,
        n_gpu_layers=-1,
        n_ctx=3072,
        temperature=0.2,
        max_tokens=2000,
        top_p=0.9,
        verbose=False,
    )


def get_db(cached, emb_model_name, paper_url):
    cached_db = cached.get("db", None) if cached else None
    if (
        cached_db
        and cached_db.embeddings.model_name == emb_model_name
        and cached.get("paper_url", None) == paper_url
    ):
        return cached_db
    else:
        return create_db(paper_url, emb_model_name)


def get_llm(cached, llm_path):
    cached_llm = cached.get("llm", None) if cached else None
    if cached_llm and cached_llm.model_path == llm_path:
        return cached_llm
    else:
        return create_llm(llm_path)


def get_chain(
    paper_url: str, llm_path: str, emb_model_name: str, n_chunks: str, cached=None
):
    db = get_db(cached, emb_model_name, paper_url)
    retriever = db.as_retriever(search_kwargs={"k": n_chunks})
    prompt = PromptTemplate.from_template(
        "<s>[INST] {instructions} Hi [/INST] Hello! how can I help you</s>[INST]\n{chunks}\n\nUser question: {query} [/INST] ",
    )
    llm = get_llm(cached, llm_path)
    chain = (
        {
            "query": itemgetter("query"),
            "chunks": itemgetter("query") | retriever | format_chunks,
            "instructions": lambda _: system_prompt,
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, {"paper_url": paper_url, "db": db, "llm": llm}
