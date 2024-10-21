from dotenv import load_dotenv
load_dotenv()  # loading all the env variables
import streamlit as st
import os
import time

# faiss db
from langchain.vectorstores import FAISS

# langchain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader

# langchian with groq
from langchain_groq import ChatGroq

# langchain with gemini
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# set Groq API key
groq_api_key = os.getenv("GROQ_API_KEY")

# set Gemini API key
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# heading
st.header("Q&A ChatBot")
st.subheader("Gemma 7b model using groq API")

# load llm
llm = ChatGroq(groq_api_key=groq_api_key, model="gemma-7b-it")

# default prompt
prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}
"""
)

# vector embedding function
def vector_embedding():
    if "vectors" not in st.session_state:
        # set embedding
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
        # Data Ingestion
        st.session_state.loader = PyPDFDirectoryLoader("./us_census")
        # doc load
        st.session_state.docs = st.session_state.loader.load()
        # set chunks
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        # splitting
        st.session_state.final_docs = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])
        # vector OpenAI embeddings
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_docs, st.session_state.embeddings)

input_text = st.text_input("Enter your question here...")

# button to start vector embedding
if st.button("Start Documents Embedding"):
    vector_embedding()
    st.write("Vector DB is ready!")

if input_text:
    # doc chain
    doc_chain = create_stuff_documents_chain(llm, prompt)
    # vector retriever
    retriever = st.session_state.vectors.as_retriever()
    # retriever chain
    retriever_chain = create_retrieval_chain(retriever, doc_chain)
    # get process time
    start = time.process_time()
    # response
    response = retriever_chain.invoke({'input':input_text})
    # print response time
    st.write("Response time :", time.process_time() - start)
    st.write(response['answer'])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("-"*10)