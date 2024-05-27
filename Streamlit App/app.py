# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import PyPDF2
import os
from htmlTemplate import css, bot_template, user_template

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import LlamaCpp
#from langchain.llms import LlamaCpp


from langchain.embeddings import HuggingFaceEmbeddings, GPT4AllEmbeddings # import hf embedding
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer, util
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

import streamlit as st





template = """[INST]
As an AI expert, based on the provided document,please provide accurate, important and relevant information. Your responses should follow the following guidelines:
- Answer the question based on the provided documents.
- Be direct, factual and precise while answering, limited to 50 words and 2-3 sentences. Begin your response without using introductory phrases like yes, no etc.
- Maintain an ethical, unbiased and neutral tone, avoiding harmful or offensive content.
- If the document does not contain relevant information, state "The document doesn't have any relevent information avilable."
- Do not include questions in your responses.
- Answer the questions directly. do not ask me questions
{question}
[/INST]
"""



def process_pdf(pdf_files):
    documents = []
    metadata = []
    content = []

    for i in pdf_files:

        pdf_read = PyPDF2.PdfReader(i)
        for ind, text in enumerate(pdf_read.pages):
            doc_page = {'title': i.name + " page " + str(ind + 1),
                        'content': pdf_read.pages[ind].extract_text()}
            documents.append(doc_page)
    for doc in documents:
        content.append(doc["content"])
        metadata.append({
            "title": doc["title"]
        })
    print("Content and metadata are extracted from the documents")
    return content, metadata


def split_content(content, metadata):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=512,chunk_overlap=256)
    smaller_docs = splitter.create_documents(content, metadatas=metadata)
    print(f"Docs are split into {len(smaller_docs)} passages")
    return smaller_docs

def ingest_into_vectordb(smaller_docs):
    emb = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
    #emb = GPT4AllEmbeddings(model_name='all-MiniLM-L6-v2.gguf2.f16.gguf', gpt4all_kwargs={'allow_download': 'True'})
    db = FAISS.from_documents(smaller_docs, emb)

    DB_FAISS_PATH = 'vectorstore/db_faiss'
    db.save_local(DB_FAISS_PATH)
    return db


def conversation_func(vector):
    llama_llm = LlamaCpp(
    model_path="C:/Users/Mrinal Kalita/langchain-notes/mistral-7b-openorca.gguf2.Q4_0.gguf",
    temperature=0.75,
    max_tokens=200,
    top_p=1,
    n_ctx=3000)

    retriever = vector.as_retriever()
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(template)

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True, output_key='answer')

    conversation_chat = (ConversationalRetrievalChain.from_llm
                          (llm=llama_llm,
                           retriever=retriever,
                           #condense_question_prompt=CONDENSE_QUESTION_PROMPT,
                           memory=memory,
                           return_source_documents=True))
    print("Conversation function created for the LLM using the vector store")
    return conversation_chat


def validate_answer_against_sources(response_answer, source_documents):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    similarity_threshold = 0.5  
    source_texts = [doc.page_content for doc in source_documents]

    answer_embedding = model.encode(response_answer, convert_to_tensor=True)
    source_embeddings = model.encode(source_texts, convert_to_tensor=True)

    cosine_scores = util.pytorch_cos_sim(answer_embedding, source_embeddings)


    if any(score.item() > similarity_threshold for score in cosine_scores[0]):
        return True  

    return False


           
            
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    
    for i, message in enumerate(st.session_state.chat_history):
        print(i)
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            print(message.content)
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)



def main():

    st.set_page_config(page_title="Chat with your PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")

    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)

        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                content, metadata = process_pdf(pdf_docs)

                # get the text chunks
                split_docs = split_content(content, metadata)

                # create vector store
                vectorstore = ingest_into_vectordb(split_docs)

                # create conversation chain
                st.session_state.conversation = conversation_func(vectorstore)


if __name__ == '__main__':
    main()