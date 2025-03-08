import os
import streamlit as st
from langchain_community.vectorstores import FAISS  
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory


## Building UI
st.title('Retrieval-Augmented Generation (RAG) with Chat History')
st.sidebar.title('Configuration')
st.sidebar.write('Last 5 conversation messages are used to build the query context')
session_id = st.sidebar.text_input(label='Change the session ID as required', value='Default')

## Configuration required for LLM, Embedding and text splitting
model = ChatOllama(model='llama3.1', temperature=0.5)
embedng_techn = OllamaEmbeddings(model='llama3.1')
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

## Declaring placeholder for chat history
if 'store' not in st.session_state:
    st.session_state.store = {}

## Uploading files for building context
uploaded_files = st.sidebar.file_uploader('Upload your document to build the context', type='pdf', accept_multiple_files=False)

## creating vector space and retriever for the uploaded file
if uploaded_files:

    if 'last_uploaded_files' not in st.session_state or st.session_state.last_uploaded_files != uploaded_files.name:
        st.session_state.last_uploaded_files = uploaded_files.name

        ## Deleting older vector space and retriever if a new file is uploaded
        if 'retriever' in st.session_state:
            ## Look for a way to reset user input element once a new file is uploaded
            # del st.session_state.user_input
            del st.session_state.splits
            del st.session_state.vector_store
            del st.session_state.retriever
 
        temp_pdf = f'./temp.pdf'
        with open(temp_pdf, 'wb') as file:
            file.write(uploaded_files.getvalue())

        pdf_loader = PyPDFLoader(file_path=temp_pdf)
        doc = pdf_loader.load()
        os.remove(temp_pdf)

        st.session_state.splits = text_splitter.split_documents(documents=doc)
        st.session_state.vector_store = FAISS.from_documents(documents=st.session_state.splits, embedding=embedng_techn)
        st.session_state.retriever = st.session_state.vector_store.as_retriever()


## Starting the conversation once the retriever is created
if 'retriever' in st.session_state:

    ## Prompt for creating a standalone user input if it refere to chat history context
    system_message = (
        "Given the chat history and current user question, which might refer to chat history context, "
        "formulate a standalone question which can be understood without the chat history."
        "Do not answer the question, just reformulate it if needed otherwise return it as it is."
    )

    chat_prompt = ChatPromptTemplate.from_messages(
        [
            ('system', system_message),
            MessagesPlaceholder(variable_name='chat_history'),
            ('human',"{input}")
        ]
    )

    history_aware_retriever = create_history_aware_retriever(model, st.session_state.retriever, chat_prompt)

    ## Prompt for adjusting system response from the LLM
    system_prompt = (
        "You are assistant for question-answering task. Use the below retrived context to answer the user question."
        "If you do not know the answer, just say that 'I don't know.'"
        "Keep the answer concise and to the point"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ('system', system_prompt),
            MessagesPlaceholder(variable_name='chat_history'),
            ('human', '{input}')
        ]
    )

    question_answer_chain = create_stuff_documents_chain(model, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    ## function for managing chat conversation history
    def get_session_history(session_id:str)->BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()

        chat_history = st.session_state.store[session_id]
        # print(chat_history)
        if len(chat_history.messages) > 10:
            chat_history.messages = chat_history.messages[-10:]

        return chat_history

    ## Runnable for chat continuation
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key='input',
        history_messages_key='chat_history',
        output_messages_key='answer'
    )

    ## accepts user input
    st.session_state.user_input = st.text_input(label='Ask your question regarding uploaded document.', key='User Input')
    if st.session_state.user_input:
        session_history = get_session_history(session_id)
        response = conversational_rag_chain.invoke(
            {
                'input':st.session_state.user_input
                },
            config={
                'configurable':{'session_id':session_id}
            }
        )

        st.write(response['answer'])

