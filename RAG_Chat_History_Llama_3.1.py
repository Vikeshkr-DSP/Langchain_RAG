import streamlit as st
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory

## Setting-up Streamlit
st.title('RAG with PDF with message history')
st.write('Upload PDF')

## Configuration required for LLM with chat histiry
model = ChatOllama(model='llama3.1', temperature=0.5)
embedng_techn = OllamaEmbeddings(model='llama3.1')
session_id = st.text_input(label='Session ID', value='Default')
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

## Managing the concersation
if 'store' not in st.session_state:
    st.session_state.store = {}

uploaded_files = st.file_uploader('Upload a PDF file', type='pdf', accept_multiple_files=True)
print(uploaded_files)

if uploaded_files:

    document = []
    for files in uploaded_files:

        temp_pdf = f'./temp.pdf'
        with open(temp_pdf, 'wb') as file:
            file.write(files.getvalue())

        pdf_loader = PyPDFLoader(file_path=temp_pdf)
        doc = pdf_loader.load()
        document.extend(doc)

    splits = text_splitter.split_documents(documents=document)
    vector_store = Chroma.from_documents(documents=splits, embedding=embedng_techn)
    retriever = vector_store.as_retriever()

    ## Prompts required for the model
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

    history_aware_retriever = create_history_aware_retriever(model, retriever, chat_prompt)

    system_prompt = (
        "You are assistant for question-answering task. Use the below retrived context to answer the user question."
        "If you do not know the answer, just say that 'I son't know.'"
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

    def get_session_history(session_id:str)->BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key='input',
        history_messages_key='chat_history',
        output_messages_key='answer'
    )

    user_input = st.text_input('Ask your question regarding uploaded document.')
    if user_input:
        session_history = get_session_history(session_id)
        response = conversational_rag_chain.invoke(
            {'input':user_input},
            config={
                'configurable':{'session_id':session_id}
            }
        )

        st.write(st.session_state.store)
        st.write('Assistant', response['answer'])
        st.write('Chat history', session_history.messages)

