# Retrieval-Augmented Generation (RAG)-based ChatBot with chat history context

## Overview
This is a RAG-based Chatbot built using LangChain, Faiss, and Ollama(Llama 3.1 - 8B parameter model). It allows users to upload PDF documents and communicate with them. The Chatbot maintains the last five conversations to build the context of any previous question referred to.

## Features
* Flexibility in building any custom knowledge base
* The knowledge base build isn't stored anywhere.
* All the operations are performed locally and in system memory.
* Can change the LLM as and when required.
* Keep track of the last 5 user queries to enhance the response quality but is deleted once the session is finished.

## Working and User Interface
1. Once the application is launched, it asks the user to upload a PDF file to build the knowledge base for the Chat session.
![Screenshot_1](https://github.com/user-attachments/assets/b533f2ab-dc97-4b60-8c32-4b10dec2765d)

2. The user should upload the PDF file to build the context. I'm uploading the research paper 'Attention is All You Need'. Once the PDF file is uploaded, the app will fetch the data from the PDF, perform text embedding using the embedding technique specified, and store it in system memory using FAISS.
![Screenshot_2](https://github.com/user-attachments/assets/6df664ab-f5f6-45c2-992a-1b312fd8c89f)
   You can see the Running message in the upper right corner

3. Once the knowledge base or context for the chat is ready, you can see a message to start the conversation as below. The time taken depends on the uploaded PDF size and your system config.
![Screenshot_3](https://github.com/user-attachments/assets/d86c4691-efd0-4ae2-8662-c5cb810636fa)

4. Now the user can query the document. The user input is first encoded, its similarity with the context is retrieved and the result is generated.
![Screenshot_4](https://github.com/user-attachments/assets/b05cebed-4fa2-4827-9845-1ad7c7da619f)

5. You can ask your next query that you have.
![Screenshot_5](https://github.com/user-attachments/assets/1dd7b721-7c30-4390-ba53-e9ef2c7114d6)
![Screenshot_6](https://github.com/user-attachments/assets/a529a236-da1c-45f5-9f68-688f1b1c25a9)

7. You can also ask about the previous conversation that you had.
![Screenshot_7](https://github.com/user-attachments/assets/092b8390-7135-4753-9a7c-b8a818fdd7dd)


## Technology and Framework used
* Python
* LangChain
* Ollama
* Open Source LLM's

## Future improvements and updates
* Add support for tabular data and other file format.
* Add multiple data at once
* Support for changing the knowledge in the same session by uploading other files
* Select the LLM to be used for the conversation.
