import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_huggingface.llms import HuggingFacePipeline
from streamlit_chat import message



def get_pdf_text(pdf_docs):                             #loop through pdf, loop through each pages of pdf, concatenate text
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):                             #create chunck of text
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    #embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    llm = HuggingFacePipeline.from_model_id(
        model_id="google/flan-t5-base",  # You can change to flan-t5-large or other
        task="text2text-generation",
        pipeline_kwargs={"max_new_tokens": 512, "temperature": 0.3}
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")

    st.header("Chat with multiple PDFs :books:")
    #st.text_input("Ask a question about your docs:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here", accept_multiple_files=True)
        if st.button("Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF.")
            else:
                with st.spinner("Processing..."):
                    # get pdf text
                    raw_text = get_pdf_text(pdf_docs)
                    #st.write(raw_text)

                    # get the text chunks
                    text_chunks = get_text_chunks(raw_text)
                    #st.write(text_chunks)

                    # create vector store
                    vectorstore = get_vectorstore(text_chunks)

                    # create conversation chain
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.success("✅ Documents processed successfully! You can start chatting now.")




    # Chat input
    user_input = st.text_input("Ask a question about your documents:")
    if user_input and st.session_state.conversation:
        with st.spinner("Thinking..."):
            response = st.session_state.conversation.run(user_input)

        # Store chat history
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("ai", response))

    # Display chat messages
    for i, (role, msg) in enumerate(st.session_state.chat_history):
        if role == "user":
            message(msg, is_user=True, key=f"user_{i}")
        else:
            message(msg, is_user=False, key=f"ai_{i}")

if __name__ == '__main__':
    main()
