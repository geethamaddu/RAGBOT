import os

import streamlit as st

from rag_utility import process_document_ingestion,answer_question

working_dir=os.getcwd()

st.set_page_config(
    page_icon="📑",
    page_title="RAGBOT",
    layout="centered"
    
)

st.title("RAG-BOT🤖")
st.subheader("llama model document RAG")
uploaded_files=st.file_uploader("Upload a PDF file",type=["pdf"],accept_multiple_files=True)

if uploaded_files:
    files_names=[]
    for uploaded_file in uploaded_files:
        save_path=os.path.join(working_dir,uploaded_file.name)
        with open(save_path,"wb") as f:
            f.write(uploaded_file.getbuffer())
        files_names.append(uploaded_file.name)

    process_document=process_document_ingestion(uploaded_file.name)
    st.info("Document Processes Successfully")
    user_question=st.text_area("Ask your question about the document")
    if st.button("Answer"):
            answer,sources=answer_question(user_question)
            st.markdown("llama response")
            st.markdown(answer)
            st.markdown("sources")
            st.markdown(sources)
