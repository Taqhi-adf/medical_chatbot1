import streamlit as st
import os
#%%writefile app.py
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
# -------------------------------------------------
# API KEY
# -------------------------------------------------
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# -------------------------------------------------
# STREAMLIT PAGE
# -------------------------------------------------
st.set_page_config(
    page_title="Medical AI Chatbot",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Medical AI Chatbot")

st.warning(
    "This chatbot provides general medical information only. "
    "Please consult professional doctors for real medical advice."
)

# -------------------------------------------------
# LOAD PDFs
# -------------------------------------------------
documents = []

pdf_folder = "medical_pdfs"

for file in os.listdir(pdf_folder):

    if file.endswith(".pdf"):

        loader = PyPDFLoader(
            os.path.join(pdf_folder, file)
        )

        docs = loader.load()

        documents.extend(docs)

# -------------------------------------------------
# SPLIT DOCUMENTS
# -------------------------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

split_docs = splitter.split_documents(documents)

# -------------------------------------------------
# CREATE VECTOR STORE
# -------------------------------------------------
embeddings = OpenAIEmbeddings()

vectorstore = FAISS.from_documents(
    split_docs,
    embeddings
)

# -------------------------------------------------
# USER QUESTION
# -------------------------------------------------
user_question = st.text_input(
    "Ask Medical Question"
)

# -------------------------------------------------
# AI RESPONSE
# -------------------------------------------------
if user_question:

    with st.spinner("Analyzing medical information..."):

        # Retrieve relevant docs
        retrieved_docs = vectorstore.similarity_search(
            user_question,
            k=4
        )

        medical_context = "\n\n".join(
            [doc.page_content for doc in retrieved_docs]
        )

        # Prompt
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are an AI Medical Assistant.

Medical Information:
{context}

Patient Question:
{question}

Tasks:
1. Explain the condition simply
2. Mention possible causes
3. Suggest specialist doctor
4. Recommend basic medical tests/checkups
5. Mention emergency warning signs
6. Give safe health guidance

Important:
- Do not prescribe medicines
- Do not give dangerous advice
- Always recommend consulting real doctors
"""
        )

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0
        )

        final_prompt = prompt.format(
            context=medical_context,
            question=user_question
        )

        response = llm.invoke(final_prompt)

        st.subheader("🤖 AI Medical Guidance")

        st.write(response.content)

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown("---")

st.caption(
    "Medical AI Chatbot | Streamlit + LangChain + FAISS + OpenAI"
)