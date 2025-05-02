import streamlit as st
import warnings
import json
import torch
import transformers
from transformers import AutoTokenizer, pipeline
from huggingface_hub import login
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document

warnings.filterwarnings('ignore')

st.title("Restaurant Q&A with LLM")

# Hugging Face Token Input
hf_token = st.text_input("Enter your Hugging Face Hub token:", type="password")
if hf_token:
    login(token=hf_token)

    # Load restaurant data
    try:
        with open("/content/Nugget-Assignment/restaurants_data.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    # Preprocess and convert to LangChain Document format
    documents = []
    for restaurant in data:
        content = "\n".join([
            f"Name: {restaurant.get('name')}",
            f"Opening Hours: {restaurant.get('opening_hours')}",
            f"Phone Number: {restaurant.get('phone_number')}",
            f"Price for Two: {restaurant.get('price_per_person')}",
            f"Address: {restaurant.get('address')}",
            f"More Info: {restaurant.get('url')}"
        ])
        documents.append(Document(page_content=content, metadata={"source": restaurant.get("name")}))

    # Model selection and loading
    model="meta-llama/Llama-2-7b-chat-hf"
    with st.spinner("Loading model and tokenizer..."):
        tokenizer = AutoTokenizer.from_pretrained(model)
        pipe = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            truncation=True,
            device_map="auto",
            max_length=512,
            max_new_tokens=100,
            do_sample=True,
            repetition_penalty=1.1,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id
        )

    llm = HuggingFacePipeline(pipeline=pipe, model_kwargs={'temperature': 0.2})

    # Embeddings and vector store
    with st.spinner("Creating embeddings and vector store..."):
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(documents, embedding_model)
        retriever = vectorstore.as_retriever()

    # Prompt template
    template = """
    Provide answer in bullet Points.
    and give answer in brief.
    Always end the answer with "Thanks for asking!".

    Context: {context}

    Question: {input}

    Response:
    """
    prompt = PromptTemplate(template=template, input_variables=['context', 'input'])

    # Create LLM Document Chain and Retrieval Chain
    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    retrieval_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=document_chain)

    # User input for question
    input_question = st.text_input("Ask a question about the restaurants:")
    if st.button("Get Answer"):
        if input_question.strip() == "":
            st.warning("Please enter a question.")
        else:
            with st.spinner("Generating answer..."):
                result = retrieval_chain.invoke({'input': input_question})
                answer = result['answer'].split('\n\n\n')[-1]
                st.markdown("### Answer")
                st.write(answer)
else:
    st.info("Please enter your Hugging Face Hub token to start.")
