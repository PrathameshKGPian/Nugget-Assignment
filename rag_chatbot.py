# Standard library imports
import warnings
import json

# Third-party imports
import torch
import transformers
from transformers import AutoTokenizer, pipeline
from huggingface_hub import login

# LangChain and related imports
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document
from kaggle_secrets import UserSecretsClient

# Suppress warnings
warnings.filterwarnings('ignore')

# Authenticate with Hugging Face Hub
user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("hf_token")
login(token=hf_token)

# Step 1: Load restaurant data from JSON
with open("restaurants_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Step 2: Preprocess and convert to LangChain Document format
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

# Load tokenizer and model pipeline
model = "meta-llama/Llama-2-7b-chat-hf"
# model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
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

# Wrap pipeline with LangChain's HuggingFacePipeline LLM
llm = HuggingFacePipeline(pipeline=pipeline, model_kwargs={'temperature': 0.2})

# Create embeddings and vector store
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents, embedding_model)
retriever = vectorstore.as_retriever()

# Define prompt template
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

# Inferencing
input_question = "Which restaurant has the lowest price for two?"
result = retrieval_chain.invoke({'input': input_question})

print(result['answer'].split('\n\n\n')[-1])
