from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.llms import Together  # ✅ Changed from OpenAI to Together
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *  # where system_prompt is defined
import os

app = Flask(__name__)
load_dotenv()

# ✅ Load API keys
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
TOGETHER_API_KEY = os.environ.get('TOGETHER_API_KEY')  # ✅ New

# ✅ Set them for the environment
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY  # ✅ New

# ✅ Load HF embeddings
embeddings = download_hugging_face_embeddings()

# ✅ Setup Pinecone retriever
index_name = "ashaai"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# ✅ Initialize Together LLM using Mistral
llm = Together(
    model="mistralai/Mistral-7B-Instruct-v0.1",
    temperature=0.4,
    max_tokens=500
)

# ✅ Define ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# ✅ Set up chains
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# ✅ Flask endpoints
@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print("User: ", input)
    response = rag_chain.invoke({"input": msg})
    print("AshaAI: ", response["answer"])
    return str(response["answer"])

# ✅ Run the app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
