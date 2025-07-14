import json
from flask import Flask, request, jsonify
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document

# 🔌 Load your stored DB
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory="db", embedding_function=embedding)
retriever = vectordb.as_retriever(search_type="similarity", k=1)

# ⚙️ Create Flask app
app = Flask(__name__)

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get('question')

    if not question:
        return jsonify({"error": "Missing 'question' in request"}), 400

    results = retriever.get_relevant_documents(question)
    if results:
        return jsonify({"answer": results[0].page_content})
    else:
        return jsonify({"answer": "Sorry, I couldn't find an answer."})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
