import gradio as gr
from rag_backend import load_and_index_documents, generate_rag_response

retriever = None

def chatbot(query, history):
    global retriever

    if retriever is None:
        retriever = load_and_index_documents()

    response = generate_rag_response(query, retriever)
    history.append((query, response))

    return "", history


with gr.Blocks() as demo:
    gr.Markdown("## 💬 RAG Chatbot (NEC + Wattmonk)")

    chatbot_ui = gr.Chatbot()
    msg = gr.Textbox(placeholder="Ask a question...")
    
    msg.submit(chatbot, [msg, chatbot_ui], [msg, chatbot_ui])

demo.launch()