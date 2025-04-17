import streamlit as st
from chatbot import qa_chain
from dotenv import load_dotenv

load_dotenv()

st.title("Chatbot INTO - Lista de Espera")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.chat_input("Digite sua pergunta:")

if user_input:
    result = qa_chain({"query": user_input})
    
    # Exibir resposta
    with st.chat_message("user"):
        st.write(user_input)
    
    with st.chat_message("assistant"):
        st.write(result["result"])
        
        # Exibir fontes
        with st.expander("🔍 Ver fontes"):
            for doc in result["source_documents"]:
                st.write(doc.page_content[:500] + "...")

    # Atualizar histórico
    st.session_state.history.append({
        "pergunta": user_input,
        "resposta": result["result"]
    })
