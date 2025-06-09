import streamlit as st
import requests

st.title("🦙 TinyLlama Chatbot")
st.write("Hello! I'm your chatbot.")

prompt = st.text_input("💬 Enter your prompt:")

if st.button("🚀 Generate Response"):
    with st.spinner("Generating response..."):
        response = requests.post(
            "http://localhost:8000/infer",
            json={"prompt": prompt, "max_new_tokens": 100, "temperature": 0.7}
        )
        if response.status_code == 200:
            result = response.json()
            st.success("✅ Response received!")
            st.write(result["response"])
        else:
            st.error(f"❌ Failed to get response from API: {response.text}")
