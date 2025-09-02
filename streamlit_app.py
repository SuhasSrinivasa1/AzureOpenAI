import streamlit as st
import os
import requests
import traceback

# Proxy Settings
os.environ["HTTP_PROXY"] = "http://127.0.0.1:3128"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:3128"

# Azure OpenAI Config
ENDPOINT = "https://ijj3kor-7111-resource.cognitiveservices.azure.com/"
API_VERSION = "2024-12-01-preview"
DEPLOYMENT = "gpt-4.1"
API_KEY = "4PV2I3zS721qEgLi3JQ5VA62D3pCa5LHumYf3gJMOjzOer8BkULYJQQJ99BHACHYHv6XJ3w3AAAAACOGEvmc"

# Build request URL
url = f"{ENDPOINT}openai/deployments/{DEPLOYMENT}/chat/completions?api-version={API_VERSION}"
headers = {
    "Content-Type": "application/json",
    "api-key": API_KEY
}

def ask_azure_openai(history):
    """Send chat history to Azure OpenAI and return the assistant's reply"""
    payload = {
        "messages": history,
          "max_tokens": 4096,
        "temperature": 0.7
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}: {response.text}")
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        traceback.print_exc()
        raise

def main():
    st.title("Azure OpenAI Chat")
    # Initialize chat history in session state
    if "history" not in st.session_state:
        st.session_state.history = []
    # Display previous messages
    for msg in st.session_state.history:
        st.chat_message(msg["role"]).write(msg["content"])
    # Input box
    prompt = st.chat_input("You: ")
    if prompt:
        # Append user message to history and display it
        st.session_state.history.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        # Get assistant reply
        with st.spinner("Thinking..."):
            try:
                reply = ask_azure_openai(st.session_state.history)
                st.session_state.history.append({"role": "assistant", "content": reply})
                st.chat_message("assistant").write(reply)
            except Exception as e:
                st.error(str(e))

if __name__ == "__main__":
    main()
