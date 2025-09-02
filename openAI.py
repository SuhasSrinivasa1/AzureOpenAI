# -*- coding: utf-8 -*-
"""
Interactive Azure OpenAI Chat via Bosch CNTLM proxy
Author: IJJ3KOR
Date: 2025-07-28

This script provides a simple command-line chat loop for interacting
with an Azure OpenAI deployment while respecting the Bosch corporate
proxy configuration. The environment variables HTTP_PROXY and
HTTPS_PROXY are set to point at a locally running CNTLM forward proxy
so that outbound HTTPS requests to Azure's API can authenticate via
Bosch's NTLM proxy stack. You can type messages at the ``You:`` prompt
and the assistant's responses will be streamed back to the console.

Usage:
    python openAI.py

Press ``Ctrl+C`` or type ``exit``/``quit`` to terminate.
"""

import os
import requests
import traceback

# ===== Proxy Settings =====
# CNTLM or local forward proxy on port 3128
os.environ["HTTP_PROXY"] = "http://127.0.0.1:3128"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:3128"

# ===== Azure OpenAI Config =====
ENDPOINT: str = "https://ijj3kor-7111-resource.cognitiveservices.azure.com/"
API_VERSION: str = "2024-12-01-preview"
DEPLOYMENT: str = "gpt-4.1"
API_KEY: str = (
    "4PV2I3zS721qEgLi3JQ5VA62D3pCa5LHumYf3gJMOjzOer8BkULYJQQJ99BHACHYHv6XJ3w3AAAAACOGEvmc"
)

# ===== Build Request URL =====
url: str = f"{ENDPOINT}openai/deployments/{DEPLOYMENT}/chat/completions?api-version={API_VERSION}"

headers = {
    "Content-Type": "application/json",
    "api-key": API_KEY,
}

def ask_azure_openai(prompt: str, history: list | None = None) -> list:
    """Send a prompt to Azure OpenAI and return the updated history.

    Parameters
    ----------
    prompt: str
        The user's message to send.
    history: list | None
        List of prior conversation messages. Each element is a dict with
        ``role`` and ``content`` keys. The conversation is maintained
        across calls so that the model has context.

    Returns
    -------
    list
        The updated history including the new user and assistant messages.
    """
    if history is None:
        history = []

    # Add the new user message to history
    history.append({"role": "user", "content": prompt})

    payload = {
        "messages": history,
        "max_tokens": 200,
        "temperature": 0.7,
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code != 200:
            print(f"[ERROR] HTTP {response.status_code}")
            print(response.text)
            return history

        data = response.json()
        reply = data["choices"][0]["message"]["content"]
        print(f"Assistant: {reply}")

        # Add assistant reply to history
        history.append({"role": "assistant", "content": reply})
    except Exception:
        print("[EXCEPTION] An error occurred:")
        traceback.print_exc()

    return history


def main() -> None:
    """Main interactive loop for chatting with the assistant."""
    print("Azure OpenAI Chat — type 'exit' to quit")
    chat_history: list = []
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if not user_input:
            continue
        chat_history = ask_azure_openai(user_input, chat_history)


if __name__ == "__main__":
    main()
