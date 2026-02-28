from langchain_ollama import ChatOllama
import re

def get_deepseek_llm():
    """Initializes DeepSeek-R1 via Ollama."""
    return ChatOllama(
        model="deepseek-r1:7b",
        temperature=0.2, # Keeps legal analysis factual
    )

def clean_reasoning(response_text):
    """
    DeepSeek-R1 outputs 'thinking' inside <think> tags.
    This removes them to show only the final translation to the user.
    """
    # Removes everything between <think> and </think>
    clean_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
    return clean_text.strip()