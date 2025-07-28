import os
from llama_index.llms.google_genai import GoogleGenAI
import google.generativeai as genai

# Configure Gemini API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


#Choose a version, such as gemini-2.5-flash
def loadGemini(version):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")
    
    return GoogleGenAI(model=version, api_key=api_key)