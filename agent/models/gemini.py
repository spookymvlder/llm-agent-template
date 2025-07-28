import os
from llama_index.llms.google_genai import GoogleGenAI
import google.generativeai as genai

# Configure Gemini API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


#Choose a version, such as gemini-2.5-flash
def loadGemin(version):
    return GoogleGenAI(model=version, api_key=os.getenv("GOOGLE_API_KEY"))