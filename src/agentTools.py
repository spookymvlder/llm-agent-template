from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool
from setupAgent import q_engine, logger
from datetime import datetime, timedelta

#importing here for example function
import google.generativeai as genai

# --- Custom LlamaIndex Tool for  URL Analysis ---
# Make sure the function signature matches the expected arguments for the tool
def analyze_media_url(url: str, time_limit_days: int = None) -> str:
    """
    Analyzes a given  media URL to extract relevant tags.
    This tool uses the Gemini API's URL context feature to read the webpage.
    It works best for publicly accessible sites.
    """
    try:
        # The structure for URL context in the *new* SDK's generate_content call
        # looks like this (you embed the URL directly as a Part):

        prompt_text = "Analyze the content of the following webpage and extract "
        if time_limit_days is not None:
            current_date = datetime.now()
            start_date_limit = current_date - timedelta(days=time_limit_days)
            prompt_text += (f"**Only consider updates made within the last {time_limit_days} days "
                            f"(from approximately {start_date_limit.strftime('%Y-%m-%d')} to {current_date.strftime('%Y-%m-%d')}).** ")

        prompt_text += ("a comprehensive list of tags or keywords that represent discussed topics. Focus on recurring "
                        "themes rather than one-off mentions. "
                        "Provide the output as a comma-separated string of tags/keywords. "
                        "If no clear themes can be identified, return 'N/A'.")

        

        model = genai.GenerativeModel("gemini-2.5-flash") # Or "gemini-1.5-pro"

        logger.info(f"Sending prompt to Gemini for URL: {url} with time limit: {time_limit_days} days")

        # Method 1: Try with URL as text (simpler approach)
        try:
            full_prompt = f"{prompt_text}\n\nURL to analyze: {url}"
            response = model.generate_content(
                full_prompt,
                generation_config=genai.GenerationConfig(temperature=0.2)
            )
        except Exception as url_error:
            logger.warning(f"Direct URL method failed: {url_error}")
            # Method 2: If URL processing fails, try with Part (if supported)
            try:
                contents = [
                    prompt_text,
                    {"mime_type": "text/plain", "data": f"Please analyze this URL: {url}"}
                ]
                response = model.generate_content(
                    contents,
                    generation_config=genai.GenerationConfig(temperature=0.2)
                )
            except Exception as part_error:
                logger.error(f"Part method also failed: {part_error}")
                return f"Error: Unable to process URL {url}. Both methods failed."

        if response and response.text:
            logger.info(f"Successfully analyzed {url}. Raw response: {response.text[:200]}...")
            cleaned_response = response.text.strip().replace("```", "").replace("json", "").replace("JSON", "").replace("text", "").strip()
            return cleaned_response
        else:
            logger.warning(f"No text response from Gemini for URL: {url}")
            return "N/A - Could not extract themes."

    except Exception as e:
        logger.error(f"Error analyzing URL {url}: {e}")
        return f"Error: {e}"




example_function_tool = FunctionTool(
    fn=analyze_media_url,
    metadata=ToolMetadata(
        name="sample function tool",
        description=(
            "Performs analysis of X to extract Y in a specified format."
            "Input should include A, B, and C, with the specified types. "
            "Specify return in event of failure and provide example function call including any keywords."
            "Example usage: analyze_media_url(url='https://example.com/profile', time_limit_days=30)"
        )
    )
)

example_query_tool = QueryEngineTool(
        query_engine=q_engine,
        metadata=ToolMetadata(
            name="sample tool 1",
            description=(
                "Use a detailed plain text question to act as input for the tool."),
        ),
    )

query_engine_tools = [
    example_query_tool,
    example_function_tool
]

def getQueryTools():
    return query_engine_tools