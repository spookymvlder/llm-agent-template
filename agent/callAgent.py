from setupAgent import agent, logger


#If tool is simply asking the agent a question
response = agent.chat("When doing something, what should be done?")
print(str(response))



# Example usage if the agent is instructed to perform a more complex task, like visit a URL
url_list = [
    #"[https://example.com/not_interested_event_1](https://example.com/not_interested_event_1)", # Replace with actual event URLs
    #"[https://example.com/not_interested_event_2](https://example.com/not_interested_event_2)",
]

def get_tags_for_urls(agent_instance, url_list, time_limit_days=None):
    all_tags = []
    for url in url_list:
        logger.info(f"Agent asking to analyze URL: {url} with time limit: {time_limit_days}")
        try:
            # Construct the chat prompt to clearly tell the agent what arguments to use for the tool
            if time_limit_days is not None:
                # The agent should see the arguments explicitly mentioned to map them correctly
                prompt_query = f"Using the 'media_analyzer' tool, extract themes from this content: {url}. Focus on the last {time_limit_days} days."
            else:
                prompt_query = f"Using the 'media_analyzer' tool, extract themes from this content: {url}."

            response = agent_instance.chat(prompt_query)
            tags_str = str(response).strip()
            if tags_str and tags_str != 'N/A - Could not extract themes.' and not tags_str.startswith("Error:"):
                tags = [tag.strip().lower() for tag in tags_str.split(',') if tag.strip()]
                all_tags.extend(tags)
                logger.info(f"Extracted tags for {url}: {tags}")
            else:
                logger.warning(f"No meaningful tags extracted or error for {url}: {tags_str}")
        except Exception as e:
            logger.error(f"Failed to get tags for {url} using agent: {e}")
    return list(set(all_tags))

interest_tags = get_tags_for_urls(agent, url_list, time_limit_days=90)
print(f"\n--- Theme Tags from Media/Public Content (Last 90 Days) ---")
print(interest_tags)