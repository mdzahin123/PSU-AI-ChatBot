from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime


def save_to_txt(data: str, filename: str = "research_output.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)
    
    return f"Data successfully saved to {filename}"

def penn_state_search(query: str):
    query_with_site = f"{query} site:psu.edu"
    return search.run(query_with_site)

def get_tools_for_query(query: str): #filter out for ai to use only txt or others
    """
    Returns a list of tools based on the type of query.
    Only enable `save_tool` for .txt files or saving requests.
    """
    query_lower = query.lower()
    
    # If query mentions saving to txt
    if  "save to file" in query_lower:
        return [save_tool]  # only allow to save data
    elif "only pennstate links" in query_lower:
        return [penn_state_tool]  # only allow to use data from penn state websites
    else:
        # For normal research/search queries
        return [search_tool, wiki_tool]

penn_state_tool = Tool(
    name="penn_state_search",
    func=penn_state_search,
    description="Search the web only on Penn State websites (psu.edu). Use this tool when the user wants information exclusively from Penn State."
)
save_tool = Tool(
    name="save_text_to_file",
    func=save_to_txt,
    description="Saves structured research data to a text file.",
)

search = DuckDuckGoSearchRun()
search_tool = Tool( #general search tool
    name="search",
    func=search.run,
    description="Search the web for relevant information to answer the user's query. Use this tool when you need to find current information or specific details about a topic."
)
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)#limit the content size
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper) # wikipedia tool

