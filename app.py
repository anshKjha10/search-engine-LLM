import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
# DuckDuckGoSearchRun is used for web search queries

from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv
load_dotenv()

# arxiv and wiki tools
api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

search = DuckDuckGoSearchRun(name="Search")

st.title("Langchain - Chat with Search")
"""
In this example, We are using 'StreamlitCallBackHandler' to display the thoughts and the final answer in the Streamlit app.
"""

# sidebar for settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API key : ", type = "password")

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant", "content":"Hi, I'm a chatbot who can search the web for you."}
    ]
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])
    
if prompt:=st.chat_input(placeholder="What is Machine Learning?"):
    st.session_state.messages.append({"role":"user", "content":prompt})
    st.chat_message("user").write(prompt)
    
    llm = ChatGroq(groq_api_key = api_key, model_name = "llama3-8b-8192", streaming=True)
    tools = [search, arxiv, wiki]
    
    search_agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handling_parsing_error=True)
    
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role":"assistant", "content":response})
        st.write(response)