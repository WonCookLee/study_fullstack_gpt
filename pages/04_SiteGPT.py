from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import Html2TextTransformer
import streamlit as st
import sys
import asyncio


st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)
if 'win32' in sys.platform:
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    cmds = [['C:/Windows/system32/HOSTNAME.EXE']]
else:
    # Unix default event-loop policy & cmds
    cmds = [
        ['du', '-sh', '/Users/fredrik/Desktop'],
        ['du', '-sh', '/Users/fredrik'],
        ['du', '-sh', '/Users/fredrik/Pictures']
    ]

html2text_transformer = Html2TextTransformer()

st.markdown(
    """
    # SiteGPT
            
    Ask questions about the content of a website.
            
    Start by writing the URL of the website on the sidebar.
"""
)


with st.sidebar:
    url = st.text_input(
        "Write down a URL",
        placeholder="https://example.com",
    )


if url:
    loader = AsyncChromiumLoader([url])
    docs = loader.load()
    transformed = html2text_transformer.transform_documents(docs)
    st.write(docs)
