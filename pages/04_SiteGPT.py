from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import Html2TextTransformer
import streamlit as st
import sys
import asyncio
from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return (
        str(soup.get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
        .replace("CloseSearch Submit Blog", "")
    )


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


@st.cache_data(show_spinner="Loading website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    # loader = SitemapLoader(
    #     url,
    #     # filter_urls=[
    #     #     r"^(.*\/ko\/).*",
    #     # ],
    #     parsing_function=parse_page,
    # )
    # loader.requests_per_second = 2
    # docs = loader.load_and_split(text_splitter=splitter)
    # return docs
    loader = AsyncChromiumLoader([url])
    return loader.load()


with st.sidebar:
    url = st.text_input(
        "Write down a URL",
        placeholder="https://example.com",
        value="https://www.google.com/gmail/sitemap.xml"
    )


if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write down a Sitemap URL.")
    else:
        docs = load_website(url)
        st.write(docs)
