import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import ollama


st.set_page_config(page_title="Cookbook Router", layout="wide")
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
LLM_MODEL_NAME = "llama3"

Settings.embed_model = HuggingFaceEmbedding(
    model_name=EMBED_MODEL_NAME,
    device="cuda"
)

Settings.llm = Ollama(model=LLM_MODEL_NAME, request_timeout=60.0)

@st.cache_resource(show_spinner=False)
def load_data():
        italian_docs = SimpleDirectoryReader(input_files=["italian_cooking.txt"]).load_data()
        american_docs = SimpleDirectoryReader(input_files=["american_cooking.txt"]).load_data()
        ancient_docs = SimpleDirectoryReader(input_files=["ancient_english.txt"]).load_data()
        chinese_docs = SimpleDirectoryReader(input_files=["chinese_cooking.txt"]).load_data()
        vegetarian_docs = SimpleDirectoryReader(input_files=["vegetarian_cooking.txt"]).load_data()

        italian_index = VectorStoreIndex.from_documents(italian_docs)
        american_index = VectorStoreIndex.from_documents(american_docs)
        ancient_index = VectorStoreIndex.from_documents(ancient_docs)
        chinese_index = VectorStoreIndex.from_documents(chinese_docs)
        vegetarian_index = VectorStoreIndex.from_documents(vegetarian_docs)

        return {
            "Italian Cuisine": italian_index.as_query_engine(),
            "American (19th Century)": american_index.as_query_engine(),
            "Medieval English": ancient_index.as_query_engine(),
            "Chinese Cuisine": chinese_index.as_query_engine(),
            "Vegetarian Recipes": vegetarian_index.as_query_engine()
        }


engines = load_data()

query_engine_tools = [
    QueryEngineTool(
        query_engine=engines["Italian Cuisine"],
        metadata=ToolMetadata(
            name="italian_chef",
            description="Useful for finding Italian recipes, pasta dishes, and Mediterranean ingredients."
        )
    ),
    QueryEngineTool(
        query_engine=engines["American (19th Century)"],
        metadata=ToolMetadata(
            name="whitehouse_chef",
            description="Useful for formal American/French dinners, carving meats, and 19th-century presidential banquet styles."
        )
    ),
    QueryEngineTool(
        query_engine=engines["Medieval English"],
        metadata=ToolMetadata(
            name="medieval_cook",
            description="Useful for ancient English recipes (1390 AD), medieval ingredients, and historical cooking methods."
        )
    ),
    QueryEngineTool(
        query_engine=engines["Chinese Cuisine"],
        metadata=ToolMetadata(
            name="chinese_cook",
            description="Useful for Chinese recipes, stir-fry techniques, and traditional Asian ingredients."
        )
    ),
    QueryEngineTool(
        query_engine=engines["Vegetarian Recipes"],
        metadata=ToolMetadata(
            name="vegetarian_chef",
            description="Useful for vegetarian recipes, plant-based ingredients, and meatless cooking techniques."
        )
    ),
]

router = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=query_engine_tools,
    verbose=True
)

with st.sidebar:
    st.title("Control Panel")
    
    if st.button("Clear Chat History", type="primary"):
        st.session_state.messages = []
    
    st.divider()

    st.subheader("Loaded Data Sources")
    for name in engines.keys():
        st.markdown(f"✅ **{name}**")
    
    st.divider()

    st.subheader("System Stats")
    st.caption("Embedding Model")
    st.info(f"{EMBED_MODEL_NAME}")

    st.caption("LLM Model (Ollama)")
    try:
        model_info = ollama.show(LLM_MODEL_NAME)
        details = model_info.get('details', {})
        
        c1, c2 = st.columns(2)
        c1.metric("Size", details.get('parameter_size', 'Unknown'))
        c2.metric("Quant", details.get('quantization_level', 'Unknown'))
            
    except Exception as e:
        st.error("Ollama not connected")
        st.caption(f"Error: {e}")

st.title("Cookbook Assistant")
st.markdown("Ask a question, and I will answer it based on the appropriate cookbook!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How do I make a medieval stew?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consulting the library..."):
            try:
                response = router.query(prompt)
                
                st.markdown(response.response)
                
                source_tool = response.metadata.get("selector_result")
                if source_tool:
                    st.caption(f"*Sourced from: {source_tool}*")
                    
                st.session_state.messages.append({"role": "assistant", "content": response.response})
                
            except Exception as e:
                if "Failed to select query engine" in str(e):
                    error_msg = "I can only answer based on the available data sources. Please ask another question."
                    st.markdown(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                else:
                    st.error(f"Error generating response: {str(e)}")