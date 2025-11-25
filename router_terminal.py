import sys
import torch
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


device_type = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device_type}")

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    device=device_type
)

Settings.llm = Ollama(model="llama3", request_timeout=60.0)

def load_and_index():
    def load_book(filename, name):
        try:
            docs = SimpleDirectoryReader(input_files=[filename]).load_data()
            index = VectorStoreIndex.from_documents(docs)
            return index.as_query_engine()
        except Exception as e:
            print(f"   Error loading {filename}: {e}")
            sys.exit(1)

    return {
        "Italian": load_book("italian_cooking.txt", "Italian Cuisine"),
        "American": load_book("american_cooking.txt", "American 19th Century"),
        "Medieval": load_book("ancient_english.txt", "Medieval English"),
        "Chinese":  load_book("chinese_cooking.txt", "Chinese Cuisine"),
        "Vegetarian": load_book("vegetarian_cooking.txt", "Vegetarian Recipes"),


    }

engines = load_and_index()

query_engine_tools = [
    QueryEngineTool(
        query_engine=engines["Italian"],
        metadata=ToolMetadata(
            name="italian_chef",
            description="Useful for finding Italian recipes, pasta dishes, and Mediterranean ingredients."
        )
    ),
    QueryEngineTool(
        query_engine=engines["American"],
        metadata=ToolMetadata(
            name="whitehouse_chef",
            description="Useful for formal American/French dinners, carving meats, and 19th-century presidential banquet styles."
        )
    ),
    QueryEngineTool(
        query_engine=engines["Medieval"],
        metadata=ToolMetadata(
            name="medieval_cook",
            description="Useful for ancient English recipes (1390 AD), medieval ingredients, and historical cooking methods."
        )
    ),
    QueryEngineTool(
        query_engine=engines["Chinese"],
        metadata=ToolMetadata(
            name="chinese_cook",
            description="Useful for Chinese recipes, stir-fry techniques, and traditional Asian ingredients."
        )
    ),
    QueryEngineTool(
        query_engine=engines["Vegetarian"],
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


def main():
    print(" Type 'exit' or 'quit' to stop.")

    while True:
        user_input = input("\nAsk a question: ")

        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        
        if not user_input.strip():
            continue

        print("\nConsulting the cookbooks...")

        try:
            response = router.query(user_input)
            
            print("-" * 50)
            print(f"ANSWER: {response.response}")
            
            source_tool = response.metadata.get("selector_result")
            if source_tool:
                print(f"\n[SOURCE]: {source_tool}")
            print("-" * 50)

        except Exception as e:
            if "Failed to select query engine" in str(e):
                print("ROUTER DECISION: I cannot answer that based on the available cookbooks.")
            else:
                print(f"SYSTEM ERROR: {e}")

if __name__ == "__main__":
    main()