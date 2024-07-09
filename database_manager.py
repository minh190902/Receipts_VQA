import weaviate
from weaviate.embedded import EmbeddedOptions
import torch
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import BaseNode, Document
from llama_index.core.postprocessor import SentenceTransformerRerank

from typing import List, Dict, Sequence

class DatabaseManager:
    def __init__(self, index_name: str):
        self.client = weaviate.Client(embedded_options=EmbeddedOptions())
        self.index_name = index_name
        device_type = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", device=device_type)

        self.vector_store = WeaviateVectorStore(
            weaviate_client=self.client,
            index_name=self.index_name,
        )

        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        self.create_index_schema()

        self.index = VectorStoreIndex(
            nodes=[],
            storage_context=self.storage_context,
            embed_model=self.embed_model
        )

        Settings.llm = None

    def create_index_schema(self):
        class_obj = {
            # Class definition
            "class": self.index_name,
            # Property definitions
            "properties": {
                "prompt": "string",
                "ocr_text": "string",
             },

            # Specify a vectorizer
            "vector_index_config": {
                "vectorizer": "text2vec-contextionary",
                "index": "hnsw",
                "cleanup_interval_seconds": 60
            }
        }
        if not self.client.schema.exists(self.index_name):
            self.client.schema.create(class_obj)

    def add_nodes(self, nodes: Sequence[BaseNode]):
        self.index.insert_nodes(nodes)

    def delete_node(self, node_id: str):
        self.index.delete_nodes([node_id])

    def query(self, query_text: str):
        rerank_postprocessor = SentenceTransformerRerank(
            model='mixedbread-ai/mxbai-rerank-xsmall-v1',
            top_n=2, # number of nodes after re-ranking,
            keep_retrieval_score=True
        )
        query_engine = self.index.as_query_engine(
            similarity_top_k=2,
            node_postprocessors=[rerank_postprocessor],
        )
        return query_engine.query(query_text)

    def ingest_data(self, data: List[Dict[str, str]]):
        documents = []
        for item in data:
            prompt = item.get('prompt', '')
            ocr_text = item.get('ocr_text', '')
            combined_text = f"Prompt: {prompt}\nOCR Text: {ocr_text}\n"
            documents.append(Document(text=combined_text))

        splitter = SemanticSplitterNodeParser(
            buffer_size=1, breakpoint_percentile_threshold=95, embed_model=self.embed_model
        )
        nodes = splitter.get_nodes_from_documents(documents)

        for node in nodes:
            print('-' * 100)
            print(node.get_content())

        self.add_nodes(nodes)
    
    def reset_database(self):
        if self.client.schema.exists(self.index_name):
            self.client.schema.delete_class(self.index_name)

    def get_monthly_spending(self, month: str, year: str) -> float:
        query_text = f"Total spending for {month}/{year}"
        results = self.query(query_text)
        if results:
            return float(results[0]["response"].split("$")[1].strip())
        return 0.0
