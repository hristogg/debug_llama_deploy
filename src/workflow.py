from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from pathlib import Path
from llama_index.embeddings.vertex import VertexTextEmbedding
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.storage.chat_store.redis import RedisChatStore
from llama_index.core.memory import ChatMemoryBuffer
from google.auth import default
from llama_index.core.postprocessor import LLMRerank
import google.auth
import vertexai
from llama_index.core import Settings
from llama_index.llms.vertex import Vertex
import asyncio
from llama_index.core.schema import NodeWithScore
from typing import List
from dotenv import load_dotenv
load_dotenv()
from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
import json

PROJECT_ID= 'yettel-bg-ds'
REGION = 'europe-west3'
credentials, project_id = google.auth.default()
REDIS_URL = ""
TTL = 100

from phoenix.otel import register
tracer_provider = register(
  project_name="testdebug", # Default is 'default'
  endpoint="https://app.phoenix.arize.com/v1/traces",
)
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from openinference.instrumentation.llama_index import get_current_span


from opentelemetry import trace
LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)


vertexai.init(project=PROJECT_ID, location=REGION)
f1 = Path(__file__).with_name('paulgraham1')
f2 = Path(__file__).with_name('paulgraham2')
print(f1)



embed_model = VertexTextEmbedding(model_name="text-multilingual-embedding-002",
            project=PROJECT_ID,
            location=REGION,
            #REMOVE
            credentials=credentials,
        )
llm = Vertex(
    model="gemini-1.5-pro-002",
    temperature=0,
    context_window=12288,
    max_tokens=3000,
    #context_window=10000,
    #REMOVE
    #credentials=credentials,
)
Settings._embed_model = embed_model
Settings.llm = llm

rerank_llm = Vertex(
    model="gemini-1.5-flash-002",
    temperature=0,
    context_window=4000,
    max_tokens=3000,
    #context_window=10000,
    #REMOVE
    #credentials=credentials,
)

ranker = LLMRerank(choice_batch_size=10, top_n=10, llm=rerank_llm)



class DocQuery(Event):
  #Result of Start Event - condensed question
  condensed_question: str

class OutputDoc(Event):
    data: str

class AnswerEvent(Event):
   new_nodes: List[NodeWithScore]

class RerankEvent(Event):
    new_nodes: List[NodeWithScore]


class RAGWorkflow(Workflow):

    document1 = SimpleDirectoryReader(f1).load_data()
    document2 = SimpleDirectoryReader(f2).load_data()
    
    index = VectorStoreIndex.from_documents(document1)
    retriever = index.as_retriever(verbose=True)

    index2 = VectorStoreIndex.from_documents(document2)
    retriever2 = index2.as_retriever(verbose=True)


    def init_history(self, user: str) -> "ChatMemoryBuffer":
      chat_store = RedisChatStore(redis_url=REDIS_URL, ttl=TTL)
      chat_memory = ChatMemoryBuffer.from_defaults(
                    token_limit=3000,
                    chat_store=chat_store,
                    chat_store_key=user,
                    )
      return chat_memory

    @step
    async def question_condense(self, ctx: Context, ev: StartEvent) ->  DocQuery:
        
        
        
        query = ev.get('query')
        user = ev.get('user')

        chat_memory = self.init_history(user)
        query_engine = CondensePlusContextChatEngine.from_defaults(
                    retriever=self.retriever,
                    memory=chat_memory,  
                    skip_condense=True,
                    verbose=True,
                    )
        chat_history = query_engine._memory.get(input=query)
        print(f'Chat history is {chat_history}')
        
        await ctx.set("query", query)
        await ctx.set('user', user)

        #no work to do just a mockup
        condensed_question = query
        return DocQuery(condensed_question=condensed_question)
    
    @step
    async def retriev(self, ctx: Context, ev: DocQuery) -> RerankEvent:
        condensed_query = ev.condensed_question
        print(f'Condensed question from {condensed_query} and type is {type(condensed_query)}')
        doc_task = asyncio.create_task(self.retriever.aretrieve(condensed_query))
        qna_task = asyncio.create_task(self.retriever2.aretrieve(condensed_query))
        doc_output, qna_output = await asyncio.gather(doc_task, qna_task)
        all_nodes = qna_output + doc_output 
        #json_data = json.dumps([node_with_score.__dict__ for node_with_score in all_nodes], default=lambda o: o.__dict__, ensure_ascii=False)
        return RerankEvent(new_nodes=all_nodes)
    
    @step 
    async def rerank(self, ctx: Context, ev: RerankEvent) -> AnswerEvent:
        condensed_question = await ctx.get('query')
        nodes_to_rerank = ev.new_nodes
        new_nodes = ranker.postprocess_nodes(
                nodes_to_rerank, query_str=condensed_question
            )
        return(AnswerEvent(new_nodes=new_nodes))

    
    @step
    async def answer_ev(self, ctx: Context, ev:  AnswerEvent ) -> StopEvent:
        query = await ctx.get('query')
        user = await ctx.get('user')
        chat_memory = self.init_history(user)
        crnt_span = get_current_span()
        span_id = crnt_span.get_span_context().span_id.to_bytes(8,'big').hex()
        print(f'Span id of answer evnet is {span_id}')
        nodes = ev.new_nodes
        query_engine = CondensePlusContextChatEngine.from_defaults(
                    retriever=self.retriever,
                    memory=chat_memory,              
                    skip_condense=True,
                    )
        response = query_engine.chat(message=query)
        return StopEvent(result=response)
        #deserialized_nodes_after_rrnk = [json_to_node_with_score(node_json) for node_json in json.loads(nodes)]


testworkflow = RAGWorkflow(verbose=True,timeout=60)

async def main():
        ans = await testworkflow.run(query='How many books did Paul Graham Publish', user='test')
        print(ans)

if __name__ == "__main__":
    asyncio.run(main())
