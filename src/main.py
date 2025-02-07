from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import logging

from langchain_core.messages import ToolMessage
from contextlib import asynccontextmanager
from src.utils.helpers import load_config
from src.rag_pipeline import RAGCook

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):

    config = load_config()

    app.state.rag_assistant = RAGCook(
        persistent_directory_path=config["persistent_dir_path"],
        collection_name=config["db_collection_name"],
        retriever_name=config["retriever_tool_name"],
        retriever_description=config["retriever_tool_description"],
        llm_config=config,
    )
    yield
    logger.info("Clearing RAG pipeline...")
    del app.state.rag_assistant


app = FastAPI(title="My FastAPI Project", lifespan=lifespan)


class BaseResponse(BaseModel):
    response: str
    retriever_used: bool


class BaseQuery(BaseModel):
    user_message: str


@app.get("/")
def heartbeat_ping():
    return {"status": "ok", "message": "Service is working"}


@app.post("/chat", response_model=BaseResponse)
async def gptcooker(query: BaseQuery, request: Request):
    logger.info("Entering chat endpoint")
    if not hasattr(request.app.state, "rag_assistant"):
        raise HTTPException(
            status_code=503, detail="RAG pipeline not yer properly loaded"
        )

    logger.info(f"Exchange about : {query.user_message}")
    inputs = {
        "messages": [
            ("user", query.user_message),
            # ("user", "When was born Franklin Roosevelt?"),
        ]
    }
    retriever_used = False
    request_result = request.app.state.rag_assistant.graph.invoke(input=inputs)
    for sub_message in request_result.get("messages"):
        if isinstance(sub_message, ToolMessage):
            retriever_used = True

    return BaseResponse(
        **{"response": sub_message.content, "retriever_used": retriever_used}
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
