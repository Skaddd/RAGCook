from typing import Dict, Union

from langchain_core.messages import SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.utils.utils_llm import load_dense_embedding_model, load_llm_model
from src.vectorstore import load_vectorstore_as_retriever_tool


class RAGCook:
    """Generate a Simple Rag pipeline.

    This class aims at producing a really simple RAG
    pipeline using Langchain and langgraph implmentations
    to build a simple workflow composed of a Retriever Tool.
    """

    def __init__(
        self,
        persistent_directory_path: str,
        collection_name: str,
        retriever_name: str,
        retriever_description: str,
        llm_config: Dict[str, Union[str, int]],
    ):
        """Constructor of RAGCook.

        Apart from instanciating attributes, this constructor
        also instanticate the langgraph workflow graph.
        Args:
            persistent_directory_path (str): vectorstore path location.
            collection_name (str): collection name associated to
            the vectorstore.
            retriever_name (str): Tool name for the retriever tool.
            retriever_description (str): Description for the retriever tool.
            llm_config (Dict[str, Union[str, int]]): Specific configuration for
            the llm and embedding models.
        """

        self.persistant_db_directory = persistent_directory_path
        self.collection_name = collection_name

        self.retriever_tool = load_vectorstore_as_retriever_tool(
            persistent_directory_path=self.persistant_db_directory,
            collection_name=self.collection_name,
            retriever_name=retriever_name,
            retriever_description=retriever_description,
            dense_embedding_model=load_dense_embedding_model(
                embedding_config=llm_config["embedding_config"]
            ),
        )
        self.llm_model = load_llm_model(llm_config=llm_config["chat_config"])
        self.model_tool = self.llm_model.bind_tools([self.retriever_tool])

        workflow = self._setup_workflow()
        self.graph = workflow.compile()

    def _setup_workflow(self):
        """Create Langgraph workflow."""
        workflow = StateGraph(MessagesState)

        workflow.add_node("retrieve_or_respond", self.retrieve_or_answer)
        workflow.add_node("retrieve", ToolNode([self.retriever_tool]))
        workflow.add_node("generate", self.generate)

        workflow.add_edge(START, "retrieve_or_respond")
        workflow.add_conditional_edges(
            "retrieve_or_respond",
            tools_condition,
            {"tools": "retrieve", END: END},
        )
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)

        return workflow

    def retrieve_or_answer(self, state: MessagesState):
        """Retrieve from vectorstore or directly answer."""
        response = self.model_tool.invoke(state["messages"])
        return {"messages": [response]}

    def generate(self, state: MessagesState):
        """Generate answer."""
        last_tool_messages = []
        for message in reversed(state["messages"]):
            if message.type == "tool":
                last_tool_messages.append(message)
            else:
                break
        tool_messages = last_tool_messages[::-1]

        # Format into prompt
        docs_content = "\n\n".join(doc.content for doc in tool_messages)
        system_message_content = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            f"{docs_content}"
        )
        conversation_messages = [
            message
            for message in state["messages"]
            if message.type in ("human", "system")
            or (message.type == "ai" and not message.tool_calls)
        ]
        prompt = [
            SystemMessage(system_message_content)
        ] + conversation_messages

        # Run
        response = self.llm_model.invoke(prompt)
        return {"messages": [response]}


if __name__ == "__main__":
    from utils.helpers import global_loading_configuration

    config_dir = r"/home/mateo/projects/RAGCook/conf"

    conf = global_loading_configuration(configuration_dir=config_dir)
    rag_assistant = RAGCook(
        persistent_directory_path=conf["persistent_dir_path"],
        collection_name=conf["db_collection_name"],
        retriever_name=conf["retriever_tool_name"],
        retriever_description=conf["retriever_tool_description"],
        llm_config=conf["llm_config"],
    )
    query = "Peut-tu me donner les étapes à suivre si je veux cuisiner des aiguillettes de canard"

    inputs = {
        "messages": [
            ("user", query),
            # ("user", "When was born Franklin Roosevelt?"),
        ]
    }
    # import pprint
    # for msg in rag_assistant.graph.stream(inputs):
    #     if msg.get("retrieve"):
    #         print("USED TOOL")
    # pprint.pprint(msg)
    # pprint.pprint("\n---\n")

    print(rag_assistant.graph.invoke(input=inputs))
