from dotenv import load_dotenv
from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

load_dotenv()

llm = init_chat_model(
    "deepseek:deepseek-chat"
)


class MessageClassifier(BaseModel):
    message_type: Literal["emotional", "logical"] = Field(
        ...,
        description="Classify if the message requires an emotional (therapist) or logical response."
    )

class RateScore(BaseModel):
    score: int = Field(
        ...,
        ge=1, le=10,  # 限制分数在1-10之间
        description="Score the quality of the response to the user's question, from 1 (worst) to 10 (best)."
    )
    reason: str = Field(
        ...,
        description="Brief reason for the score (within 100 words)."
    )

class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: str | None
    score: int | None  # 评分结果
    score_reason: str | None  # 评分理由

def classify_message(state: State):
    last_message = state["messages"][-1]
    classifier_llm = llm.with_structured_output(MessageClassifier)

    result = classifier_llm.invoke([
        {
            "role": "system",
            "content": """Classify the user message as either:
            - 'emotional': if it asks for emotional support, therapy, deals with feelings, or personal problems
            - 'logical': if it asks for facts, information, logical analysis, or practical solutions
            """
        },
        {"role": "user", "content": last_message.content}
    ])
    return {"message_type": result.message_type}


def router(state: State):
    message_type = state.get("message_type", "logical")
    if message_type == "emotional":
        return {"next": "therapist"}

    return {"next": "logical"}


def therapist_agent(state: State):
    last_message = state["messages"][-1]

    messages = [
        {"role": "system",
         "content": """You are a compassionate therapist. Focus on the emotional aspects of the user's message.
                        Show empathy, validate their feelings, and help them process their emotions.
                        Ask thoughtful questions to help them explore their feelings more deeply.
                        Avoid giving logical solutions unless explicitly asked."""
         },
        {
            "role": "user",
            "content": last_message.content
        }
    ]
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}]}


def logical_agent(state: State):
    last_message = state["messages"][-1]

    messages = [
        {"role": "system",
         "content": """You are a purely logical assistant. Focus only on facts and information.
            Provide clear, concise answers based on logic and evidence.
            Do not address emotions or provide emotional support.
            Be direct and straightforward in your responses."""
         },
        {
            "role": "user",
            "content": last_message.content
        }
    ]
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}]}


def rate_agent(state: State):
    """评分agent：基于用户问题和agent回答打分"""
    # 获取用户问题和agent回答
    user_question = state["messages"][-2].content  # 倒数第二条是用户消息
    agent_answer = state["messages"][-1].content  # 最后一条是agent回答

    # 调用LLM生成评分（结构化输出）
    rate_llm = llm.with_structured_output(RateScore)
    result = rate_llm.invoke([
        {
            "role": "system",
            "content": """Score the response quality (1-10) based on:
            - How well it addresses the user's question
            - Relevance and accuracy
            - Clarity of expression
            Provide a brief reason for the score.
            """
        },
        {"role": "user", "content": f"User question: {user_question}"},
        {"role": "assistant", "content": f"Agent answer: {agent_answer}"}
    ])

    # 更新状态中的评分信息
    return {
        "score": result.score,
        "score_reason": result.reason
    }

graph_builder = StateGraph(State)

graph_builder.add_node("classifier", classify_message)
graph_builder.add_node("router", router)
graph_builder.add_node("therapist", therapist_agent)
graph_builder.add_node("logical", logical_agent)
graph_builder.add_node("rate", rate_agent)

graph_builder.add_edge(START, "classifier")
graph_builder.add_edge("classifier", "router")
graph_builder.add_conditional_edges(
    "router",
    lambda state: state.get("next"),
    {"therapist": "therapist", "logical": "logical"}
)


graph_builder.add_edge("therapist", "rate")
graph_builder.add_edge("logical", "rate")
graph_builder.add_edge("rate", END)

graph = graph_builder.compile()


def run_chatbot():
    state = {"messages": [], "message_type": None}

    while True:
        user_input = input("Message: ")
        if user_input == "exit":
            print("Bye")
            break

        state["messages"] = state.get("messages", []) + [
            {"role": "user", "content": user_input}
        ]

        state = graph.invoke(state)

        if state["score"] is not None and state["messages"]:
            last_answer = state["messages"][-1].content
            # 分数放在答案前面，格式：[评分] 回答内容（理由）
            print(f"Assistant: [Score: {state['score']}/10] {last_answer}")
            # print(f"Reason: {state['score_reason']}\n")


if __name__ == "__main__":
    run_chatbot()
