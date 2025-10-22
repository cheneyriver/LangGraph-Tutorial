from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver  # ✅ 导入短期记忆组件
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

# 初始化 LLM
llm = init_chat_model("deepseek:deepseek-chat")


# 定义状态结构
class State(TypedDict):
    messages: Annotated[list, add_messages]


# 构建图
graph_builder = StateGraph(State)


def chatbot(state: State):
    """每个节点执行时，从 state 中取历史消息，生成回复"""
    return {"messages": [llm.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# ✅ 创建 checkpointer —— 用于保存每个 thread_id 的状态
checkpointer = InMemorySaver()

# ✅ 编译图时注入 checkpointer
graph = graph_builder.compile(checkpointer=checkpointer)

# 使用固定的 thread_id 代表一个会话
thread_id = "chat-session-1"

print("🤖 Chatbot started! Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Bye!")
        break

    # 调用图并指定 thread_id
    state = graph.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        {"configurable": {"thread_id": thread_id}}  # ✅ 关键参数：用来定位记忆
    )

    # 输出最新回复
    print("Bot:", state["messages"][-1].content)
