from dotenv import load_dotenv
from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
import os

load_dotenv()

# 初始化DeepSeek模型
llm = init_chat_model(
    "deepseek:deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY")
)


# 1. 辩论状态定义（存储用户问题、辩论历史、汇总结果等）
class DebateState(TypedDict):
    user_question: str  # 用户原始问题
    debate_history: Annotated[list, add_messages]  # 辩论历史（包含两个agent的交替回应）
    debate_round: int  # 辩论轮次（控制最大轮次）
    final_answer: str | None  # 汇总agent生成的最终回答
    score: int | None  # 评分结果
    score_reason: str | None  # 评分理由


# 2. 评分结构体（复用之前的定义）
class RateScore(BaseModel):
    score: int = Field(..., ge=1, le=10, description="评分1-10")
    reason: str = Field(..., description="评分理由（100字内）")


# 3. 节点函数定义
def init_debate(state: DebateState):
    """初始化辩论：将用户问题同时发给两个agent，生成初始观点"""
    user_question = state["user_question"]

    # logical_agent初始观点
    logical_init = llm.invoke([
        {"role": "system", "content": "你是逻辑助手，针对问题先给出理性分析（1-2句话）"},
        {"role": "user", "content": user_question}
    ])
    # therapist_agent初始观点
    therapist_init = llm.invoke([
        {"role": "system", "content": "你是情感咨询师，针对问题先给出情感视角分析（1-2句话）"},
        {"role": "user", "content": user_question}
    ])

    # 初始化辩论历史（用assistant作为role，内容前缀区分）
    debate_history = [
        {"role": "assistant", "content": f"[逻辑助手] {logical_init.content}"},  # 前缀标识
        {"role": "assistant", "content": f"[情感咨询师] {therapist_init.content}"}  # 前缀标识
    ]
    return {
        "debate_history": debate_history,
        "debate_round": 1  # 初始轮次为1
    }


def check_debate_end(state: DebateState):
    """判断辩论是否结束（最大3轮）"""
    max_rounds = 3  # 最多3轮辩论
    return state["debate_round"] >= max_rounds


def logical_debate(state: DebateState):
    """logical_agent回应（修正role和内容前缀）"""
    user_question = state["user_question"]
    debate_history = state["debate_history"]

    prompt = [
        {"role": "system", "content": f"""你是逻辑助手，针对用户问题「{user_question}」，
            基于之前的辩论历史，回应情感咨询师的观点（指出逻辑补充点，1-2句话）。
            注意：你的回答会被标记为「[逻辑助手]」，无需自己添加前缀。"""},
        *debate_history
    ]

    reply = llm.invoke(prompt)
    # 用assistant作为role，内容加前缀
    return {
        "debate_history": debate_history + [{"role": "assistant", "content": f"[逻辑助手] {reply.content}"}],
        "debate_round": state["debate_round"] + 1
    }


def therapist_debate(state: DebateState):
    """therapist_agent回应（修正role和内容前缀）"""
    user_question = state["user_question"]
    debate_history = state["debate_history"]

    prompt = [
        {"role": "system", "content": f"""你是情感咨询师，针对用户问题「{user_question}」，
            基于之前的辩论历史，回应逻辑助手的观点（指出情感补充点，1-2句话）。
            注意：你的回答会被标记为「[情感咨询师]」，无需自己添加前缀。"""},
        *debate_history
    ]

    reply = llm.invoke(prompt)
    # 用assistant作为role，内容加前缀
    return {
        "debate_history": debate_history + [{"role": "assistant", "content": f"[情感咨询师] {reply.content}"}],
    }


def summarize_agent(state: DebateState):
    """汇总agent：整合辩论历史，生成兼顾逻辑与情感的最终回答（修复对象属性访问）"""
    user_question = state["user_question"]
    debate_history = state["debate_history"]

    # 关键修复：用AIMessage的content属性访问内容（而非字典下标）
    # 辩论历史格式化：直接取每个消息的content（已包含[逻辑助手]/[情感咨询师]前缀）
    formatted_history = "\n".join([m.content for m in debate_history])

    prompt = [
        {"role": "system", "content": f"""你是汇总助手，需要：
            1. 紧扣用户问题「{user_question}」
            2. 整合以下辩论历史中两个agent的观点（逻辑助手关注理性，情感咨询师关注心理）
            3. 生成一个兼顾理性分析和情感关怀的完整回答（3-5句话），语言自然流畅
            辩论历史：
            {formatted_history}"""},
        {"role": "user", "content": "请基于上述辩论历史，生成最终回答"}
    ]

    final_reply = llm.invoke(prompt)
    return {"final_answer": final_reply.content}


def rate_agent(state: DebateState):
    """评分agent：基于用户问题和汇总结果打分"""
    user_question = state["user_question"]
    final_answer = state["final_answer"]

    rate_llm = llm.with_structured_output(RateScore)
    result = rate_llm.invoke([
        {"role": "system", "content": """评分标准：
            - 是否同时兼顾逻辑准确性和情感相关性
            - 回答完整性和清晰度
            请打分（1-10）并说明理由。"""},
        {"role": "user", "content": f"用户问题：{user_question}"},
        {"role": "assistant", "content": f"最终回答：{final_answer}"}
    ])

    return {
        "score": result.score,
        "score_reason": result.reason
    }


# 4. 构建LangGraph流程图
graph_builder = StateGraph(DebateState)

# 添加节点
graph_builder.add_node("init_debate", init_debate)  # 初始化辩论（生成初始观点）
graph_builder.add_node("logical_debate", logical_debate)  # logical回应
graph_builder.add_node("therapist_debate", therapist_debate)  # therapist回应
graph_builder.add_node("summarize", summarize_agent)  # 汇总
graph_builder.add_node("rate", rate_agent)  # 评分

# 定义流程逻辑
graph_builder.add_edge(START, "init_debate")  # 开始→初始化辩论

# 辩论循环：先判断是否结束，未结束则进入逻辑→情感交替回应
graph_builder.add_conditional_edges(
    "init_debate",
    check_debate_end,  # 条件：是否达到最大轮次
    {
        True: "summarize",  # 结束→汇总
        False: "logical_debate"  # 未结束→逻辑agent先回应
    }
)

# 逻辑回应后→情感回应
graph_builder.add_edge("logical_debate", "therapist_debate")

# 情感回应后→再次判断是否结束辩论
graph_builder.add_conditional_edges(
    "therapist_debate",
    check_debate_end,
    {
        True: "summarize",
        False: "logical_debate"
    }
)

# 汇总→评分→结束
graph_builder.add_edge("summarize", "rate")
graph_builder.add_edge("rate", END)

# 编译流程图
graph = graph_builder.compile()


# 5. 运行聊天机器人
def run_chatbot():
    print("输入问题开始对话（输入exit退出）")
    while True:
        user_input = input("Message: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Bye!")
            break

        # 初始化状态（传入用户问题）
        initial_state = {
            "user_question": user_input,
            "debate_history": [],
            "debate_round": 0,
            "final_answer": None,
            "score": None,
            "score_reason": None
        }

        # 执行流程
        final_state = graph.invoke(initial_state)

        # 输出结果
        print(f"辩论过程：{final_state['debate_history']}\n")
        print(f"\n[评分: {final_state['score']}/10] 最终回答：{final_state['final_answer']}")
        print(f"评分理由：{final_state['score_reason']}\n")


if __name__ == "__main__":
    run_chatbot()