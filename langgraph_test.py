import asyncio
import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI

# 定义状态
class RagState(TypedDict):
    question: str
    retrieved_chunks: List[str]
    intent: str
    answer: str

# 配置 Azure OpenAI
# AZURE_ENDPOINT = "test"
# AZURE_API_KEY = "test"
AZURE_DEPLOYMENT = "gpt-5-chat"
API_VERSION = "2024-08-01-preview"

def get_llm():
    """获取 Azure OpenAI LLM 实例"""
    return AzureChatOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        azure_deployment=AZURE_DEPLOYMENT,
        api_key=AZURE_API_KEY,
        api_version=API_VERSION,
        temperature=0.7
    )

# Agent 1: Supervisor (Retrieval)
def supervisor_retrieval_agent(state: RagState) -> RagState:
    """Supervisor agent that retrieves relevant chunks"""
    question = state["question"]
    
    # 这里是mock检索，实际应该连接到你的向量数据库
    # 你可以后续替换为真实的检索逻辑
    mock_chunks = [
        "人工智能是计算机科学的一个分支，旨在创建能够执行通常需要人类智能的任务的机器。",
        "机器学习是人工智能的一个子集，允许系统从数据中学习并改进性能，而无需显式编程。",
        "深度学习使用人工神经网络来处理大量数据并识别复杂模式。"
    ]
    
    print(f"[Supervisor] 正在检索与问题相关的信息: {question}")
    state["retrieved_chunks"] = mock_chunks
    
    return state

# Agent 2: Intent Identifier
def intent_identifier_agent(state: RagState) -> RagState:
    """Identify the intent of the question"""
    question = state["question"]
    
    # 使用 Azure OpenAI
    llm = get_llm()
    
    # 识别意图
    intent_prompt = f"""根据以下问题，识别用户的意图。只返回一个简短的意图标签（如：definition, comparison, explanation等）。
    
问题: {question}

只返回意图标签，不需要其他解释。"""
    
    try:
        response = llm.invoke([HumanMessage(content=intent_prompt)])
        intent = response.content.strip()
        print(f"[Intent Identifier] 识别的意图: {intent}")
        state["intent"] = intent
    except Exception as e:
        print(f"[Intent Identifier] 错误: {e}")
        state["intent"] = "general"
    
    return state

# Agent 3: Answer Generator
def answer_generator_agent(state: RagState) -> RagState:
    """Generate answer based on retrieved chunks and intent"""
    question = state["question"]
    chunks = state["retrieved_chunks"]
    intent = state["intent"]
    
    # 使用 Azure OpenAI
    llm = get_llm()
    
    # 生成答案
    chunks_text = "\n".join([f"- {chunk}" for chunk in chunks])
    answer_prompt = f"""基于以下检索到的信息回答用户的问题。用户的意图是: {intent}

检索到的相关信息:
{chunks_text}

用户问题: {question}

请提供一个清晰、准确且有帮助的答案。"""
    
    try:
        response = llm.invoke([HumanMessage(content=answer_prompt)])
        answer = response.content
        print(f"[Answer Generator] 生成的答案")
        state["answer"] = answer
    except Exception as e:
        print(f"[Answer Generator] 错误: {e}")
        state["answer"] = "无法生成答案，请稍后重试。"
    
    return state

# 创建流程图
def create_rag_graph():
    """创建RAG流程图"""
    workflow = StateGraph(RagState)
    
    # 添加节点
    workflow.add_node("supervisor_retrieval", supervisor_retrieval_agent)
    workflow.add_node("intent_identifier", intent_identifier_agent)
    workflow.add_node("answer_generator", answer_generator_agent)
    
    # 添加边
    workflow.set_entry_point("supervisor_retrieval")
    workflow.add_edge("supervisor_retrieval", "intent_identifier")
    workflow.add_edge("intent_identifier", "answer_generator")
    workflow.add_edge("answer_generator", END)
    
    # 编译流程图
    graph = workflow.compile()
    
    return graph

# 主函数
def main():
    """主函数"""
    # 创建流程图
    graph = create_rag_graph()
    
    # 示例输入
    initial_state = RagState(
        question="什么是人工智能？",
        retrieved_chunks=[],
        intent="",
        answer=""
    )
    
    # 运行流程
    print("=" * 60)
    print("Multi-Agent RAG Demo 开始")
    print("=" * 60)
    
    try:
        result = graph.invoke(initial_state)
        print("\n" + "=" * 60)
        print("最终答案:")
        print("=" * 60)
        print(result["answer"])
        print("=" * 60)
    except Exception as e:
        print(f"执行出错: {e}")

if __name__ == "__main__":
    main()