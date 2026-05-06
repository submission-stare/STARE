from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from state import AgentState
from utils.config import cfg
from utils.llm import get_llm
from utils.step_logger import get_step_logger
from typing import Literal
from langchain_core.output_parsers import PydanticOutputParser
import logging

logger = logging.getLogger(__name__)

class RouterDecision(BaseModel):
    next_node: str = Field(description="The exact name of the next node to execute. Must be one of: 'hypothesis_maker', 'hypothesis_investigator', 'report_generator'.")
    reasoning: str = Field(description="Brief reasoning for choosing this next node.")

def data_analyst(state: AgentState):
    """
    The central router agent. Analyzes overall progress and decides the next node.
    """
    logger.info("Data Analyst Router started.")
    llm = get_llm(temperature=cfg("temperatures.data_analyst", 0.0))
    parser = PydanticOutputParser(pydantic_object=RouterDecision)
    
    hypotheses = state.get("hypotheses", [])
    plot_paths = state.get("plot_paths", [])
    messages = state.get("messages", [])
    analysis_mode = state.get("analysis_mode", "report")
    
    logger.info(f"Data Analyst State: {len(hypotheses)} hypotheses, {len(plot_paths)} plots.")
    
    if not hypotheses:
        step_logger = get_step_logger()
        if step_logger:
            step_logger.log_router_decision("hypothesis_maker", "No hypotheses exist yet.", {"hypotheses_count": 0, "plots_count": 0})
        return {"next_node": "hypothesis_maker", "messages": [HumanMessage(content="Routing to hypothesis_maker. No hypotheses exist.")]}
        
    recent_context = "\n".join([f"{m.type}: {str(m.content)[:500]}" for m in messages[-4:]])
    
    # If plots are already generated, we just need to report and finish.
    if plot_paths:
        if analysis_mode == "report":
            next_node = "code_hypothesis_investigator"
            reasoning = "Plots are already generated. Collect code-grounded evidence before the final consensus."
        else:
            next_node = "report_generator"
            reasoning = "Plots are already generated. Time to write the report."
        step_logger = get_step_logger()
        if step_logger:
            step_logger.log_router_decision(next_node, reasoning, {
                "hypotheses_count": len(hypotheses),
                "plots_count": len(plot_paths),
            })
        msg = HumanMessage(content=f"Decision: {next_node}. Reasoning: {reasoning}")
        return {"next_node": next_node, "messages": [msg]}

    if hypotheses and analysis_mode == "report":
        next_node = "hypothesis_investigator"
        reasoning = "Hypotheses exist but no plots have been generated yet. Proceeding to investigation."
        step_logger = get_step_logger()
        if step_logger:
            step_logger.log_router_decision(next_node, reasoning, {
                "hypotheses_count": len(hypotheses),
                "plots_count": len(plot_paths),
            })
        msg = HumanMessage(content=f"Decision: {next_node}. Reasoning: {reasoning}")
        return {"next_node": next_node, "messages": [msg]}

    system_prompt = (
        "You are the Data Analyst Router for a quantitative post-mortem trading agent.\n"
        "Decide the next step based on the current state:\n"
        "- 'hypothesis_maker': If no hypotheses exist to investigate.\n"
        "- 'hypothesis_investigator': If hypotheses exist but need to be validated via code execution and plotting.\n"
        "- 'report_generator': If analysis is complete and plots are generated.\n\n"
        f"{parser.get_format_instructions()}"
    )
    
    user_prompt = (
        f"Hypotheses Count: {len(hypotheses)}\n"
        f"Plots Generated: {len(plot_paths)}\n\n"
        f"Recent Messages:\n{recent_context}\n\n"
        "What is the next step?"
    )
    
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    
    try:
        decision = parser.invoke(response)
        next_node = decision.next_node
        reasoning = decision.reasoning
    except Exception as e:
        # Fallback in case of parsing error
        content = str(response.content).lower()
        if "report_generator" in content:
            next_node = "report_generator"
        elif "hypothesis_investigator" in content:
            next_node = "hypothesis_investigator"
        else:
            next_node = "hypothesis_maker"
        reasoning = f"Fallback routing logic due to parsing error. Raw response: {response.content}"
    
    # Safety catch
    if plot_paths and next_node != "report_generator":
        next_node = "report_generator"
        reasoning = "Forced routing to report_generator because plots exist."

    step_logger = get_step_logger()
    if step_logger:
        step_logger.log_router_decision(next_node, reasoning, {
            "hypotheses_count": len(hypotheses),
            "plots_count": len(plot_paths),
        })

    msg = HumanMessage(content=f"Decision: {next_node}. Reasoning: {reasoning}")
    
    return {"next_node": next_node, "messages": [msg]}
