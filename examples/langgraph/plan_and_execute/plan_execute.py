import os
from dotenv import load_dotenv
from ragaai_catalyst import RagaAICatalyst, init_tracing
from ragaai_catalyst.tracers import Tracer
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
import operator
from typing import Annotated, List, Tuple, Union, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph, START
from IPython.display import Image, display

# Load environment variables
load_dotenv()



# Initialize RagaAI Catalyst
def initialize_catalyst():
    """Initialize RagaAI Catalyst using environment credentials."""
    catalyst = RagaAICatalyst(
    access_key="1q2igAYCIlpSBufkdB6f",
    secret_key="yG6TJOgES8D9jAi9OI0X6SgvZNtkcFvkOruukJay",
    base_url="https://llm-dev5.ragaai.ai/api")
    
    tracer = Tracer(
    project_name='langgraph',
    dataset_name='dataset',
    tracer_type="Agentic",
)
    
    init_tracing(catalyst=catalyst, tracer=tracer)

def initialize_models(model_name: str = "gpt-4o-mini", temperature: float = 0.5, max_results: int = 2):
    """Initialize the language model and search tool."""
    llm = ChatOpenAI(model=model_name, temperature=temperature)
    tavily_tool = TavilySearchResults(max_results=max_results)
    return llm, tavily_tool

# Initialize default instances
initialize_catalyst()
llm, tavily_tool = initialize_models()


# Type definitions
class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str

class Plan(BaseModel):
    """Plan to follow in future"""
    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )

class Response(BaseModel):
    """Response to user."""
    response: str

class Act(BaseModel):
    """Action to perform."""
    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )

# Prompt templates
planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.""",
        ),
        ("placeholder", "{messages}"),
    ]
)

replanner_prompt = ChatPromptTemplate.from_template(
    """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

Your objective was this:
{input}

Your original plan was this:
{plan}

You have currently done the follow steps:
{past_steps}

Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan."""
)

def setup_tools():
    """Set up the tools for the agent."""
    return [TavilySearchResults(max_results=3)]

def setup_agent(tools):
    """Set up the agent with tools."""
    llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)
    prompt = "You are a helpful assistant."
    return create_react_agent(llm, tools, prompt=prompt)

def should_end(state: PlanExecute):
    """Determine if the workflow should end."""
    if "response" in state and state["response"]:
        return END
    # Add check for empty plan
    elif "plan" in state and not state["plan"]:
        return END
    else:
        return "agent"

async def execute_step(state: PlanExecute):
    """Execute a single step in the plan."""
    plan = state["plan"]
    # Add guard against empty plan
    if not plan:
        return {
            "response": "No more steps to execute.",
            "past_steps": []
        }
    
    plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
    task = plan[0]
    task_formatted = f"""For the following plan:
{plan_str}\n\nYou are tasked with executing step {1}, {task}."""
    agent_response = await agent_executor.ainvoke(
        {"messages": [("user", task_formatted)]}
    )
    return {
        "past_steps": [(task, agent_response["messages"][-1].content)],
        "plan": plan[1:]  # Remove the completed step
    }

async def plan_step(state: PlanExecute):
    """Create initial plan."""
    plan = await planner.ainvoke({"messages": [("user", state["input"])]})
    return {"plan": plan.steps}

async def replan_step(state: PlanExecute):
    """Replan based on execution results."""
    output = await replanner.ainvoke(state)
    if isinstance(output.action, Response):
        return {"response": output.action.response}
    else:
        return {"plan": output.action.steps}

def setup_workflow():
    """Set up the workflow graph."""
    workflow = StateGraph(PlanExecute)
    
    # Add nodes
    workflow.add_node("planner", plan_step)
    workflow.add_node("agent", execute_step)
    workflow.add_node("replan", replan_step)
    
    # Add edges
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "agent")
    workflow.add_edge("agent", "replan")
    
    workflow.add_conditional_edges(
        "replan",
        should_end,
        ["agent", END],
    )
    
    return workflow.compile()

# Initialize global variables

tracer = initialize_catalyst()
tools = setup_tools()
agent_executor = setup_agent(tools)
planner = None
replanner = None


async def main():
    """Main execution function."""
    # Setup prompts
    global planner, replanner
    planner = planner_prompt | ChatOpenAI(
        model="gpt-4",
        temperature=0
    ).with_structured_output(Plan)
    
    replanner = replanner_prompt | ChatOpenAI(
        model="gpt-4",
        temperature=0
    ).with_structured_output(Act)
    
    # Setup and compile workflow
    app = setup_workflow()
    
    # Optional: Display workflow graph
    try:
        display(Image(app.get_graph(xray=True).draw_mermaid_png()))
    except:
        print("Could not display workflow graph")
    
    # Run the workflow
    config = {"recursion_limit": 50}
    inputs = {
        "input": "what is the hometown of the mens 2024 Australia open winner?",
        "plan": [],
        "past_steps": [],
        "response": ""
    }
    
    print("\nExecuting workflow...")
    async for event in app.astream(inputs, config=config):
        for k, v in event.items():
            if k != "__end__":
                print(f"{k}: {v}")
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())