import os
from ragaai_catalyst import RagaAICatalyst, init_tracing
from ragaai_catalyst.tracers import Tracer
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool


load_dotenv()

# Initialize RagaAI Catalyst
catalyst = RagaAICatalyst(
    access_key=os.getenv('CATALYST_ACCESS_KEY'), 
    secret_key=os.getenv('CATALYST_SECRET_KEY'), 
    base_url=os.getenv('CATALYST_BASE_URL')
)
# Initialize tracer
tracer = Tracer(
    project_name=os.getenv('PROJECT_NAME'),
    dataset_name=os.getenv('DATASET_NAME'),
    tracer_type="agentic/llamaindex",
)
init_tracing(catalyst=catalyst, tracer=tracer)

@tool
def write_to_file(filename: str, content: str) -> str:
    """Write content to a file with the specified filename."""
    with open(filename, "w") as f:
        f.write(content)
    return f"Content successfully written to {filename}"


market_researcher = Agent(
    role="Market Research Analyst",
    goal="Identify market opportunities and analyze potential business ideas",
    backstory="You are an experienced market analyst with expertise in identifying profitable business opportunities and market gaps.",
    verbose=True,
    allow_delegation=False
        )



buisness_strategy = Agent(
        role="Business Strategist",
        goal="Develop a comprehensive business strategy and revenue model",
        backstory="You are a strategic thinker with experience in business model development and strategic planning.",
        verbose=True,
        allow_delegation=False
    )



financial_planner=Agent(
        role="Financial Planner",
        goal="Create detailed financial projections and write the complete business plan",
        backstory="You are a financial expert skilled in creating business plans and financial forecasts.",
        verbose=True,
        allow_delegation=False
    )



research_task= Task(
        description="""Conduct market research and propose a innovative business idea. 
                    Include target market, problem being solved, and unique value proposition.""",
        expected_output="A detailed market analysis and business idea proposal (2-3 paragraphs).",
        agent=market_researcher
    )



create_strategy_task=Task(
        description="""Develop a business strategy including:
                    - Business model
                    - Revenue streams
                    - Marketing approach
                    - Competitive analysis""",
        expected_output="A comprehensive business strategy document with all key components.",
        agent=buisness_strategy,
        context=[research_task]
    )



create_planning_task= Task(
        description="""Create a complete business plan including:
                    - Executive summary
                    - Financial projections
                    - Implementation timeline
                    Save the final plan as 'business_plan.md'""",
        expected_output="A markdown file containing the complete business plan.",
        agent=financial_planner,
        context=[create_strategy_task]
    )
def main():
    # Create agents
    researcher = market_researcher
    strategist = buisness_strategy
    planner = financial_planner


    market_research = research_task
    strategy = create_strategy_task
    planning = create_planning_task

    # Create and configure crew
    crew = Crew(
        agents=[researcher, strategist, planner],
        tasks=[market_research, strategy, planning],
        process=Process.sequential,
        verbose=True
    )

    print("Starting the CrewAI Business Plan Generation process...")
    result = crew.kickoff()

    print("\nProcess completed! Final output:")
    print(result)

    try:
        with open("business_plan.md", "r") as file:
            print("\nGenerated Business Plan Content:")
            print(file.read())
    except FileNotFoundError:
        print("Business plan file not found. Check the financial planner agent's execution.")
    
    return result

if __name__ == "__main__":
    with tracer:
        main()
    tracer.get_upload_status()