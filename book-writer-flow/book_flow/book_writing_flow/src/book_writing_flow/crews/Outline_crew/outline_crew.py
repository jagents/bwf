from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool
from pydantic import BaseModel
from book_writing_flow.tools.custom_tool import BrightDataWebSearchTool

# Configure LLM
llm = LLM(
    model="gpt-4-turbo",
    temperature=0.7,
    max_tokens=4000
)

# Set the same LLM for function calling
function_calling_llm = llm

class Outline(BaseModel):
    """Outline of the book"""
    total_chapters: int
    titles: list[str]

@CrewBase
class OutlineCrew:
    """Outline Crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def research_agent(self) -> Agent:
        return Agent(config=self.agents_config["research_agent"],
                     tools=[BrightDataWebSearchTool()],
                     llm=llm)

    @task
    def research_task(self) -> Task:
        return Task(config=self.tasks_config["research_task"])
    
    @agent
    def outline_writer(self) -> Agent:
        return Agent(config=self.agents_config["outline_writer"],
                     llm=llm)

    @task
    def write_outline(self) -> Task:
        return Task(config=self.tasks_config["write_outline"],
                    output_pydantic=Outline)

    @crew
    def crew(self) -> Crew:
        """Creates the Outline Crew"""

        return Crew(agents=self.agents,
                    tasks=self.tasks,
                    process=Process.sequential,
                    verbose=True)
