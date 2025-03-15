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

class Chapter(BaseModel):
    """Chapter of the book"""
    title: str
    content: str

@CrewBase
class ChapterWriterCrew:
    """Chapter Writer Crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def topic_researcher(self) -> Agent:
        return Agent(config=self.agents_config["topic_researcher"],
                     tools=[BrightDataWebSearchTool()],
                     llm=llm)

    @task
    def research_topic(self) -> Task:
        return Task(config=self.tasks_config["research_topic"])

    @agent
    def writer(self) -> Agent:
        return Agent(config=self.agents_config["writer"],
                     llm=llm)

    @task
    def write_chapter(self) -> Task:
        return Task(config=self.tasks_config["write_chapter"],
                    output_pydantic=Chapter)

    @crew
    def crew(self) -> Crew:
        """Creates the Research Crew"""

        return Crew(agents=self.agents,
                    tasks=self.tasks,
                    process=Process.sequential,
                    verbose=True)
