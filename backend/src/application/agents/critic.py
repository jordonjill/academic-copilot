from typing import Dict
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers.string import StrOutputParser
from src.domain.state import GlobalState, ResearchCritic, Resource
from src.infrastructure.tools import crawl_search
from src.infrastructure.config.prompt import CRITIC_QUERY_GENERATION_PROMPT, CRITIC_EVALUATION_PROMPT
from src.infrastructure.config.config import MAX_SEARCHES, MAX_VALIDATION_ATTEMPTS

def critic_node(state: GlobalState, llm: BaseLanguageModel) -> Dict:

    idea_validation_attempts = state.get("idea_validation_attempts", 0) + 1
    if idea_validation_attempts > MAX_VALIDATION_ATTEMPTS:
        research_critic = ResearchCritic(
            is_valid=True,
            feedback=f"The idea could not be validated within {MAX_VALIDATION_ATTEMPTS} attempts. Approving by default."
        )
        return {
            "research_critic": research_critic,
            "idea_validation_attempts": idea_validation_attempts
        }
    
    search_count = state.get("search_count", 0)
    if search_count >= MAX_SEARCHES:
        research_critic = ResearchCritic(
            is_valid=True,
            feedback="Could not perform critique search as the search limit was reached. Approving by default."
        )
        return {
            "research_critic": research_critic,
            "idea_validation_attempts": idea_validation_attempts
        }

    research_creation = state["research_creation"]
    research_idea = f"Idea: {research_creation.research_idea}"

    query_gen_chain = CRITIC_QUERY_GENERATION_PROMPT | llm | StrOutputParser()
    query = query_gen_chain.invoke({"research_idea": research_idea})
    raw_results = crawl_search.invoke({"query": query})

    new_resources = []
    for result in raw_results:
        if isinstance(result, dict) and "error" not in result:
            try:
                new_resources.append(Resource(**result))
            except Exception as e:
                print(f"[!] Warning: Could not create Resource object from data: {result}. Error: {e}")

    search_count = search_count + 1
    eval_chain = CRITIC_EVALUATION_PROMPT | llm.with_structured_output(ResearchCritic)

    search_resources = "\n\n---\n\n".join(
        [f"Source URI: {r.uri}\nTitle: {r.title}\nAbstract: {r.content}" for r in new_resources]
    )

    research_critic = eval_chain.invoke({
        "research_idea": research_idea,
        "search_results": search_resources if new_resources else "No new results found."
    })

    # operator.add reducer: 仅返回新增资源
    return {
        "research_critic": research_critic,
        "retrieved_resources": new_resources,
        "idea_validation_attempts": idea_validation_attempts,
        "search_count": search_count,
    }