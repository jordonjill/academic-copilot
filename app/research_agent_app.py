import asyncio
from datetime import datetime
from functools import partial
from typing import Dict, Any, Optional, Callable

from dotenv import load_dotenv
from langchain_ollama.chat_models import ChatOllama
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph

from src.state import GraphState
from src.agents import planner_node, researcher_node, synthesizer_node, critic_node, reporter_node

load_dotenv() 

class ResearchAgentApp:
    
    def __init__(self, model_type: str = "ollama", model_name: str = "llama3.1:8b", temperature: float = 0):
        self.model_type = model_type
        self.model_name = model_name
        self.temperature = temperature
        self.llm = self._initialize_llm()
        self.graph = self._build_graph()
        self.current_state = None
        
    def _initialize_llm(self):
        if self.model_type == "ollama":
            return ChatOllama(model=self.model_name, temperature=self.temperature)
        elif self.model_type == "gemini":
            return init_chat_model(
                model=self.model_name, 
                model_provider="google_genai",
                temperature=self.temperature
            )
        elif self.model_type == "openai":
            return init_chat_model(
                model=self.model_name,
                model_provider="openai", 
                temperature=self.temperature
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _build_graph(self):
        planner = partial(planner_node, llm=self.llm)
        researcher = researcher_node
        synthesizer = partial(synthesizer_node, llm=self.llm)
        critic = partial(critic_node, llm=self.llm)
        reporter = partial(reporter_node, llm=self.llm)

        def route_planning(state: GraphState):
            plan = state["research_plan"]
            if plan.step_type == "search":
                return "researcher"
            elif plan.step_type == "synthesize":
                return "synthesizer"

        def route_critic(state: GraphState):
            if state["research_critic"].is_valid:
                return "reporter"
            else:
                return "synthesizer"
        
        graph_builder = StateGraph(GraphState)

        graph_builder.add_node("planner", planner)
        graph_builder.add_node("researcher", researcher)
        graph_builder.add_node("synthesizer", synthesizer)
        graph_builder.add_node("critic", critic)
        graph_builder.add_node("reporter", reporter)
        
        graph_builder.set_entry_point("planner")
        
        graph_builder.add_edge("researcher", "planner") 
        graph_builder.add_edge("synthesizer", "critic")
        
        graph_builder.add_conditional_edges(
            "planner",
            route_planning,
            {
                "researcher": "researcher",
                "synthesizer": "synthesizer"
            }
        )
        
        graph_builder.add_conditional_edges(
            "critic",
            route_critic,
            {
                "synthesizer": "synthesizer",
                "reporter": "reporter"
            }
        )
        
        graph_builder.set_finish_point("reporter")
        
        return graph_builder.compile()
    
    async def run_research_async(
        self, 
        initial_topic: str, 
        websocket_send: Optional[Callable] = None,
        recursion_limit: int = 20
    ) -> Dict[str, Any]:

        inputs = {"initial_topic": initial_topic}
        
        try:
            if websocket_send:
                await websocket_send({
                    "type": "status",
                    "message": f"Studying: {initial_topic}",
                    "timestamp": self._get_timestamp()
                })
            
            final_state = None
            step_count = 0
            
            async for step in self._async_stream(inputs, {"recursion_limit": recursion_limit}):
                step_count += 1
                node_name, output = next(iter(step.items()))

                final_state = step
                self.current_state = output
                
                print(f"Processing step {step_count}: {node_name}")  # Debug log

                step_info = {
                    "type": "step",
                    "step_number": step_count,
                    "node_name": node_name,
                    "message": f"{node_name} is working...",
                    "timestamp": self._get_timestamp()
                }

                if "research_plan" in output and output["research_plan"]:
                    plan = output["research_plan"]
                    step_info["details"] = {
                        "action_type": plan.step_type,
                        "query": getattr(plan, 'query', None),
                        "has_enough_content": getattr(plan, 'has_enough_content', False)
                    }
                
                if "retrieved_resources" in output:
                    resource_count = len(output["retrieved_resources"])
                    step_info["resource_count"] = resource_count
                
                if "research_creation" in output and output["research_creation"]:
                    step_info["research_gap"] = output["research_creation"].research_gap
                
                if "research_critic" in output and output["research_critic"]:
                    critic = output["research_critic"]
                    step_info["critic_result"] = {
                        "is_valid": critic.is_valid,
                        "feedback": critic.feedback if critic.feedback else None
                    }
                
                if websocket_send:
                    try:
                        await websocket_send(step_info)
                        print(f"Sent step info: {step_info}")
                    except Exception as e:
                        print(f"Failed to send step info: {e}") 
                
                await asyncio.sleep(0.5)
            
            final_result = self._extract_final_result(final_state)
            
            if websocket_send:
                await websocket_send({
                    "type": "completion",
                    "message": "Research process completed.",
                    "final_result": final_result,
                    "timestamp": self._get_timestamp()
                })
            
            return final_result
            
        except Exception as e:
            error_message = f"Error occurred during execution: {str(e)}"
            
            if websocket_send:
                await websocket_send({
                    "type": "error",
                    "message": error_message,
                    "timestamp": self._get_timestamp()
                })
            
            raise e
    
    async def _async_stream(self, inputs: Dict, config: Dict):
        loop = asyncio.get_event_loop()
        
        def sync_stream():
            return self.graph.stream(inputs, config)
        
        stream = await loop.run_in_executor(None, sync_stream)
        
        for step in stream:
            yield step
    
    def _extract_final_result(self, final_state: Optional[Dict]) -> Dict[str, Any]:
        if not final_state:
            return {"success": False, "message": "Process not completed, no final result generated."}
        
        last_output = next(iter(final_state.values()))
        final_proposal = last_output.get("final_proposal")
        
        if final_proposal:
            return {
                "success": True,
                "proposal": {
                    "title": final_proposal.Title,
                    "introduction": final_proposal.Introduction,
                    "research_problem": final_proposal.ResearchProblem,
                    "methodology": final_proposal.Methodology,
                    "expected_outcomes": final_proposal.ExpectedOutcomes,
                    "references": final_proposal.References
                }
            }
        else:
            return {"success": False, "message": "Process finished but no final research proposal was generated."}
    
    def _get_timestamp(self) -> str:
        return datetime.now().isoformat()
    
    def get_current_state(self) -> Optional[Dict]:
        return self.current_state
    
    def health_check(self) -> Dict[str, Any]:
        """Health check method to test LLM connectivity"""
        try:
            self.llm.invoke([HumanMessage(content="test")])
            return {"status": "healthy", "message": "LLM connection successful"}
        except Exception as e:
            return {"status": "unhealthy", "message": f"LLM connection failed: {str(e)}"}


def create_research_agent(model_type: str = "ollama") -> ResearchAgentApp:
    """Factory function to create a research agent with predefined model configurations"""
    model_configs = {
        "ollama": {"model_type": "ollama", "model_name": "llama3.1:8b"},
        "gemini": {"model_type": "gemini", "model_name": "gemini-2.5-flash"},
        "openai": {"model_type": "openai", "model_name": "gpt-4o"}
    }
    
    if model_type not in model_configs:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return ResearchAgentApp(**model_configs[model_type])
