import getpass
import os
import json
import pickle
from datetime import datetime

from rclpy.node import Node

from langchain_core.tools import BaseTool
from langchain_core.tools import tool

from langchain_core.messages import HumanMessage, AIMessage, FunctionMessage
from langgraph.prebuilt import create_react_agent

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate

from langchain.chat_models import init_chat_model
from langchain_community.tools.tavily_search import TavilySearchResults

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

from collections import defaultdict

from pm_co_pilot_planning.submodules.langchain.tools.Tools import Tools
from pm_co_pilot_planning.submodules.langchain.LLMConfig import LLMConfig



class Agent:
    """
    The Agent class is responsible for managing the interaction with the LLM.
    It handles the conversation history and the tools available to the LLM.
    """

    def __init__(self, service_node: Node, thread_id: str, rsap_instance=None):

        # Use provided RSAP instance or create Tools with service_node
        if rsap_instance:
            tools_instance = Tools(service_node, rsap_instance=rsap_instance)
            self.rsap_instance = rsap_instance
        else:
            tools_instance = Tools(service_node)
            self.rsap_instance = None
        
        # Initialize interaction log
        self.interaction_log = []
        
        llm_config = LLMConfig('agent')
        
        # Store model info for logging
        self.model_name = llm_config.model
        self.model_provider = llm_config.model_provider
        
        # Comprehensive list of tools for RSAP control - ordered by efficiency
        self.tools = [
            # Efficient query tools (use these first for simple queries!)
            tools_instance.get_action_at_index_tool,        # For "what's at index X?"
            tools_instance.get_sequence_summary_tool,       # For "show me the sequence"
            tools_instance.get_action_parameters_tool,      # For "what are the current parameters?"
            
            # Service/Action discovery
            tools_instance.get_available_services_tool,
            # tools_instance.get_available_ros_actions_tool,
            tools_instance.get_service_parameters_tool,
            tools_instance.get_parameter_value_recommendations_tool,
            
            # Building sequence
            tools_instance.add_service_to_sequence_tool,
            # tools_instance.add_ros_action_to_sequence_tool,
            tools_instance.add_user_interaction_tool,
            
            # Modifying sequence
            tools_instance.set_action_parameters_tool,
            tools_instance.delete_action_tool,
            tools_instance.move_action_tool,
            
            # Execution
            tools_instance.execute_sequence_tool,
            tools_instance.execute_single_action_tool,
            
            # Sequence persistence
            tools_instance.save_sequence_tool,
            tools_instance.load_sequence_tool,
            tools_instance.clear_sequence_tool,
            
            # Heavy tool (use sparingly - only when full details needed!)
            tools_instance.get_action_list_tool,
        ]
        
        # Bind tools to the model with parallel_tool_calls enabled
        # This allows models to emit multiple tool calls in a single response
        # The threading lock in Tools.py ensures correct sequential execution when needed
        self.model = llm_config.llm.bind_tools(self.tools, parallel_tool_calls=True)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", llm_config.system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        self.memory = MemorySaver()
        self.config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 100  
        }
        self.service_node = service_node

        # When  app starts, try to load any previously saved memory state
        # self.load_memory()

        self.service_node.get_logger().info(f"Agent initialized with model: {llm_config.model}")


    def create_executor(self):
        return create_react_agent(
            self.model,
            self.tools,
            checkpointer=self.memory,
            prompt=self.prompt
        )

    def handle_user_input(self, user_message: str) -> str:
        """
        Non-streaming approach (returns final string) with debug logging and token tracking.
        """
        agent_executor = self.create_executor()
        
        self.service_node.get_logger().info(f"Starting agent execution for: {user_message[:100]}...")
        
        # Track token usage, step details, and timing
        interaction_start = datetime.now()
        total_input_tokens = 0
        total_output_tokens = 0
        step_details = []  # Store detailed step information
        
        try:
            # Stream to see each step
            step_count = 0
            for step in agent_executor.stream(
                {"messages": [HumanMessage(content=user_message)]},
                self.config,
                stream_mode="values"
            ):
                step_count += 1
                last_message = step["messages"][-1]
                
                # Create step log entry
                step_log = {
                    "step": step_count,
                    "type": type(last_message).__name__,
                    "content": str(last_message.content)[:500] if hasattr(last_message, 'content') else None
                }
                
                # Extract token usage if available
                if hasattr(last_message, 'usage_metadata') and last_message.usage_metadata:
                    usage = last_message.usage_metadata
                    if isinstance(usage, dict):
                        total_input_tokens += usage.get('input_tokens', 0)
                        total_output_tokens += usage.get('output_tokens', 0)
                    else:
                        total_input_tokens += getattr(usage, 'input_tokens', 0)
                        total_output_tokens += getattr(usage, 'output_tokens', 0)
                
                # Log each step for debugging
                if hasattr(last_message, 'content'):
                    self.service_node.get_logger().info(
                        f"Step {step_count}: {type(last_message).__name__} - {str(last_message.content)[:200]}"
                    )
                
                # Capture tool calls (when agent calls tools)
                if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                    tool_calls_log = []
                    for tool_call in last_message.tool_calls:
                        tool_call_info = {
                            "name": tool_call.get('name', 'unknown'),
                            "args": tool_call.get('args', {})
                        }
                        tool_calls_log.append(tool_call_info)
                        self.service_node.get_logger().info(
                            f"  → Tool call: {tool_call_info['name']} with args: {str(tool_call_info['args'])[:200]}"
                        )
                    step_log["tool_calls"] = tool_calls_log
                
                # Capture tool responses (may be multiple in parallel calls)
                # Check if there are multiple new ToolMessages since last step
                from langchain_core.messages import ToolMessage
                if isinstance(last_message, ToolMessage):
                    # Count how many new messages are ToolMessages in this step
                    new_messages = step["messages"][len(step["messages"]) - 1:]
                    tool_responses = []
                    
                    # Go backwards to find all ToolMessages added in this step
                    for msg in reversed(step["messages"]):
                        if isinstance(msg, ToolMessage):
                            tool_responses.insert(0, {
                                "tool_call_id": getattr(msg, 'tool_call_id', 'unknown'),
                                "content": str(msg.content)[:500]
                            })
                        elif not isinstance(msg, ToolMessage):
                            # Stop when we hit a non-ToolMessage
                            break
                    
                    if len(tool_responses) > 1:
                        step_log["tool_responses"] = tool_responses
                        self.service_node.get_logger().info(
                            f"  → Received {len(tool_responses)} tool responses"
                        )
                        for i, resp in enumerate(tool_responses, 1):
                            self.service_node.get_logger().info(
                                f"     {i}. {resp['content'][:100]}"
                            )
                
                step_details.append(step_log)
                
                # Check if we're done (AIMessage with no tool calls)
                if isinstance(last_message, AIMessage) and not getattr(last_message, 'tool_calls', None):
                    interaction_end = datetime.now()
                    execution_time = (interaction_end - interaction_start).total_seconds()
                    
                    self.service_node.get_logger().info(f"Agent completed after {step_count} steps")
                    self.service_node.get_logger().info(
                        f"Token usage - Input: {total_input_tokens}, Output: {total_output_tokens}, Total: {total_input_tokens + total_output_tokens}"
                    )
                    self.service_node.get_logger().info(f"Execution time: {execution_time:.2f} seconds")
                    
                    # Log this interaction with detailed steps and timing
                    self.interaction_log.append({
                        "timestamp": interaction_start.isoformat(),
                        "timestamp_end": interaction_end.isoformat(),
                        "execution_time_seconds": execution_time,
                        "user_message": user_message,
                        "agent_response": last_message.content,
                        "steps": step_count,
                        "step_details": step_details,
                        "tokens": {
                            "input": total_input_tokens,
                            "output": total_output_tokens,
                            "total": total_input_tokens + total_output_tokens
                        }
                    })
                    
                    return last_message.content
            
            # Fallback: get the last message
            messages = agent_executor.invoke({"messages": [HumanMessage(content=user_message)]}, self.config)
            ai_message = messages["messages"][-1]
            response = ai_message.content
            self.service_node.get_logger().info(f"Agent messages: {messages}")
            self.service_node.get_logger().info(
                f"Token usage - Input: {total_input_tokens}, Output: {total_output_tokens}, Total: {total_input_tokens + total_output_tokens}"
            )
            return response
            
        except Exception as e:
            self.service_node.get_logger().error(f"Agent execution error after {step_count} steps: {e}")
            if total_input_tokens > 0 or total_output_tokens > 0:
                self.service_node.get_logger().info(
                    f"Token usage before error - Input: {total_input_tokens}, Output: {total_output_tokens}, Total: {total_input_tokens + total_output_tokens}"
                )
            raise

    
    def save_memory(self):
        """
        Save the current memory state to a file.
        """
        data_to_save = self.convert_defaultdict_to_dict(self.memory.storage)
        with open("chat_history.pkl", "wb") as f:
            self.service_node.get_logger().info(f"Saving memory state: {data_to_save}")
            pickle.dump(data_to_save, f)
            self.service_node.get_logger().info("Memory state saved.")

    def load_memory(self):
        """
        Load the memory state from a file.
        """
        try:
            with open("chat_history.pkl", "rb") as f:
                loaded_data = pickle.load(f)
                # now loaded_data is just a nested dict
                self.memory.storage = loaded_data
                self.service_node.get_logger().info("Memory state loaded.")
        except FileNotFoundError:
            self.service_node.get_logger().info("No saved memory state found.")
            pass

    

    # a helper function to recursively convert defaultdict -> dict
    def convert_defaultdict_to_dict(self, obj):
        if isinstance(obj, defaultdict):
            # convert its contents recursively into a normal dict
            obj = {k: self.convert_defaultdict_to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, dict):
            # just handle nested dict
            obj = {k: self.convert_defaultdict_to_dict(v) for k, v in obj.items()}
        return obj

    def save_interaction_log(self, task_success=None, comment=None):
        """Save the interaction log to a JSON file with timestamp.
        
        Args:
            task_success: Optional bool indicating if the user confirmed task was successful (True/False/None)
            comment: Optional string with user's comment about the task execution
        """
        if not self.interaction_log:
            self.service_node.get_logger().info("No interactions to save")
            return
        
        # Get folder path from RSAP instance
        if self.rsap_instance and hasattr(self.rsap_instance, 'rsap_file_manager'):
            folder_path = self.rsap_instance.rsap_file_manager.get_folder_path()
        else:
            folder_path = "/home/match-pm/Desktop"  # Default fallback
        
        # Create filename with timestamp and model name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sanitize model name for filename (replace / and : with _)
        safe_model_name = self.model_name.replace('/', '_').replace(':', '_').replace('-', '_')
        filename = f"copilot_log_{timestamp}_{safe_model_name}.json"
        filepath = os.path.join(folder_path, filename)
        
        # Calculate totals
        total_interactions = len(self.interaction_log)
        total_steps = sum(log["steps"] for log in self.interaction_log)
        total_tokens = sum(log["tokens"]["total"] for log in self.interaction_log)
        total_input_tokens = sum(log["tokens"]["input"] for log in self.interaction_log)
        total_output_tokens = sum(log["tokens"]["output"] for log in self.interaction_log)
        
        # Capture current sequence state
        final_sequence = []
        if self.rsap_instance:
            try:
                for idx, action in enumerate(self.rsap_instance.action_list):
                    action_info = {
                        "index": idx + 1,  # 1-based for consistency with GUI
                        "name": action.get_name() if hasattr(action, 'get_name') else str(action),
                        "type": type(action).__name__,
                        "is_active": action.is_active() if hasattr(action, 'is_active') else True
                    }
                    # Add client info if available
                    if hasattr(action, 'client'):
                        action_info["client"] = action.client
                    # Add parameters if available
                    if hasattr(action, 'get_request_as_ordered_dict'):
                        action_info["parameters"] = dict(action.get_request_as_ordered_dict())
                    final_sequence.append(action_info)
            except Exception as e:
                self.service_node.get_logger().warning(f"Could not capture final sequence: {e}")
        
        # Create log data structure
        log_data = {
            "model": self.model_name,
            "task_success": task_success,
            "comment": comment,
            "session_start": self.interaction_log[0]["timestamp"] if self.interaction_log else None,
            "session_end": datetime.now().isoformat(),
            "summary": {
                "total_interactions": total_interactions,
                "total_steps": total_steps,
                "total_tokens": total_tokens,
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens
            },
            "interactions": self.interaction_log,
            "final_sequence": {
                "total_actions": len(final_sequence),
                "actions": final_sequence
            }
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(log_data, f, indent=2)
            self.service_node.get_logger().info(f"Interaction log saved to: {filepath}")
            self.service_node.get_logger().info(
                f"Session summary: {total_interactions} interactions, {total_steps} steps, {total_tokens} tokens"
            )
        except Exception as e:
            self.service_node.get_logger().error(f"Failed to save interaction log: {e}")





