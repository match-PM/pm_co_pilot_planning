import yaml
from ament_index_python.packages import get_package_share_directory

from langchain.chat_models import init_chat_model

class LLMConfig:

    def __init__(self, llm_tool: str):
        path = get_package_share_directory("pm_co_pilot_planning")        
        prompt = path + '/Prompts.yaml'        
        with open(prompt, 'r') as file:
            config_data = yaml.safe_load(file)
            tool = config_data[llm_tool]
            self.model = config_data[llm_tool]['model']
            self.model_provider = config_data[llm_tool]['model_provider']
            self.temperature = config_data[llm_tool]['temperature']
            self.system_prompt = config_data[llm_tool]['system_prompt']

        self.llm = init_chat_model(
            self.model, 
            model_provider=self.model_provider, 
            temperature=self.temperature
        )