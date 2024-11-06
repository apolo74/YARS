
import os
import yaml

abs_path = os.path.dirname(__file__) #<-- absolute dir the script is in

class LoadConfig:
    def __init__(self) -> None:
        with open(abs_path + "/app_config.yml") as cfg:
            app_config = yaml.load(cfg, Loader=yaml.FullLoader)

        self.load_llm_configs(app_config=app_config)

    def load_llm_configs(self, app_config):
        # self.model_name = "llama3.2:1b" # os.getenv("gpt_deployment_name")
        self.template_chat = app_config["llm_config"]["template_chat"]
        self.template_query = app_config["llm_config"]["template_query"]
        self.template_answer = app_config["llm_config"]["template_answer"]
        self.template_agent = app_config["llm_config"]["template_agent_alone"]
        # self.rag_llm_system_role = app_config["llm_config"]["rag_llm_system_role"]
        self.temperature = app_config["llm_config"]["temperature"]
        self.embedding_model = "mxbai-embed-large" # os.getenv("embed_deployment_name")

