import os


os.environ["USE_MOCK"] = "true"
os.environ["ROUTER_MODE"] = "rules"
os.environ["LLM_PROVIDER"] = "openai"
os.environ["AGENT_ENABLED"] = "true"
os.environ["LLM_ROUTER_RUNTIME_CONFIG_PATH"] = "__pytest_missing_runtime_config.json"
