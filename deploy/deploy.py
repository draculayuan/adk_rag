import os
import yaml
import subprocess
from dotenv import load_dotenv

def load_config():
    """Load environment variables and configuration."""
    load_dotenv()
    
    # Load agent engine config
    with open("deploy/agent_engine_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Replace environment variables
    def replace_env_vars(value):
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            return os.getenv(env_var, "")
        return value
    
    def process_dict(d):
        for k, v in d.items():
            if isinstance(v, dict):
                process_dict(v)
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        process_dict(item)
                    else:
                        v[i] = replace_env_vars(item)
            else:
                d[k] = replace_env_vars(v)
    
    process_dict(config)
    return config

def deploy_agent():
    """Deploy the agent to Vertex AI Agent Engine."""
    try:
        # Load configuration
        config = load_config()
        
        # Save processed config
        with open("deploy/processed_config.yaml", "w") as f:
            yaml.dump(config, f)
        
        # Deploy using gcloud
        subprocess.run([
            "gcloud",
            "ai",
            "agents",
            "deploy",
            "--config=deploy/processed_config.yaml",
            f"--project={os.getenv('GOOGLE_CLOUD_PROJECT')}",
            f"--region={os.getenv('VERTEX_AI_LOCATION')}"
        ], check=True)
        
        print("Agent deployed successfully!")
        
    except Exception as e:
        print(f"Error deploying agent: {str(e)}")
        raise

if __name__ == "__main__":
    deploy_agent() 