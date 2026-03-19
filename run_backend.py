"""Start the FastAPI backend server."""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
import yaml
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

if __name__ == "__main__":
    import uvicorn

    config_path = Path(__file__).parent / "configs" / "config.yaml"
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)

    api_config = config.get("api", {})
    uvicorn.run(
        "src.api.server:app",
        host=api_config.get("host", "0.0.0.0"),
        port=api_config.get("port", 8000),
        reload=False,  # No fork — avoids segfault with faiss/torch on Mac
    )
