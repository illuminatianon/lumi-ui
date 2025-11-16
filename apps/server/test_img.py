import asyncio
import logging
import sys
import os

# Add the server directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import initialize_config, get_config
from services.inference.factory import create_unified_inference_service
from services.inference.models import UnifiedRequest, Attachment

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)

    initialize_config()
    config = get_config()
    config.inference.enabled = True
    config.inference.openai.enabled = True
    config.inference.openai.api_key = config.api_keys.openai

    service = create_unified_inference_service()

    request = UnifiedRequest(
        prompt="Make an image of a sad clown",
        model="google/gemini-2.5-flash-image",
        temperature=0.7,
        max_tokens=100,
    )

    resp = asyncio.run(service.process_request(request))

    import pdb

    pdb.set_trace()
