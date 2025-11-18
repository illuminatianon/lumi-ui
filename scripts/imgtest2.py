import asyncio
import logging
import sys
import os
import base64
from pathlib import Path
from datetime import datetime

# Add the server directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "apps" / "server"))

from config import initialize_config, get_config
from services.inference.factory import create_unified_inference_service
from services.inference.models import UnifiedRequest, Attachment

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def load_test_image(image_name: str) -> Attachment:
    """Load a test image as an Attachment object."""
    script_dir = Path(__file__).parent
    image_path = script_dir / image_name

    if not image_path.exists():
        raise FileNotFoundError(f"Test image not found: {image_path}")

    return Attachment.from_file(str(image_path))


def save_image_from_base64(base64_data: str, filename: str, output_dir: Path) -> str:
    """Save a base64 encoded image to a file."""
    try:
        # Remove data URL prefix if present
        if base64_data.startswith("data:image/"):
            base64_data = base64_data.split(",", 1)[1]

        # Decode base64 data
        image_data = base64.b64decode(base64_data)

        # Create output directory if it doesn't exist
        output_dir.mkdir(exist_ok=True)

        # Save image
        output_path = output_dir / filename
        with open(output_path, "wb") as f:
            f.write(image_data)

        return str(output_path)
    except Exception as e:
        logger.error(f"Failed to save image {filename}: {e}")
        return None

async def main():
    if not initialize_config():
        print("❌ Failed to initialize configuration")
        return

    config = get_config()
    # Create unified inference service
    service = create_unified_inference_service()

    if not service:
        print("❌ Failed to create unified inference service")
        print("   Make sure you have API keys configured:")
        print("   - Set OPENAI_API_KEY environment variable")
        print("   - Set GOOGLE_API_KEY environment variable")
        return

    available_models = service.get_available_models()
    output_dir = Path("generated_images")

    # Run tests - three simple tests
    try:
        attachments = [
            load_test_image("dam.png"),
            load_test_image("ref_test.png"),
        ]
        messages = [
            {"role":"system", "content": "Image 1 TARGET, Image 2 REFERENCE [BUNCHAN].\n"},
            {"role":"user", "content": "BUNCHAN has CHUCKED more WOOD than a WOODCHUCK COULD. the WOODCHUCK is sitting around, looking dazed and confuzed."}
        ]
        request = UnifiedRequest(
            messages=messages,
            model="google/gemini-2.5-flash-image",
            response_format="image",
            attachments=attachments,
            extras={
                "aspect_ratio": "16:9",
                "response_modalities": ["Text", "Image"],
            },
        )

        response = await service.process_request(request)

        if response and response.images:    
            # Save the image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"img_test_{timestamp}.png"
            saved_path = save_image_from_base64(
                response.images[0], filename, output_dir
            )

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
