from PIL import Image
import base64
import io

from langchain_core.messages import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash-image")

message = {
    "role": "user",
    "content": "Generate a photorealistic image of a cuddly cat wearing a hat.",
}

response = llm.invoke(
    [message],
    # response_modalities=["text", "image"],  # Removed due to version compatibility issues
)


def _get_image_base64(response: AIMessage) -> str:
    image_block = next(
        block
        for block in response.content
        if isinstance(block, dict) and block.get("image_url")
    )
    return image_block["image_url"].get("url").split(",")[-1]


# save file to disk using PIL
image_b64 = _get_image_base64(response)
image_data = base64.b64decode(image_b64)
image = Image.open(io.BytesIO(image_data))
image.save("test.png")
