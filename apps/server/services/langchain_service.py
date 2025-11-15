"""LangChain service for AI integrations."""

import logging
from typing import Optional, Dict, Any, List, Union
from io import BytesIO
import base64

from langchain_openai import ChatOpenAI, OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI, Modality
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from PIL import Image

from config import get_config

logger = logging.getLogger(__name__)


class LangChainService:
    """Service for managing LangChain integrations with OpenAI and Google."""
    
    def __init__(self):
        """Initialize the LangChain service."""
        self._openai_chat = None
        self._openai_dalle = None
        self._google_chat = None
        self._google_image = None
        self._config = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize AI models based on available API keys."""
        try:
            self._config = get_config()
            
            # Initialize OpenAI models if API key is available
            if self._config.api_keys.openai:
                self._openai_chat = ChatOpenAI(
                    api_key=self._config.api_keys.openai,
                    model=self._config.langchain.openai_model,
                    temperature=self._config.langchain.default_temperature,
                    max_tokens=self._config.langchain.max_tokens
                )
                self._openai_dalle = OpenAI(
                    api_key=self._config.api_keys.openai
                )
                logger.info("OpenAI models initialized successfully")
            else:
                logger.warning("OpenAI API key not found, OpenAI models disabled")

            # Initialize Google models if API key is available
            if self._config.api_keys.gemini:
                self._google_chat = ChatGoogleGenerativeAI(
                    google_api_key=self._config.api_keys.gemini,
                    model=self._config.langchain.google_model,
                    temperature=self._config.langchain.default_temperature,
                    max_tokens=self._config.langchain.max_tokens
                )
                # Initialize Google image generation model
                self._google_image = ChatGoogleGenerativeAI(
                    google_api_key=self._config.api_keys.gemini,
                    model="gemini-2.5-flash-image",
                    temperature=self._config.langchain.default_temperature
                )
                logger.info("Google Gemini models initialized successfully")
            else:
                logger.warning("Google API key not found, Google models disabled")
                
        except Exception as e:
            logger.error(f"Failed to initialize AI models: {e}")
    
    def get_available_models(self) -> Dict[str, bool]:
        """Get status of available AI models."""
        return {
            "openai_chat": self._openai_chat is not None,
            "openai_dalle": self._openai_dalle is not None,
            "google_chat": self._google_chat is not None,
            "google_image": self._google_image is not None
        }
    
    async def generate_text(
        self, 
        prompt: str, 
        model: str = "auto",
        system_message: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """Generate text using specified AI model.
        
        Args:
            prompt: The user prompt
            model: Model to use ("openai", "google", or "auto")
            system_message: Optional system message
            temperature: Optional temperature override
            
        Returns:
            Dictionary with generated text and metadata
        """
        try:
            # Select model
            selected_model = self._select_model(model, "chat")
            if not selected_model:
                raise ValueError("No available models for text generation")
            
            # Prepare messages
            messages = []
            if system_message:
                messages.append(SystemMessage(content=system_message))
            messages.append(HumanMessage(content=prompt))
            
            # Set temperature if provided
            if temperature is not None:
                selected_model.temperature = temperature
            
            # Generate response
            response = await selected_model.ainvoke(messages)
            
            return {
                "text": response.content,
                "model_used": selected_model.__class__.__name__,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            return {
                "text": "",
                "error": str(e),
                "success": False
            }
    
    def _select_model(self, model_preference: str, task_type: str):
        """Select appropriate model based on preference and availability."""
        if model_preference == "openai" and task_type == "chat":
            return self._openai_chat
        elif model_preference == "google" and task_type == "chat":
            return self._google_chat
        elif model_preference == "openai" and task_type == "dalle":
            return self._openai_dalle
        elif model_preference == "auto":
            # Auto-select based on availability
            if task_type == "chat":
                return self._openai_chat or self._google_chat
            elif task_type == "dalle":
                return self._openai_dalle
        
        return None

    async def process_image_with_text(
        self,
        image_data: Union[str, bytes],
        prompt: str,
        model: str = "auto",
        image_format: str = "base64"
    ) -> Dict[str, Any]:
        """Process image with text prompt using vision models.

        Args:
            image_data: Image data (base64 string or bytes)
            prompt: Text prompt about the image
            model: Model to use ("openai", "google", or "auto")
            image_format: Format of image_data ("base64" or "bytes")

        Returns:
            Dictionary with analysis result and metadata
        """
        try:
            # Convert image data to appropriate format
            if image_format == "bytes" and isinstance(image_data, bytes):
                # Convert bytes to base64
                image_b64 = base64.b64encode(image_data).decode('utf-8')
            elif image_format == "base64" and isinstance(image_data, str):
                image_b64 = image_data
            else:
                raise ValueError("Invalid image data format")

            # Select model (prefer OpenAI for vision tasks)
            if model == "auto":
                selected_model = self._openai_chat or self._google_chat
            else:
                selected_model = self._select_model(model, "chat")

            if not selected_model:
                raise ValueError("No available models for image processing")

            # Create message with image
            if isinstance(selected_model, ChatOpenAI):
                # OpenAI format
                message = HumanMessage(
                    content=[
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                        }
                    ]
                )
            else:
                # Google format - convert base64 to PIL Image
                image_bytes = base64.b64decode(image_b64)
                pil_image = Image.open(BytesIO(image_bytes))

                message = HumanMessage(
                    content=[
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": pil_image}
                    ]
                )

            # Generate response
            response = await selected_model.ainvoke([message])

            return {
                "text": response.content,
                "model_used": selected_model.__class__.__name__,
                "success": True
            }

        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            return {
                "text": "",
                "error": str(e),
                "success": False
            }

    async def generate_image(
        self,
        prompt: str,
        model: str = "auto",
        size: Optional[str] = None,
        quality: Optional[str] = None,
        n: int = 1
    ) -> Dict[str, Any]:
        """Generate image using DALL-E or Google Gemini.

        Args:
            prompt: Description of the image to generate
            model: Model to use ("openai", "google", or "auto")
            size: Image size (for DALL-E: "256x256", "512x512", "1024x1024", "1792x1024", "1024x1792")
            quality: Image quality (for DALL-E: "standard" or "hd")
            n: Number of images to generate (1-10, DALL-E 3 only supports 1)

        Returns:
            Dictionary with generated image URLs and metadata
        """
        try:
            # Select model for image generation
            if model == "openai" or (model == "auto" and self._openai_dalle):
                return await self._generate_image_openai(prompt, size, quality, n)
            elif model == "google" or (model == "auto" and self._google_image):
                return await self._generate_image_google(prompt)
            else:
                raise ValueError("No available models for image generation")

        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            return {
                "images": [],
                "error": str(e),
                "success": False
            }

    async def _generate_image_openai(
        self,
        prompt: str,
        size: Optional[str] = None,
        quality: Optional[str] = None,
        n: int = 1
    ) -> Dict[str, Any]:
        """Generate image using OpenAI DALL-E."""
        if not self._openai_dalle:
            raise ValueError("OpenAI DALL-E not available")

        # Use configuration defaults if not provided
        if size is None:
            size = self._config.langchain.dalle_size
        if quality is None:
            quality = self._config.langchain.dalle_quality

        # Import OpenAI client for DALL-E (LangChain doesn't have direct DALL-E support)
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self._config.api_keys.openai)

        response = await client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=size,
            quality=quality,
            n=min(n, 1)  # DALL-E 3 only supports n=1
        )

        images = []
        for image in response.data:
            images.append({
                "url": image.url,
                "revised_prompt": getattr(image, 'revised_prompt', None)
            })

        return {
            "images": images,
            "model_used": "DALL-E-3",
            "success": True
        }

    async def _generate_image_google(self, prompt: str) -> Dict[str, Any]:
        """Generate image using Google Gemini."""
        if not self._google_image:
            raise ValueError("Google Gemini image generation not available")

        message = {
            "role": "user",
            "content": prompt,
        }

        response = await self._google_image.ainvoke(
            [message],
            response_modalities=[Modality.TEXT, Modality.IMAGE],
        )

        images = []

        # Extract images from response content
        if hasattr(response, 'content') and isinstance(response.content, list):
            for block in response.content:
                if isinstance(block, dict) and block.get("image_url"):
                    images.append({
                        "url": block["image_url"].get("url"),
                        "revised_prompt": None  # Google doesn't provide revised prompts
                    })

        return {
            "images": images,
            "model_used": "Gemini-2.5-Flash-Image",
            "success": True
        }

    def refresh_models(self):
        """Refresh model initialization (useful when API keys are updated)."""
        self._initialize_models()
