"""Google/Gemini provider implementation using pure REST API calls."""

import logging
import json
import base64
from typing import Dict, Any, List, Optional
import httpx

from ..base import Provider
from ..models import (
    UnifiedRequest,
    UnifiedResponse,
    ModelConfig,
    Attachment,
    AttachmentType,
    TokenUsage,
    RequestType,
    ModelCapabilities,
    ParameterMapping
)

logger = logging.getLogger(__name__)


class GoogleProvider(Provider):
    """Google/Gemini provider using pure REST API calls."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize Google provider.

        Args:
            config: Provider configuration including api_key
        """
        super().__init__(config)
        self.api_key = config.get('api_key')
        self.base_url = 'https://generativelanguage.googleapis.com'
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=60.0
        )

    @classmethod
    def get_provider_name(cls) -> str:
        """Return the provider name for registry."""
        return "google"

    @classmethod
    def get_supported_models(cls) -> Dict[str, ModelConfig]:
        """Return all models supported by Google."""
        return {
            "gemini-1.5-pro": ModelConfig(
                name="gemini-1.5-pro",
                display_name="Gemini 1.5 Pro",
                provider="google",
                capabilities=ModelCapabilities(
                    text_generation=True,
                    vision=True,
                    function_calling=True,
                    streaming=True,
                    max_context_length=2000000,
                    supports_multimodal=True
                ),
                parameter_mapping=ParameterMapping(
                    max_tokens_param="max_output_tokens",
                    temperature_param="temperature",
                    top_p_param="top_p"
                ),
                supported_parameters={
                    "max_output_tokens", "temperature", "top_p", "stop_sequences"
                },
                cost_per_1k_tokens=0.0035,
                context_window=2000000
            ),
            "gemini-1.5-flash": ModelConfig(
                name="gemini-1.5-flash",
                display_name="Gemini 1.5 Flash",
                provider="google",
                capabilities=ModelCapabilities(
                    text_generation=True,
                    vision=True,
                    function_calling=True,
                    streaming=True,
                    max_context_length=1000000,
                    supports_multimodal=True
                ),
                parameter_mapping=ParameterMapping(
                    max_tokens_param="max_output_tokens",
                    temperature_param="temperature",
                    top_p_param="top_p"
                ),
                supported_parameters={
                    "max_output_tokens", "temperature", "top_p", "stop_sequences"
                },
                cost_per_1k_tokens=0.00075,
                context_window=1000000
            ),
            "gemini-2.5-flash": ModelConfig(
                name="gemini-2.5-flash",
                display_name="Gemini 2.5 Flash",
                provider="google",
                capabilities=ModelCapabilities(
                    text_generation=True,
                    vision=True,
                    function_calling=True,
                    streaming=True,
                    max_context_length=1000000,
                    supports_multimodal=True
                ),
                parameter_mapping=ParameterMapping(
                    max_tokens_param="max_output_tokens",
                    temperature_param="temperature",
                    top_p_param="top_p"
                ),
                supported_parameters={
                    "max_output_tokens", "temperature", "top_p", "stop_sequences"
                },
                cost_per_1k_tokens=0.00075,
                context_window=1000000
            ),
            "gemini-2.5-flash-image": ModelConfig(
                name="gemini-2.5-flash-image",
                display_name="Gemini 2.5 Flash Image",
                provider="google",
                capabilities=ModelCapabilities(
                    text_generation=True,
                    image_editing=True,  # This is the key capability for reference images
                    vision=True,  # Also needed for processing reference images
                    image_generation=True,
                    function_calling=False,
                    streaming=False,
                    max_context_length=1000000,
                ),
                parameter_mapping=ParameterMapping(
                    max_tokens_param="max_output_tokens",
                    temperature_param="temperature",
                    top_p_param="top_p",
                ),
                supported_parameters={
                    "max_output_tokens",
                    "temperature",
                    "top_p",
                    "stop_sequences",
                    "aspect_ratio",
                    "response_modalities"
                },
                cost_per_1k_tokens=0.00075,
                context_window=1000000,
            )
        }

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.client.aclose()

    def is_available(self) -> bool:
        """Check if the provider is available."""
        return bool(self.api_key)

    async def process_request(self, model_name: str, request: UnifiedRequest) -> UnifiedResponse:
        """Process a unified request for a specific model.

        Args:
            model_name: Name of the model to use
            request: The unified request to process

        Returns:
            Unified response containing the result
        """
        # Get model config
        supported_models = self.get_supported_models()
        if model_name not in supported_models:
            available = list(supported_models.keys())
            raise ValueError(f"Model '{model_name}' not supported. Available: {available}")

        model_config = supported_models[model_name]
        request_type = self.determine_request_type(request, model_config)

        if request_type == RequestType.VISION_ANALYSIS:
            return await self._handle_vision_request(request, model_config)
        elif request_type == RequestType.IMAGE_GENERATION:
            return await self._handle_image_generation(request, model_config)
        elif request_type == RequestType.IMAGE_EDIT:
            return await self._handle_image_generation(request, model_config)  # Gemini handles editing via generation
        else:
            return await self._handle_text_request(request, model_config)
    
    async def _handle_text_request(self, request: UnifiedRequest, model_config: ModelConfig) -> UnifiedResponse:
        """Handle pure text generation requests via REST API."""
        # Build contents for Gemini API
        contents = self._build_contents(request)

        # Map parameters
        generation_config = self._build_generation_config(request, model_config)

        payload = {
            "contents": contents,
            "generationConfig": generation_config
        }

        try:
            url = f'/v1beta/models/{model_config.name}:generateContent'
            params = {'key': self.api_key}

            response = await self.client.post(url, json=payload, params=params)
            response.raise_for_status()
            data = response.json()

            # Extract response content with error handling
            content = ""
            try:
                candidate = data['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    content = candidate['content']['parts'][0]['text']
                elif 'content' in candidate and 'text' in candidate['content']:
                    # Alternative structure
                    content = candidate['content']['text']
                else:
                    # Check if content is empty due to safety filters or other reasons
                    finish_reason = candidate.get('finishReason', 'UNKNOWN')
                    if finish_reason == 'MAX_TOKENS':
                        content = "[Response truncated due to max tokens limit]"
                    elif finish_reason in ['SAFETY', 'RECITATION']:
                        content = f"[Response blocked due to {finish_reason} filters]"
                    else:
                        logger.error(f"No content found in Google response. Structure: {data}")
                        raise ValueError(f"No content in Google API response. Finish reason: {finish_reason}")
            except (KeyError, IndexError) as e:
                logger.error(f"Failed to extract content from Google response: {e}")
                logger.error(f"Response structure: {data}")
                raise ValueError(f"Invalid response structure from Google API: {e}")

            # Extract usage metadata if available
            usage_metadata = data.get('usageMetadata', {})

            return UnifiedResponse(
                content=content,
                model_used=model_config.name,
                provider="google",
                usage=TokenUsage(
                    prompt_tokens=usage_metadata.get('promptTokenCount', 0),
                    completion_tokens=usage_metadata.get('candidatesTokenCount', 0),
                    total_tokens=usage_metadata.get('totalTokenCount', 0)
                ),
                finish_reason=data['candidates'][0].get('finishReason', 'STOP'),
                metadata={
                    "safety_ratings": data['candidates'][0].get('safetyRatings', [])
                }
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"Google API error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Google text request failed: {e}")
            raise
    
    async def _handle_vision_request(self, request: UnifiedRequest, model_config: ModelConfig) -> UnifiedResponse:
        """Handle vision requests with image attachments via REST API."""
        # Build content parts for Gemini API
        if request.is_multi_turn():
            # For multi-turn, we need to build contents with messages and add images to the last user message
            contents = self._build_contents_with_attachments(request)
            payload = {
                "contents": contents,
                "generationConfig": self._build_generation_config(request, model_config)
            }
        else:
            # Single-turn format (backward compatibility)
            parts = [{"text": request.prompt}]

            # Add images as inline data
            for attachment in request.attachments:
                if attachment.attachment_type == AttachmentType.IMAGE:
                    parts.append({
                        "inline_data": {
                            "mime_type": attachment.mime_type,
                            "data": attachment.to_base64()
                        }
                    })

            payload = {
                "contents": [{"parts": parts}],
                "generationConfig": self._build_generation_config(request, model_config)
            }

        try:
            url = f'/v1beta/models/{model_config.name}:generateContent'
            params = {'key': self.api_key}

            response = await self.client.post(url, json=payload, params=params)
            response.raise_for_status()
            data = response.json()

            # Extract response content
            content = data['candidates'][0]['content']['parts'][0]['text']

            # Extract usage metadata if available
            usage_metadata = data.get('usageMetadata', {})

            return UnifiedResponse(
                content=content,
                model_used=model_config.name,
                provider="google",
                usage=TokenUsage(
                    prompt_tokens=usage_metadata.get('promptTokenCount', 0),
                    completion_tokens=usage_metadata.get('candidatesTokenCount', 0),
                    total_tokens=usage_metadata.get('totalTokenCount', 0)
                ),
                finish_reason=data['candidates'][0].get('finishReason', 'STOP'),
                attachments_processed=[
                    {"type": att.attachment_type, "filename": att.filename}
                    for att in request.attachments
                ],
                metadata={
                    "safety_ratings": data['candidates'][0].get('safetyRatings', [])
                }
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"Google API error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Google vision request failed: {e}")
            raise

    async def _handle_image_generation(self, request: UnifiedRequest, model_config: ModelConfig) -> UnifiedResponse:
        """Handle image generation requests for Gemini 2.5 Flash Image via REST API."""
        # For Gemini 2.5 Flash Image, we need to use the generateContent endpoint
        # but with specific image generation parameters

        # Build contents with support for both single-turn and multi-turn formats
        if request.is_multi_turn():
            contents = self._build_contents_with_attachments(request)
        else:
            # Single-turn format (backward compatibility)
            parts = [{"text": request.prompt}]

            # Add reference images if provided (for editing/style transfer)
            for attachment in request.attachments:
                if attachment.attachment_type == AttachmentType.IMAGE:
                    parts.append({
                        "inline_data": {
                            "mime_type": attachment.mime_type,
                            "data": attachment.to_base64()
                        }
                    })

            contents = [{"parts": parts}]

        # Build generation config with image-specific parameters
        generation_config = {}

        # Handle aspect ratio configuration
        aspect_ratio = request.extras.get("aspect_ratio", "1:1")
        if aspect_ratio:
            generation_config["imageConfig"] = {
                "aspectRatio": aspect_ratio
            }

        # Handle response modalities (image only or text+image)
        response_modalities = request.extras.get("response_modalities", ["Text", "Image"])
        if response_modalities:
            generation_config["responseModalities"] = response_modalities

        payload = {
            "contents": contents,
            "generationConfig": generation_config
        }

        try:
            url = f'/v1beta/models/{model_config.name}:generateContent'
            params = {'key': self.api_key}

            response = await self.client.post(url, json=payload, params=params)
            response.raise_for_status()
            data = response.json()

            # Extract response content and images
            candidate = data['candidates'][0]
            content_parts = candidate['content']['parts']

            text_content = ""
            images = []

            # Debug: Log the response structure
            logger.debug(f"Gemini response structure: {json.dumps(data, indent=2)}")

            for part in content_parts:
                if 'text' in part:
                    text_content += part['text']
                elif 'inlineData' in part:
                    # Image data is base64 encoded (note: camelCase in Gemini API)
                    image_data = part['inlineData']['data']
                    mime_type = part['inlineData']['mimeType']
                    images.append(f"data:{mime_type};base64,{image_data}")
                elif 'inline_data' in part:
                    # Fallback for snake_case (just in case)
                    image_data = part['inline_data']['data']
                    mime_type = part['inline_data']['mime_type']
                    images.append(f"data:{mime_type};base64,{image_data}")

            # If no images found in parts, check if the response is malformed
            if not images and not text_content:
                logger.warning(f"No content found in Gemini response. Raw response: {data}")
                # Sometimes the image might be in a different location
                # Let's check the entire response structure
                logger.warning(f"Full response structure: {json.dumps(data, indent=2)}")

            # Extract usage metadata if available
            usage_metadata = data.get('usageMetadata', {})

            return UnifiedResponse(
                content=text_content if text_content else None,
                images=images,
                model_used=model_config.name,
                provider="google",
                usage=TokenUsage(
                    prompt_tokens=usage_metadata.get('promptTokenCount', 0),
                    completion_tokens=usage_metadata.get('candidatesTokenCount', 0),
                    total_tokens=usage_metadata.get('totalTokenCount', 0)
                ),
                finish_reason=candidate.get('finishReason', 'STOP'),
                attachments_processed=[
                    {"type": att.attachment_type, "filename": att.filename}
                    for att in request.attachments
                ],
                metadata={
                    "safety_ratings": candidate.get('safetyRatings', []),
                    "aspect_ratio": aspect_ratio,
                    "response_modalities": response_modalities
                }
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"Google API error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Google image generation request failed: {e}")
            raise

    def _build_generation_config(self, request: UnifiedRequest, model_config: ModelConfig) -> Dict[str, Any]:
        """Build Gemini generation configuration from unified request."""
        params = self.map_parameters(request.model_dump(), model_config)

        config_params = {}
        if "temperature" in params and params["temperature"] is not None:
            config_params["temperature"] = params["temperature"]
        if "max_output_tokens" in params and params["max_output_tokens"] is not None:
            config_params["maxOutputTokens"] = params["max_output_tokens"]
        if "top_p" in params and params["top_p"] is not None:
            config_params["topP"] = params["top_p"]
        if "stop_sequences" in params and params["stop_sequences"] is not None:
            config_params["stopSequences"] = params["stop_sequences"]

        return config_params

    def _build_prompt(self, request: UnifiedRequest) -> str:
        """Build prompt from system message and user prompt (single-turn only)."""
        if request.system_message:
            return f"System: {request.system_message}\n\nUser: {request.prompt}"
        return request.prompt

    def _build_contents(self, request: UnifiedRequest) -> List[Dict[str, Any]]:
        """Build contents array for Gemini API from either single-turn or multi-turn format."""
        if request.is_multi_turn():
            # Convert messages to Gemini format
            contents = []
            for msg in request.messages:
                # Map roles: system -> user (Gemini doesn't have system role)
                role = "user" if msg.role in ["system", "user"] else "model"
                contents.append({
                    "role": role,
                    "parts": [{"text": msg.content}]
                })
            return contents
        else:
            # Single-turn format (backward compatibility)
            content = self._build_prompt(request)
            return [{"parts": [{"text": content}]}]

    def _build_contents_with_attachments(self, request: UnifiedRequest) -> List[Dict[str, Any]]:
        """Build contents array with attachment support for vision/image generation requests."""
        if request.is_multi_turn():
            # Convert messages to Gemini format
            contents = []
            for msg in request.messages:
                # Map roles: system -> user (Gemini doesn't have system role)
                role = "user" if msg.role in ["system", "user"] else "model"
                contents.append({
                    "role": role,
                    "parts": [{"text": msg.content}]
                })

            # Add attachments to the last user message
            if request.attachments:
                # Find the last user message
                for i in range(len(contents) - 1, -1, -1):
                    if contents[i]["role"] == "user":
                        # Add images to this message
                        for attachment in request.attachments:
                            if attachment.attachment_type == AttachmentType.IMAGE:
                                contents[i]["parts"].append({
                                    "inline_data": {
                                        "mime_type": attachment.mime_type,
                                        "data": attachment.to_base64()
                                    }
                                })
                        break

            return contents
        else:
            # Single-turn format - handled by caller
            content = self._build_prompt(request)
            parts = [{"text": content}]

            # Add images
            for attachment in request.attachments:
                if attachment.attachment_type == AttachmentType.IMAGE:
                    parts.append({
                        "inline_data": {
                            "mime_type": attachment.mime_type,
                            "data": attachment.to_base64()
                        }
                    })

            return [{"parts": parts}]

    async def discover_models(self) -> Dict[str, ModelConfig]:
        """Discover available Google models via REST API."""
        try:
            url = '/v1beta/models'
            params = {'key': self.api_key}

            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            discovered = {}
            for model in data.get('models', []):
                # Only include generative models
                if 'generateContent' in model.get('supportedGenerationMethods', []):
                    model_id = model['name'].split('/')[-1]  # Extract model name from full path
                    discovered[model_id] = self._infer_model_config(model_id, model)

            return discovered
        except httpx.HTTPStatusError as e:
            logger.error(f"Google API error: {e.response.status_code} - {e.response.text}")
            return {}
        except Exception as e:
            logger.error(f"Failed to discover Google models: {e}")
            return {}

    def map_parameters(self, unified_params: Dict[str, Any], model_config: ModelConfig) -> Dict[str, Any]:
        """Map unified parameters to Google-specific parameters."""
        mapped = {}
        mapping = model_config.parameter_mapping

        # Map standard parameters
        if unified_params.get("max_tokens") and mapping.max_tokens_param:
            mapped[mapping.max_tokens_param] = unified_params["max_tokens"]

        if unified_params.get("temperature") and mapping.temperature_param:
            mapped[mapping.temperature_param] = unified_params["temperature"]

        if unified_params.get("top_p") and mapping.top_p_param:
            mapped[mapping.top_p_param] = unified_params["top_p"]

        # Add stop sequences if supported
        if "stop_sequences" in model_config.supported_parameters:
            mapped["stop_sequences"] = unified_params.get("stop_sequences")

        # Log warning for unsupported parameters
        if unified_params.get("frequency_penalty") or unified_params.get("presence_penalty"):
            logger.warning("Google Gemini doesn't support frequency/presence penalty")

        if unified_params.get("reasoning_effort"):
            logger.warning("Google Gemini doesn't support reasoning effort parameter")

        return mapped

    # Legacy compatibility methods
    async def process_request_legacy(self, request: UnifiedRequest, model_config: ModelConfig) -> UnifiedResponse:
        """Legacy method for backward compatibility."""
        return await self.process_request(model_config.name, request)

    def prepare_attachments(self, attachments: List[Attachment], model_config: ModelConfig) -> Any:
        """Convert attachments to Google-specific format."""
        # Google handles attachments as PIL Images in content, so return as-is
        return attachments

    def _infer_model_config(self, model_id: str, model_info=None) -> ModelConfig:
        """Infer model configuration based on model ID and info."""
        # Default configuration
        capabilities = ModelCapabilities(text_generation=True)
        parameter_mapping = ParameterMapping(
            max_tokens_param="max_output_tokens",
            temperature_param="temperature",
            top_p_param="top_p"
        )
        supported_parameters = {"max_output_tokens", "temperature", "top_p", "stop_sequences"}

        # Adjust based on model name patterns
        if "vision" in model_id.lower() or "pro" in model_id.lower():
            capabilities.vision = True
            capabilities.function_calling = True

        context_window = 32768  # Conservative default
        if "1.5" in model_id:
            if "pro" in model_id.lower():
                context_window = 2000000
            else:
                context_window = 1000000

        return ModelConfig(
            name=model_id,
            display_name=model_id,
            provider="google",
            capabilities=capabilities,
            parameter_mapping=parameter_mapping,
            supported_parameters=supported_parameters,
            context_window=context_window
        )
