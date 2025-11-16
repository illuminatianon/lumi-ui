"""Google/Gemini provider shim implementation."""

import logging
from typing import Dict, Any, List, Optional
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

from ..base import ProviderShim
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


class GoogleShim(ProviderShim):
    """Google/Gemini provider shim implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Google shim.
        
        Args:
            config: Provider configuration including api_key
        """
        super().__init__(config)
        genai.configure(api_key=config.get('api_key'))
        self._models_cache = {}
    
    async def process_request(self, request: UnifiedRequest, model_config: ModelConfig) -> UnifiedResponse:
        """Process a unified request using Google Gemini API.
        
        Args:
            request: The unified request to process
            model_config: Configuration for the specific model to use
            
        Returns:
            Unified response containing the result
        """
        request_type = self.determine_request_type(request, model_config)
        
        if request_type == RequestType.VISION_ANALYSIS:
            return await self._handle_vision_request(request, model_config)
        elif request_type == RequestType.IMAGE_GENERATION:
            # Google doesn't have image generation in Gemini yet
            raise NotImplementedError("Google Gemini doesn't support image generation yet")
        elif request_type == RequestType.IMAGE_EDIT:
            raise NotImplementedError("Google Gemini doesn't support image editing yet")
        else:
            return await self._handle_text_request(request, model_config)
    
    async def _handle_text_request(self, request: UnifiedRequest, model_config: ModelConfig) -> UnifiedResponse:
        """Handle pure text generation requests."""
        try:
            # Get or create model instance
            model = self._get_model_instance(model_config.name)
            
            # Map unified parameters to Gemini format
            generation_config = self._build_generation_config(request, model_config)
            
            # Build prompt from system message and user prompt
            prompt = self._build_prompt(request)
            
            # Generate content
            response = await model.generate_content_async(
                prompt,
                generation_config=generation_config
            )
            
            return UnifiedResponse(
                content=response.text,
                model_used=model_config.name,
                provider="google",
                usage=TokenUsage(
                    prompt_tokens=response.usage_metadata.prompt_token_count,
                    completion_tokens=response.usage_metadata.candidates_token_count,
                    total_tokens=response.usage_metadata.total_token_count
                ),
                finish_reason=response.candidates[0].finish_reason.name,
                metadata={"safety_ratings": [rating.category.name for rating in response.candidates[0].safety_ratings]}
            )
        except Exception as e:
            logger.error(f"Google text request failed: {e}")
            raise
    
    async def _handle_vision_request(self, request: UnifiedRequest, model_config: ModelConfig) -> UnifiedResponse:
        """Handle vision requests with image attachments."""
        try:
            # Get or create model instance
            model = self._get_model_instance(model_config.name)
            
            # Map unified parameters to Gemini format
            generation_config = self._build_generation_config(request, model_config)
            
            # Build content with text and images
            content_parts = [request.prompt]
            
            for attachment in request.attachments:
                if attachment.attachment_type == AttachmentType.IMAGE:
                    # Convert to PIL Image for Gemini
                    from PIL import Image
                    from io import BytesIO
                    image = Image.open(BytesIO(attachment.content))
                    content_parts.append(image)
            
            # Generate content
            response = await model.generate_content_async(
                content_parts,
                generation_config=generation_config
            )
            
            return UnifiedResponse(
                content=response.text,
                model_used=model_config.name,
                provider="google",
                usage=TokenUsage(
                    prompt_tokens=response.usage_metadata.prompt_token_count,
                    completion_tokens=response.usage_metadata.candidates_token_count,
                    total_tokens=response.usage_metadata.total_token_count
                ),
                finish_reason=response.candidates[0].finish_reason.name,
                attachments_processed=[
                    {"type": att.attachment_type, "filename": att.filename}
                    for att in request.attachments
                ],
                metadata={"safety_ratings": [rating.category.name for rating in response.candidates[0].safety_ratings]}
            )
        except Exception as e:
            logger.error(f"Google vision request failed: {e}")
            raise
    
    def _get_model_instance(self, model_name: str):
        """Get or create a Gemini model instance."""
        if model_name not in self._models_cache:
            self._models_cache[model_name] = genai.GenerativeModel(model_name)
        return self._models_cache[model_name]
    
    def _build_generation_config(self, request: UnifiedRequest, model_config: ModelConfig) -> GenerationConfig:
        """Build Gemini generation configuration from unified request."""
        params = self.map_parameters(request.dict(), model_config)
        
        config_params = {}
        if "temperature" in params:
            config_params["temperature"] = params["temperature"]
        if "max_output_tokens" in params:
            config_params["max_output_tokens"] = params["max_output_tokens"]
        if "top_p" in params:
            config_params["top_p"] = params["top_p"]
        if "stop_sequences" in params:
            config_params["stop_sequences"] = params["stop_sequences"]
        
        return GenerationConfig(**config_params)
    
    def _build_prompt(self, request: UnifiedRequest) -> str:
        """Build prompt from system message and user prompt."""
        if request.system_message:
            return f"System: {request.system_message}\n\nUser: {request.prompt}"
        return request.prompt

    async def discover_models(self) -> Dict[str, ModelConfig]:
        """Discover available Google models."""
        try:
            models = genai.list_models()
            discovered = {}

            for model in models:
                # Only include generative models
                if 'generateContent' in model.supported_generation_methods:
                    model_id = model.name.split('/')[-1]  # Extract model name from full path
                    discovered[model_id] = self._infer_model_config(model_id, model)

            return discovered
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
