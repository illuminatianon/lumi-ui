"""Image generation parameter validation utilities."""

from typing import Dict, List, Optional, Any, Tuple
from pydantic import ValidationError


class ImageGenerationValidator:
    """Validator for image generation parameters across different models."""
    
    # Model-specific configurations
    MODEL_CONFIGS = {
        "gemini-2.5-flash-image": {
            "supported_aspect_ratios": [
                "1:1", "2:3", "3:2", "3:4", "4:3", 
                "4:5", "5:4", "9:16", "16:9", "21:9"
            ],
            "supported_sizes": [
                "1024x1024", "832x1248", "1248x832", "864x1184", 
                "1184x864", "896x1152", "1152x896", "768x1344", 
                "1344x768", "1536x672"
            ],
            "supported_quality_levels": ["standard"],
            "supported_styles": [],
            "default_aspect_ratio": "1:1",
            "default_quality": "standard",
            "supports_reference_images": True,
            "supports_response_modalities": True
        },
        "gpt-image-1": {
            "supported_aspect_ratios": ["1:1", "3:2", "2:3"],
            "supported_sizes": ["1024x1024", "1536x1024", "1024x1536"],
            "supported_quality_levels": ["standard", "hd"],
            "supported_styles": ["vivid", "natural"],
            "default_aspect_ratio": "1:1",
            "default_quality": "standard",
            "supports_reference_images": False,
            "supports_response_modalities": False
        },
        "dall-e-3": {
            "supported_aspect_ratios": ["1:1"],
            "supported_sizes": ["1024x1024", "1792x1024", "1024x1792"],
            "supported_quality_levels": ["standard", "hd"],
            "supported_styles": ["vivid", "natural"],
            "default_aspect_ratio": "1:1",
            "default_quality": "standard",
            "supports_reference_images": False,
            "supports_response_modalities": False
        }
    }
    
    @classmethod
    def validate_parameters(cls, model: str, parameters: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """
        Validate image generation parameters for a specific model.
        
        Args:
            model: Model name
            parameters: Parameters to validate
            
        Returns:
            Tuple of (validated_parameters, warnings)
        """
        if model not in cls.MODEL_CONFIGS:
            return parameters, [f"Unknown model: {model}"]
        
        config = cls.MODEL_CONFIGS[model]
        validated = parameters.copy()
        warnings = []
        
        # Validate aspect ratio
        if "aspect_ratio" in parameters:
            aspect_ratio = parameters["aspect_ratio"]
            if aspect_ratio not in config["supported_aspect_ratios"]:
                warnings.append(f"Aspect ratio '{aspect_ratio}' not supported by {model}. "
                               f"Supported: {config['supported_aspect_ratios']}")
                validated["aspect_ratio"] = config["default_aspect_ratio"]
        
        # Validate size
        if "size" in parameters:
            size = parameters["size"]
            if size not in config["supported_sizes"]:
                warnings.append(f"Size '{size}' not supported by {model}. "
                               f"Supported: {config['supported_sizes']}")
                # Try to map aspect ratio to valid size
                if "aspect_ratio" in validated:
                    validated["size"] = cls._map_aspect_ratio_to_size(
                        validated["aspect_ratio"], config["supported_sizes"]
                    )
        
        # Validate quality
        if "quality" in parameters:
            quality = parameters["quality"]
            if quality not in config["supported_quality_levels"]:
                warnings.append(f"Quality '{quality}' not supported by {model}. "
                               f"Supported: {config['supported_quality_levels']}")
                validated["quality"] = config["default_quality"]
        
        # Validate style
        if "style" in parameters:
            style = parameters["style"]
            if config["supported_styles"] and style not in config["supported_styles"]:
                warnings.append(f"Style '{style}' not supported by {model}. "
                               f"Supported: {config['supported_styles']}")
                validated.pop("style", None)
        
        # Validate reference images
        if "reference_images" in parameters and not config["supports_reference_images"]:
            warnings.append(f"Reference images not supported by {model}")
            validated.pop("reference_images", None)
        
        # Validate response modalities
        if "response_modalities" in parameters and not config["supports_response_modalities"]:
            warnings.append(f"Response modalities not supported by {model}")
            validated.pop("response_modalities", None)
        
        return validated, warnings
    
    @classmethod
    def _map_aspect_ratio_to_size(cls, aspect_ratio: str, supported_sizes: List[str]) -> str:
        """Map aspect ratio to the best matching supported size."""
        # Aspect ratio to size mapping
        ratio_mappings = {
            "1:1": ["1024x1024"],
            "3:2": ["1536x1024", "1248x832"],
            "2:3": ["1024x1536", "832x1248"],
            "4:3": ["1184x864"],
            "3:4": ["864x1184"],
            "16:9": ["1344x768"],
            "9:16": ["768x1344"],
            "21:9": ["1536x672"],
            "4:5": ["896x1152"],
            "5:4": ["1152x896"]
        }
        
        if aspect_ratio in ratio_mappings:
            for size in ratio_mappings[aspect_ratio]:
                if size in supported_sizes:
                    return size
        
        # Fallback to first supported size
        return supported_sizes[0] if supported_sizes else "1024x1024"
    
    @classmethod
    def get_model_capabilities(cls, model: str) -> Optional[Dict[str, Any]]:
        """Get capabilities for a specific model."""
        return cls.MODEL_CONFIGS.get(model)
    
    @classmethod
    def list_supported_models(cls) -> List[str]:
        """List all supported image generation models."""
        return list(cls.MODEL_CONFIGS.keys())
