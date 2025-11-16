"""Services package for Lumi UI backend."""

import logging

logger = logging.getLogger(__name__)

# Global service instances
_langchain_service = None
_compatibility_service = None


def get_langchain_service():
    """Get the global LangChain service instance."""
    global _langchain_service
    if _langchain_service is None:
        try:
            from .langchain_service import LangChainService
            _langchain_service = LangChainService()
        except Exception as e:
            logger.error(f"Failed to initialize LangChain service: {e}")
            raise
    return _langchain_service


def get_inference_service():
    """Get the global inference service instance (compatibility layer)."""
    global _compatibility_service
    if _compatibility_service is None:
        try:
            from .inference.compatibility import InferenceCompatibilityService
            _compatibility_service = InferenceCompatibilityService()
        except Exception as e:
            logger.error(f"Failed to initialize inference compatibility service: {e}")
            raise
    return _compatibility_service


def get_unified_inference_service():
    """Get the unified inference service."""
    try:
        from .inference.factory import get_unified_inference_service as _get_unified
        return _get_unified()
    except Exception as e:
        logger.error(f"Failed to get unified inference service: {e}")
        return None


__all__ = [
    "get_langchain_service",
    "get_inference_service",
    "get_unified_inference_service"
]
