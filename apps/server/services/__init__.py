"""Services package for Lumi UI backend."""

import logging

logger = logging.getLogger(__name__)


def get_unified_inference_service():
    """Get the unified inference service."""
    try:
        from .inference.factory import get_unified_inference_service as _get_unified
        return _get_unified()
    except Exception as e:
        logger.error(f"Failed to get unified inference service: {e}")
        return None


__all__ = [
    "get_unified_inference_service"
]
