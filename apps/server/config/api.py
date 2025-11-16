"""Configuration API endpoints for safe frontend access."""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
import logging

from .models import (
    SafeUserConfig, SafeAppConfig, UserConfigUpdate,
    UIStateUpdate, LumiState
)
from .loader import ConfigurationLoader
from .state_manager import StateManager

logger = logging.getLogger(__name__)

# Global configuration loader instance
config_loader = ConfigurationLoader()

# Initialize configuration on module load
config_loader.load_configuration()
config_loader.load_state()

router = APIRouter(prefix="/api/config", tags=["configuration"])


def get_config_loader() -> ConfigurationLoader:
    """Dependency to get configuration loader instance."""
    return config_loader


def get_state_manager() -> StateManager:
    """Dependency to get state manager instance."""
    return StateManager(config_loader)


@router.get("/user", response_model=SafeUserConfig)
async def get_user_config(
    loader: ConfigurationLoader = Depends(get_config_loader)
) -> SafeUserConfig:
    """Get safe user configuration for frontend."""
    try:
        config = loader.config
        return SafeUserConfig(
            theme=config.user.theme,
            language=config.user.language,
            timezone=config.user.timezone,
            date_format=config.user.date_format,
            time_format=config.user.time_format,
        )
    except Exception as e:
        logger.error(f"Failed to get user config: {e}")
        raise HTTPException(status_code=500, detail="Failed to load user configuration")


@router.get("/app", response_model=SafeAppConfig)
async def get_app_config(
    loader: ConfigurationLoader = Depends(get_config_loader)
) -> SafeAppConfig:
    """Get safe application configuration for frontend."""
    try:
        config = loader.config
        return SafeAppConfig(
            version="1.0.0",
            auto_save=config.app.auto_save,
            auto_save_interval=config.app.auto_save_interval,
            max_recent_items=config.app.max_recent_items,
        )
    except Exception as e:
        logger.error(f"Failed to get app config: {e}")
        raise HTTPException(status_code=500, detail="Failed to load app configuration")


@router.put("/user")
async def update_user_config(
    updates: UserConfigUpdate,
    loader: ConfigurationLoader = Depends(get_config_loader)
) -> Dict[str, str]:
    """Update user configuration from frontend."""
    try:
        # Convert to dict, excluding None values
        update_dict = {k: v for k, v in updates.model_dump().items() if v is not None}
        
        if not update_dict:
            raise HTTPException(status_code=400, detail="No valid updates provided")
        
        success = loader.update_user_config(update_dict)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update configuration")
        
        return {"message": "User configuration updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update user config: {e}")
        raise HTTPException(status_code=500, detail="Failed to update user configuration")


@router.get("/ui-state")
async def get_ui_state(
    loader: ConfigurationLoader = Depends(get_config_loader)
) -> Dict[str, Any]:
    """Get UI state for frontend."""
    try:
        state = loader.state
        return {
            "sidebar_collapsed": state.ui.sidebar_collapsed,
            "active_tab": state.ui.active_tab,
            "selected_theme": state.ui.selected_theme,
            "panel_sizes": {
                "sidebar": state.ui.panel_sizes.sidebar,
                "main": state.ui.panel_sizes.main,
                "details": state.ui.panel_sizes.details,
            },
            "window_state": {
                "width": state.ui.window_state.width,
                "height": state.ui.window_state.height,
                "maximized": state.ui.window_state.maximized,
            }
        }
    except Exception as e:
        logger.error(f"Failed to get UI state: {e}")
        raise HTTPException(status_code=500, detail="Failed to load UI state")


@router.put("/ui-state")
async def update_ui_state(
    updates: UIStateUpdate,
    loader: ConfigurationLoader = Depends(get_config_loader)
) -> Dict[str, str]:
    """Update UI state from frontend."""
    try:
        state = loader.state
        
        # Apply updates to current state
        if updates.sidebar_collapsed is not None:
            state.ui.sidebar_collapsed = updates.sidebar_collapsed
        
        if updates.active_tab is not None:
            state.ui.active_tab = updates.active_tab
        
        if updates.selected_theme is not None:
            state.ui.selected_theme = updates.selected_theme
        
        if updates.panel_sizes is not None:
            for key, value in updates.panel_sizes.items():
                if hasattr(state.ui.panel_sizes, key):
                    setattr(state.ui.panel_sizes, key, value)
        
        if updates.window_state is not None:
            for key, value in updates.window_state.items():
                if hasattr(state.ui.window_state, key):
                    setattr(state.ui.window_state, key, value)
        
        # Save updated state
        success = loader.save_state(state)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save UI state")
        
        return {"message": "UI state updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update UI state: {e}")
        raise HTTPException(status_code=500, detail="Failed to update UI state")


@router.get("/health")
async def config_health_check(
    loader: ConfigurationLoader = Depends(get_config_loader)
) -> Dict[str, Any]:
    """Health check for configuration system."""
    try:
        # Check if configuration is loaded
        config_loaded = loader._config is not None
        state_loaded = loader._state is not None
        
        # Check directory structure
        fs_ok = loader.fs.ensure_directory_structure()
        
        return {
            "status": "healthy" if (config_loaded and state_loaded and fs_ok) else "degraded",
            "config_loaded": config_loaded,
            "state_loaded": state_loaded,
            "filesystem_ok": fs_ok,
            "config_file_exists": loader.fs.config_file.exists(),
            "state_file_exists": loader.fs.state_file.exists(),
        }
    except Exception as e:
        logger.error(f"Config health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@router.get("/recent")
async def get_recent_items(
    state_manager: StateManager = Depends(get_state_manager)
) -> Dict[str, Any]:
    """Get recent items (conversations, searches, actions)."""
    try:
        return state_manager.get_recent_items()
    except Exception as e:
        logger.error(f"Failed to get recent items: {e}")
        raise HTTPException(status_code=500, detail="Failed to load recent items")


@router.post("/recent/search")
async def add_recent_search(
    search_data: Dict[str, str],
    state_manager: StateManager = Depends(get_state_manager)
) -> Dict[str, str]:
    """Add a search query to recent items."""
    try:
        query = search_data.get("query")
        if not query:
            raise HTTPException(status_code=400, detail="Search query is required")

        success = state_manager.add_recent_search(query)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to add recent search")

        return {"message": "Recent search added successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add recent search: {e}")
        raise HTTPException(status_code=500, detail="Failed to add recent search")


@router.get("/user-data")
async def get_user_data(
    state_manager: StateManager = Depends(get_state_manager)
) -> Dict[str, Any]:
    """Get user data (bookmarks, favorites, etc.)."""
    try:
        return state_manager.get_user_data()
    except Exception as e:
        logger.error(f"Failed to get user data: {e}")
        raise HTTPException(status_code=500, detail="Failed to load user data")


@router.post("/bookmarks")
async def add_bookmark(
    bookmark_data: Dict[str, Any],
    state_manager: StateManager = Depends(get_state_manager)
) -> Dict[str, str]:
    """Add a bookmark to user data."""
    try:
        success = state_manager.add_bookmark(bookmark_data)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to add bookmark")

        return {"message": "Bookmark added successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add bookmark: {e}")
        raise HTTPException(status_code=500, detail="Failed to add bookmark")


@router.delete("/bookmarks/{bookmark_id}")
async def remove_bookmark(
    bookmark_id: str,
    state_manager: StateManager = Depends(get_state_manager)
) -> Dict[str, str]:
    """Remove a bookmark from user data."""
    try:
        success = state_manager.remove_bookmark(bookmark_id)
        if not success:
            raise HTTPException(status_code=404, detail="Bookmark not found")

        return {"message": "Bookmark removed successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to remove bookmark: {e}")
        raise HTTPException(status_code=500, detail="Failed to remove bookmark")


@router.get("/metrics")
async def get_metrics(
    state_manager: StateManager = Depends(get_state_manager)
) -> Dict[str, Any]:
    """Get application metrics."""
    try:
        return state_manager.get_metrics()
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to load metrics")
