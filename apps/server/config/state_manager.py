"""State management for Lumi UI application."""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from .models import LumiState, MetricsState
from .loader import ConfigurationLoader

logger = logging.getLogger(__name__)


class StateManager:
    """Manages application state persistence and updates."""
    
    def __init__(self, config_loader: ConfigurationLoader):
        """Initialize state manager.
        
        Args:
            config_loader: Configuration loader instance
        """
        self.loader = config_loader
    
    def increment_startup_count(self) -> bool:
        """Increment application startup count.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            state = self.loader.state
            state.metrics.startup_count += 1
            state.session.last_active = datetime.now().isoformat()
            state.session.app_version = "1.0.0"
            
            return self.loader.save_state(state)
        except Exception as e:
            logger.error(f"Failed to increment startup count: {e}")
            return False
    
    def add_recent_conversation(self, conversation_data: Dict[str, Any]) -> bool:
        """Add a conversation to recent items.
        
        Args:
            conversation_data: Conversation data to add
            
        Returns:
            True if successful, False otherwise
        """
        try:
            state = self.loader.state
            max_items = self.loader.config.app.max_recent_items
            
            # Add to beginning of list
            state.recent.conversations.insert(0, conversation_data)
            
            # Trim to max items
            if len(state.recent.conversations) > max_items:
                state.recent.conversations = state.recent.conversations[:max_items]
            
            return self.loader.save_state(state)
        except Exception as e:
            logger.error(f"Failed to add recent conversation: {e}")
            return False
    
    def add_recent_search(self, search_query: str) -> bool:
        """Add a search query to recent items.
        
        Args:
            search_query: Search query to add
            
        Returns:
            True if successful, False otherwise
        """
        try:
            state = self.loader.state
            max_items = self.loader.config.app.max_recent_items
            
            search_data = {
                "query": search_query,
                "timestamp": datetime.now().isoformat()
            }
            
            # Remove duplicate if exists
            state.recent.searches = [
                s for s in state.recent.searches 
                if s.get("query") != search_query
            ]
            
            # Add to beginning of list
            state.recent.searches.insert(0, search_data)
            
            # Trim to max items
            if len(state.recent.searches) > max_items:
                state.recent.searches = state.recent.searches[:max_items]
            
            return self.loader.save_state(state)
        except Exception as e:
            logger.error(f"Failed to add recent search: {e}")
            return False
    
    def add_bookmark(self, bookmark_data: Dict[str, Any]) -> bool:
        """Add a bookmark to user data.
        
        Args:
            bookmark_data: Bookmark data to add
            
        Returns:
            True if successful, False otherwise
        """
        try:
            state = self.loader.state
            
            # Check if bookmark already exists (by URL or ID)
            bookmark_id = bookmark_data.get("id") or bookmark_data.get("url")
            if bookmark_id:
                existing = next(
                    (b for b in state.user_data.bookmarks 
                     if b.get("id") == bookmark_id or b.get("url") == bookmark_id),
                    None
                )
                if existing:
                    # Update existing bookmark
                    existing.update(bookmark_data)
                else:
                    # Add new bookmark
                    state.user_data.bookmarks.append(bookmark_data)
            else:
                state.user_data.bookmarks.append(bookmark_data)
            
            return self.loader.save_state(state)
        except Exception as e:
            logger.error(f"Failed to add bookmark: {e}")
            return False
    
    def remove_bookmark(self, bookmark_id: str) -> bool:
        """Remove a bookmark from user data.
        
        Args:
            bookmark_id: ID or URL of bookmark to remove
            
        Returns:
            True if successful, False otherwise
        """
        try:
            state = self.loader.state
            
            # Remove bookmark by ID or URL
            original_count = len(state.user_data.bookmarks)
            state.user_data.bookmarks = [
                b for b in state.user_data.bookmarks
                if b.get("id") != bookmark_id and b.get("url") != bookmark_id
            ]
            
            # Check if anything was removed
            if len(state.user_data.bookmarks) < original_count:
                return self.loader.save_state(state)
            else:
                logger.warning(f"Bookmark not found: {bookmark_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to remove bookmark: {e}")
            return False
    
    def update_usage_time(self, session_duration_ms: int) -> bool:
        """Update total usage time.
        
        Args:
            session_duration_ms: Session duration in milliseconds
            
        Returns:
            True if successful, False otherwise
        """
        try:
            state = self.loader.state
            state.metrics.total_usage_time += session_duration_ms
            state.session.last_active = datetime.now().isoformat()
            
            return self.loader.save_state(state)
        except Exception as e:
            logger.error(f"Failed to update usage time: {e}")
            return False
    
    def record_crash(self) -> bool:
        """Record an application crash.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            state = self.loader.state
            state.metrics.crash_count += 1
            
            return self.loader.save_state(state)
        except Exception as e:
            logger.error(f"Failed to record crash: {e}")
            return False
    
    def get_recent_items(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all recent items.
        
        Returns:
            Dictionary with recent conversations, searches, and actions
        """
        try:
            state = self.loader.state
            return {
                "conversations": state.recent.conversations,
                "searches": state.recent.searches,
                "actions": state.recent.actions,
            }
        except Exception as e:
            logger.error(f"Failed to get recent items: {e}")
            return {"conversations": [], "searches": [], "actions": []}
    
    def get_user_data(self) -> Dict[str, Any]:
        """Get user data (bookmarks, favorites, etc.).
        
        Returns:
            User data dictionary
        """
        try:
            state = self.loader.state
            return {
                "bookmarks": state.user_data.bookmarks,
                "favorites": state.user_data.favorites,
                "custom_settings": state.user_data.custom_settings,
            }
        except Exception as e:
            logger.error(f"Failed to get user data: {e}")
            return {"bookmarks": [], "favorites": [], "custom_settings": {}}
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get application metrics.
        
        Returns:
            Metrics dictionary
        """
        try:
            state = self.loader.state
            return {
                "startup_count": state.metrics.startup_count,
                "total_usage_time": state.metrics.total_usage_time,
                "last_version": state.metrics.last_version,
                "crash_count": state.metrics.crash_count,
                "last_active": state.session.last_active,
            }
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            return {
                "startup_count": 0,
                "total_usage_time": 0,
                "last_version": "1.0.0",
                "crash_count": 0,
                "last_active": None,
            }
