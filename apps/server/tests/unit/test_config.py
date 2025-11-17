#!/usr/bin/env python3
"""Test script for the configuration system."""

import asyncio
import json
import tempfile
import shutil
from pathlib import Path

from config import (
    initialize_config, get_config_loader, get_state_manager,
    get_config, get_state, get_api_key, update_user_config
)
from config.validator import ConfigurationValidator


async def test_configuration_system():
    """Test the configuration system functionality."""
    print("🧪 Testing Lumi UI Configuration System")
    print("=" * 50)
    
    # Create temporary directory for testing
    temp_dir = tempfile.mkdtemp(prefix="lumi_test_")
    print(f"📁 Using temporary directory: {temp_dir}")
    
    try:
        # Test 1: Initialize configuration system
        print("\n1️⃣ Testing configuration initialization...")
        success = initialize_config(temp_dir)
        assert success, "Configuration initialization failed"
        print("✅ Configuration system initialized successfully")
        
        # Test 2: Check directory structure
        print("\n2️⃣ Testing directory structure...")
        test_dir = Path(temp_dir)
        assert (test_dir / "cache").exists(), "Cache directory not created"
        assert (test_dir / "tmp").exists(), "Tmp directory not created"
        assert (test_dir / "logs").exists(), "Logs directory not created"
        assert (test_dir / "state.hjson").exists(), "State file not created"
        print("✅ Directory structure created correctly")
        
        # Test 3: Load and validate configuration
        print("\n3️⃣ Testing configuration loading...")
        config = get_config()
        assert config.user.theme == "dark", "Default theme not set correctly"
        assert config.user.language == "en", "Default language not set correctly"
        assert config.api_keys.openai is None, "Default OpenAI key should be None"
        print("✅ Configuration loaded with correct defaults")
        
        # Test 4: Update user configuration
        print("\n4️⃣ Testing configuration updates...")
        success = update_user_config({"theme": "light", "language": "fr"})
        assert success, "Configuration update failed"
        
        updated_config = get_config()
        assert updated_config.user.theme == "light", "Theme update failed"
        assert updated_config.user.language == "fr", "Language update failed"
        print("✅ Configuration updates working correctly")
        
        # Test 5: Test state management
        print("\n5️⃣ Testing state management...")
        state_manager = get_state_manager()
        
        # Add a recent search
        success = state_manager.add_recent_search("test query")
        assert success, "Failed to add recent search"
        
        recent_items = state_manager.get_recent_items()
        assert len(recent_items["searches"]) > 0, "Recent search not added"
        assert recent_items["searches"][0]["query"] == "test query", "Search query incorrect"
        print("✅ State management working correctly")
        
        # Test 6: Test bookmarks
        print("\n6️⃣ Testing bookmark management...")
        bookmark_data = {
            "id": "test-bookmark",
            "title": "Test Bookmark",
            "url": "https://example.com",
            "description": "A test bookmark"
        }
        
        success = state_manager.add_bookmark(bookmark_data)
        assert success, "Failed to add bookmark"
        
        user_data = state_manager.get_user_data()
        assert len(user_data["bookmarks"]) > 0, "Bookmark not added"
        assert user_data["bookmarks"][0]["title"] == "Test Bookmark", "Bookmark data incorrect"
        print("✅ Bookmark management working correctly")
        
        # Test 7: Test configuration validation
        print("\n7️⃣ Testing configuration validation...")
        test_config_data = {
            "api_keys": {"openai": "invalid-key"},
            "user": {"theme": "invalid-theme", "language": "en"},
            "paths": {"cache": "cache", "tmp": "tmp", "logs": "logs"}
        }
        
        is_valid, errors, validated_config = ConfigurationValidator.validate_config(test_config_data)
        assert not is_valid, "Validation should fail for invalid data"
        assert len(errors) > 0, "Validation errors should be reported"
        print(f"✅ Configuration validation working (found {len(errors)} errors as expected)")
        
        # Test 8: Test API key retrieval
        print("\n8️⃣ Testing API key retrieval...")
        openai_key = get_api_key("openai")
        assert openai_key is None, "OpenAI key should be None by default"
        
        gemini_key = get_api_key("gemini")
        assert gemini_key is None, "Gemini key should be None by default"
        print("✅ API key retrieval working correctly")
        
        # Test 9: Test metrics
        print("\n9️⃣ Testing metrics...")
        metrics = state_manager.get_metrics()
        assert metrics["startup_count"] > 0, "Startup count should be incremented"
        assert "last_active" in metrics, "Last active timestamp should be present"
        print("✅ Metrics tracking working correctly")
        
        print("\n🎉 All tests passed successfully!")
        print(f"📊 Configuration file: {test_dir / 'config.hjson'}")
        print(f"📊 State file: {test_dir / 'state.hjson'}")
        
        # Show final configuration
        print("\n📋 Final Configuration:")
        with open(test_dir / "config.hjson", "r") as f:
            print(f.read())
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
    finally:
        # Cleanup
        print(f"\n🧹 Cleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(test_configuration_system())
