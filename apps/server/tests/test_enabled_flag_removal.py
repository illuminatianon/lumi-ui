"""Tests to verify that the enabled flags have been properly removed."""

import pytest
from unittest.mock import Mock, patch

from config.models import UnifiedInferenceConfig, InferenceProviderConfig, LumiConfig
from services.inference.models import ProviderConfig, InferenceConfig
from services.inference.factory import create_unified_inference_service, _build_inference_config
from services.inference.base import Provider


class TestEnabledFlagRemoval:
    """Test that enabled flags have been removed from all configurations."""
    
    def test_unified_inference_config_no_enabled_flag(self):
        """Test that UnifiedInferenceConfig no longer has enabled flag."""
        config = UnifiedInferenceConfig()
        
        # Should not have enabled attribute
        assert not hasattr(config, 'enabled')
        
        # Should have other expected attributes
        assert hasattr(config, 'default_provider')
        assert hasattr(config, 'fallback_providers')
        assert hasattr(config, 'openai')
        assert hasattr(config, 'google')
    
    def test_inference_provider_config_no_enabled_flag(self):
        """Test that InferenceProviderConfig no longer has enabled flag."""
        config = InferenceProviderConfig()
        
        # Should not have enabled attribute
        assert not hasattr(config, 'enabled')
        
        # Should have other expected attributes
        assert hasattr(config, 'api_key')
        assert hasattr(config, 'base_url')
        assert hasattr(config, 'default_model')
        assert hasattr(config, 'timeout')
    
    def test_provider_config_no_enabled_flag(self):
        """Test that ProviderConfig no longer has enabled flag."""
        config = ProviderConfig(
            api_key="test-key",
            default_model="test-model"
        )
        
        # Should not have enabled attribute
        assert not hasattr(config, 'enabled')
        
        # Should have other expected attributes
        assert config.api_key == "test-key"
        assert config.default_model == "test-model"
    
    def test_base_provider_is_available_no_enabled_check(self):
        """Test that Provider.is_available() no longer checks enabled flag."""
        # Create a concrete test provider class
        class TestProvider(Provider):
            def get_provider_name(self):
                return "test"

            def map_parameters(self, request, model_config):
                return {}

            def prepare_attachments(self, request):
                return []

            async def process_request(self, request, model_config):
                return Mock()

        # Mock provider with API key but no enabled flag
        provider = TestProvider(config={"api_key": "test-key"})

        # Should be available with just API key
        assert provider.is_available() is True

        # Should not be available without API key
        provider_no_key = TestProvider(config={})
        assert provider_no_key.is_available() is False
    
    def test_factory_build_inference_config_no_enabled_checks(self):
        """Test that factory no longer checks enabled flags when building config."""
        # Mock app config
        mock_app_config = Mock()
        mock_app_config.api_keys.openai = "openai-key"
        mock_app_config.api_keys.gemini = "gemini-key"
        mock_app_config.inference.openai.base_url = None
        mock_app_config.inference.openai.default_model = "gpt-4o"
        mock_app_config.inference.openai.rate_limit_rpm = None
        mock_app_config.inference.google.base_url = None
        mock_app_config.inference.google.default_model = "gemini-1.5-pro"
        mock_app_config.inference.google.rate_limit_rpm = None
        mock_app_config.inference.default_provider = "auto"
        mock_app_config.inference.fallback_providers = ["openai", "google"]
        mock_app_config.inference.provider_selection_strategy = "cost_optimized"
        
        # Build config
        with patch('services.inference.factory._get_openai_models') as mock_openai_models, \
             patch('services.inference.factory._get_google_models') as mock_google_models:
            
            mock_openai_models.return_value = {}
            mock_google_models.return_value = {}
            
            config = _build_inference_config(mock_app_config)
        
        # Should have both providers since API keys are available
        assert "openai" in config.providers
        assert "google" in config.providers
        
        # Provider configs should not have enabled field
        openai_config = config.providers["openai"]
        google_config = config.providers["google"]
        
        assert not hasattr(openai_config, 'enabled')
        assert not hasattr(google_config, 'enabled')
        
        # Should have API keys
        assert openai_config.api_key == "openai-key"
        assert google_config.api_key == "gemini-key"
    
    def test_factory_build_inference_config_missing_api_keys(self):
        """Test that providers are not included when API keys are missing."""
        # Mock app config with no API keys
        mock_app_config = Mock()
        mock_app_config.api_keys.openai = None
        mock_app_config.api_keys.gemini = None
        mock_app_config.inference.default_provider = "auto"
        mock_app_config.inference.fallback_providers = ["openai", "google"]
        mock_app_config.inference.provider_selection_strategy = "cost_optimized"
        
        # Build config
        config = _build_inference_config(mock_app_config)
        
        # Should have no providers since no API keys
        assert len(config.providers) == 0
    
    def test_factory_create_service_no_enabled_check(self):
        """Test that create_unified_inference_service no longer checks enabled flag."""
        with patch('services.inference.factory.get_config') as mock_get_config, \
             patch('services.inference.factory._build_inference_config') as mock_build_config, \
             patch('services.inference.factory.UnifiedInferenceService') as mock_service_class:
            
            # Mock config without enabled flag
            mock_config = Mock()
            mock_get_config.return_value = mock_config
            
            # Mock inference config
            mock_inference_config = Mock()
            mock_build_config.return_value = mock_inference_config
            
            # Mock service instance
            mock_service = Mock()
            mock_service_class.return_value = mock_service
            
            # Create service
            result = create_unified_inference_service()
            
            # Should create service successfully
            assert result == mock_service
            mock_service_class.assert_called_once_with(mock_inference_config)
    
    def test_default_config_no_enabled_flags(self):
        """Test that default configuration doesn't include enabled flags."""
        # This would be loaded from the default configuration
        config = UnifiedInferenceConfig()
        
        # Check that the config can be created without enabled flags
        assert config.default_provider == "auto"
        assert "openai" in config.fallback_providers
        assert "google" in config.fallback_providers
        
        # Check provider configs
        assert config.openai.default_model == "gpt-4o"
        assert config.google.default_model == "gemini-1.5-pro"
        
        # Ensure no enabled attributes exist
        assert not hasattr(config, 'enabled')
        assert not hasattr(config.openai, 'enabled')
        assert not hasattr(config.google, 'enabled')


if __name__ == "__main__":
    pytest.main([__file__])
