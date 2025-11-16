#!/usr/bin/env python3
"""Setup script for testing unified inference with GPT-5."""

import os
import sys
from pathlib import Path

def setup_test_environment():
    """Set up the test environment for GPT-5 testing."""
    
    print("🔧 Setting up test environment for GPT-5")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("test_gpt5.py").exists():
        print("❌ Please run this script from the apps/server directory")
        return False
    
    # Check for API key
    openai_key = os.getenv("LUMI_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    
    if not openai_key:
        print("⚠️  No OpenAI API key found in environment variables")
        print("\nTo set up your API key, choose one of these options:")
        print("\n1. Set environment variable (recommended for testing):")
        print("   export LUMI_OPENAI_API_KEY='your-api-key-here'")
        print("   # or")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        
        print("\n2. Create a config file:")
        config_dir = Path.home() / ".config" / "lumi"
        config_file = config_dir / "config.hjson"
        
        print(f"   mkdir -p {config_dir}")
        print(f"   cp test_config_example.hjson {config_file}")
        print(f"   # Then edit {config_file} and add your API key")
        
        print("\n3. For this session only:")
        api_key = input("\nEnter your OpenAI API key (or press Enter to skip): ").strip()
        if api_key:
            os.environ["LUMI_OPENAI_API_KEY"] = api_key
            print("✅ API key set for this session")
        else:
            print("⚠️  Skipping API key setup - tests will run in demo mode")
    else:
        print(f"✅ OpenAI API key found: {openai_key[:8]}...")
    
    # Check virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment detected")
    else:
        print("⚠️  Not in a virtual environment")
        print("   Consider running: source venv/bin/activate")
    
    # Check dependencies
    try:
        import openai
        print(f"✅ OpenAI library available: {openai.__version__}")
    except ImportError:
        print("❌ OpenAI library not found")
        print("   Run: pip install openai")
        return False
    
    try:
        import google.generativeai
        print(f"✅ Google Generative AI library available")
    except ImportError:
        print("⚠️  Google Generative AI library not found (optional)")
        print("   Run: pip install google-generativeai")
    
    print("\n🚀 Ready to test!")
    print("Run the test with: python test_gpt5.py")
    
    return True


def show_usage_examples():
    """Show usage examples for the test script."""
    
    print("\n📚 Usage Examples")
    print("=" * 30)
    
    print("1. Basic test with environment variable:")
    print("   export LUMI_OPENAI_API_KEY='your-key'")
    print("   python test_gpt5.py")
    
    print("\n2. Test with virtual environment:")
    print("   source venv/bin/activate")
    print("   export LUMI_OPENAI_API_KEY='your-key'")
    print("   python test_gpt5.py")
    
    print("\n3. What the test does:")
    print("   - Loads configuration like the server")
    print("   - Creates unified inference service")
    print("   - Makes a GPT-5 request with reasoning_effort parameter")
    print("   - Demonstrates parameter mapping (max_tokens → max_completion_tokens)")
    print("   - Falls back to GPT-4o if GPT-5 is not available")
    
    print("\n4. Expected output:")
    print("   - Shows parameter mapping differences between GPT-4o and GPT-5")
    print("   - Displays the actual LLM response")
    print("   - Confirms model-specific parameter handling works")


if __name__ == "__main__":
    print("🧪 GPT-5 Test Setup")
    print("This script helps set up the environment for testing")
    print("the unified inference infrastructure with GPT-5.\n")
    
    success = setup_test_environment()
    show_usage_examples()
    
    if success:
        print("\n✅ Setup complete! You can now run: python test_gpt5.py")
    else:
        print("\n❌ Setup incomplete. Please address the issues above.")
