#!/usr/bin/env python3
"""
Example demonstrating multi-turn chat functionality.

This example shows how to:
1. Use the new messages format for multi-turn conversations
2. Maintain conversation history on the client side
3. Handle both single-turn and multi-turn requests
4. Manage conversation length to avoid token limits

Run this example with a running Lumi server and valid API keys configured.
"""

import requests
import json
from typing import List, Dict, Any


class MultiTurnChatExample:
    """Example client for multi-turn chat functionality."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.messages: List[Dict[str, str]] = []
    
    def set_system_message(self, content: str):
        """Set or update the system message."""
        # Remove any existing system messages
        self.messages = [m for m in self.messages if m["role"] != "system"]
        # Add new system message at the beginning
        self.messages.insert(0, {"role": "system", "content": content})
        print(f"🤖 System: {content}")
    
    def send_message(self, content: str, model: str = "openai/gpt-4o") -> str:
        """Send a message and get response using multi-turn format."""
        print(f"\n👤 User: {content}")
        
        # Add user message to history
        self.messages.append({"role": "user", "content": content})
        
        try:
            # Send request with full conversation history
            response = requests.post(f"{self.base_url}/api/ai/chat", json={
                "model": model,
                "messages": self.messages,
                "temperature": 0.7,
                "max_tokens": 500
            })
            
            result = response.json()
            
            if result.get("success"):
                assistant_response = result["content"]
                # Add assistant response to history
                self.messages.append({
                    "role": "assistant", 
                    "content": assistant_response
                })
                print(f"🤖 Assistant: {assistant_response}")
                return assistant_response
            else:
                error_msg = result.get("error", "Unknown error")
                print(f"❌ Error: {error_msg}")
                return f"Error: {error_msg}"
                
        except requests.RequestException as e:
            error_msg = f"Request failed: {e}"
            print(f"❌ {error_msg}")
            return error_msg
    
    def send_single_turn_message(self, content: str, system_message: str = None, model: str = "openai/gpt-4o") -> str:
        """Send a single-turn message (backward compatibility example)."""
        print(f"\n👤 User (single-turn): {content}")
        
        try:
            payload = {
                "model": model,
                "prompt": content,
                "temperature": 0.7,
                "max_tokens": 500
            }
            
            if system_message:
                payload["system_message"] = system_message
                print(f"🤖 System (single-turn): {system_message}")
            
            response = requests.post(f"{self.base_url}/api/ai/chat", json=payload)
            result = response.json()
            
            if result.get("success"):
                assistant_response = result["content"]
                print(f"🤖 Assistant (single-turn): {assistant_response}")
                return assistant_response
            else:
                error_msg = result.get("error", "Unknown error")
                print(f"❌ Error: {error_msg}")
                return f"Error: {error_msg}"
                
        except requests.RequestException as e:
            error_msg = f"Request failed: {e}"
            print(f"❌ {error_msg}")
            return error_msg
    
    def show_conversation_history(self):
        """Display the current conversation history."""
        print("\n📜 Conversation History:")
        for i, message in enumerate(self.messages):
            role_emoji = {"system": "🤖", "user": "👤", "assistant": "🤖"}
            emoji = role_emoji.get(message["role"], "❓")
            print(f"  {i+1}. {emoji} {message['role'].title()}: {message['content'][:100]}{'...' if len(message['content']) > 100 else ''}")
    
    def clear_history(self):
        """Clear conversation history, keeping system message."""
        system_messages = [m for m in self.messages if m["role"] == "system"]
        self.messages = system_messages
        print("\n🧹 Conversation history cleared (system message preserved)")
    
    def trim_conversation(self, max_messages: int = 10):
        """Keep conversation under a certain length."""
        if len(self.messages) <= max_messages:
            return
        
        # Keep system message and recent messages
        system_msgs = [m for m in self.messages if m["role"] == "system"]
        other_msgs = [m for m in self.messages if m["role"] != "system"]
        
        # Keep the most recent messages
        recent_msgs = other_msgs[-(max_messages - len(system_msgs)):]
        
        self.messages = system_msgs + recent_msgs
        print(f"\n✂️ Conversation trimmed to {len(self.messages)} messages")


def main():
    """Run the multi-turn chat example."""
    print("🚀 Multi-Turn Chat Example")
    print("=" * 50)
    
    # Initialize the chat client
    chat = MultiTurnChatExample()
    
    # Example 1: Single-turn request (backward compatibility)
    print("\n📝 Example 1: Single-Turn Request (Backward Compatible)")
    chat.send_single_turn_message(
        "What is the capital of France?",
        system_message="You are a helpful geography assistant."
    )
    
    # Example 2: Multi-turn conversation
    print("\n📝 Example 2: Multi-Turn Conversation")
    chat.set_system_message("You are a helpful math tutor who explains concepts clearly.")
    
    # Start a conversation about math
    chat.send_message("What is calculus?")
    chat.send_message("Can you give me a simple example of a derivative?")
    chat.send_message("What was the first concept I asked you about?")
    
    # Show conversation history
    chat.show_conversation_history()
    
    # Example 3: Conversation management
    print("\n📝 Example 3: Conversation Management")
    
    # Add more messages to demonstrate trimming
    chat.send_message("Tell me about integrals")
    chat.send_message("How are derivatives and integrals related?")
    
    # Trim conversation
    chat.trim_conversation(max_messages=6)
    chat.show_conversation_history()
    
    # Clear and start fresh
    chat.clear_history()
    chat.set_system_message("You are a creative writing assistant.")
    chat.send_message("Help me write a short story about a robot.")
    
    print("\n✅ Example completed!")
    print("\nKey takeaways:")
    print("- Multi-turn conversations maintain context across messages")
    print("- Single-turn requests still work for backward compatibility")
    print("- Conversation history is managed on the client side")
    print("- You can trim conversations to manage token limits")


if __name__ == "__main__":
    main()
