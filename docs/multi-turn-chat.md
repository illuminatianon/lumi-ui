# Multi-Turn Chat Support

This document describes the multi-turn chat functionality that allows for conversation history to be embedded in the prompt configuration without introducing a new conversation abstraction.

## Overview

The system now supports both single-turn and multi-turn chat formats:

- **Single-turn (backward compatible)**: Uses `prompt` and optional `system_message` fields
- **Multi-turn (new)**: Uses a `messages` array with conversation history

The caller is responsible for managing conversation persistence - the system does not store conversation state.

## Message Format

### Message Structure

```typescript
interface Message {
  role: "system" | "user" | "assistant";
  content: string;
}
```

### Roles

- **system**: System instructions or context
- **user**: Messages from the user
- **assistant**: Previous responses from the AI assistant

## API Usage

### Single-Turn Request (Backward Compatible)

```json
{
  "model": "openai/gpt-4o",
  "prompt": "What is machine learning?",
  "system_message": "You are a helpful AI assistant.",
  "temperature": 0.7,
  "max_tokens": 500
}
```

### Multi-Turn Request (New)

```json
{
  "model": "openai/gpt-4o",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful AI assistant."
    },
    {
      "role": "user", 
      "content": "What is machine learning?"
    },
    {
      "role": "assistant",
      "content": "Machine learning is a subset of artificial intelligence..."
    },
    {
      "role": "user",
      "content": "Can you give me a simple example?"
    }
  ],
  "temperature": 0.7,
  "max_tokens": 500
}
```

## Validation Rules

1. **Either `prompt` OR `messages` must be provided** - not both
2. **Cannot provide both** `prompt` and `messages` in the same request
3. **Empty messages array** is treated as no messages (validation error)

## Client Implementation Examples

### JavaScript/TypeScript

```typescript
interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

class ChatClient {
  private messages: ChatMessage[] = [];

  // Add system message
  setSystemMessage(content: string) {
    this.messages = this.messages.filter(m => m.role !== 'system');
    this.messages.unshift({ role: 'system', content });
  }

  // Add user message and get response
  async sendMessage(content: string): Promise<string> {
    // Add user message to history
    this.messages.push({ role: 'user', content });

    // Send request with full conversation history
    const response = await fetch('/api/ai/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'openai/gpt-4o',
        messages: this.messages,
        temperature: 0.7
      })
    });

    const result = await response.json();
    
    if (result.success) {
      // Add assistant response to history
      this.messages.push({ 
        role: 'assistant', 
        content: result.content 
      });
      return result.content;
    } else {
      throw new Error(result.error);
    }
  }

  // Clear conversation history
  clearHistory() {
    this.messages = this.messages.filter(m => m.role === 'system');
  }

  // Get conversation history
  getHistory(): ChatMessage[] {
    return [...this.messages];
  }
}
```

### Python

```python
from typing import List, Dict, Any
import requests

class ChatClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.messages: List[Dict[str, str]] = []
    
    def set_system_message(self, content: str):
        """Set or update the system message."""
        self.messages = [m for m in self.messages if m["role"] != "system"]
        self.messages.insert(0, {"role": "system", "content": content})
    
    def send_message(self, content: str, model: str = "openai/gpt-4o") -> str:
        """Send a message and get response."""
        # Add user message to history
        self.messages.append({"role": "user", "content": content})
        
        # Send request with full conversation history
        response = requests.post(f"{self.base_url}/api/ai/chat", json={
            "model": model,
            "messages": self.messages,
            "temperature": 0.7
        })
        
        result = response.json()
        
        if result["success"]:
            # Add assistant response to history
            self.messages.append({
                "role": "assistant", 
                "content": result["content"]
            })
            return result["content"]
        else:
            raise Exception(result["error"])
    
    def clear_history(self):
        """Clear conversation history, keeping system message."""
        self.messages = [m for m in self.messages if m["role"] == "system"]
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self.messages.copy()
```

## Usage Examples

### Basic Multi-Turn Conversation

```python
client = ChatClient()
client.set_system_message("You are a helpful math tutor.")

# First exchange
response1 = client.send_message("What is 2 + 2?")
print(response1)  # "2 + 2 equals 4."

# Follow-up question (with context)
response2 = client.send_message("What about 2 * 2?")
print(response2)  # "2 * 2 equals 4 as well."

# The assistant has context from previous messages
response3 = client.send_message("Which operation did I ask about first?")
print(response3)  # "You first asked about addition (2 + 2)."
```

### Managing Conversation Length

```python
def trim_conversation(messages: List[Dict], max_messages: int = 10):
    """Keep conversation under a certain length."""
    if len(messages) <= max_messages:
        return messages

    # Keep system message and recent messages
    system_msgs = [m for m in messages if m["role"] == "system"]
    other_msgs = [m for m in messages if m["role"] != "system"]

    # Keep the most recent messages
    recent_msgs = other_msgs[-(max_messages - len(system_msgs)):]

    return system_msgs + recent_msgs

# Usage
client.messages = trim_conversation(client.messages, max_messages=20)
```

### Image Generation with Multi-Turn Context

```python
# Multi-turn image generation with conversation context
client = ChatClient()
client.set_system_message("You are a creative AI artist who generates images based on conversation context.")

# Build up context through conversation
client.send_message("I'm working on a sci-fi project")
# Response: "That sounds exciting! What kind of sci-fi project are you working on?"

client.send_message("It's about a space colony on Mars in the year 2150")
# Response: "Fascinating! A Mars colony in 2150 would have some amazing technology and architecture."

# Now request image generation with full context
response = requests.post('/api/ai/chat', json={
    "model": "google/gemini-2.5-flash-image",
    "messages": client.get_history() + [
        {"role": "user", "content": "Create a concept art image of the main city in this Mars colony"}
    ],
    "extras": {
        "aspect_ratio": "16:9",
        "response_modalities": ["Text", "Image"]
    }
})

# The AI now has full context about the sci-fi project and Mars colony setting
```

## Provider-Specific Behavior

### OpenAI
- Messages are passed directly to the OpenAI Chat Completions API
- Supports all message roles (system, user, assistant)
- Images can be attached to the last user message in multi-turn conversations

### Google (Gemini)
- System messages are converted to user messages (Gemini limitation)
- Assistant messages are mapped to "model" role
- Multi-turn conversations are supported natively
- **Image generation**: Works with multi-turn conversations - the conversation context helps guide image creation
- **Vision analysis**: Attachments are added to the last user message in multi-turn conversations

## Error Handling

The API will return validation errors for:

```json
// Missing both prompt and messages
{
  "model": "openai/gpt-4o"
}
// Returns: 422 Validation Error

// Providing both prompt and messages
{
  "model": "openai/gpt-4o",
  "prompt": "Hello",
  "messages": [{"role": "user", "content": "Hi"}]
}
// Returns: 422 Validation Error
```

## Migration Guide

### From Single-Turn to Multi-Turn

**Before:**
```json
{
  "model": "openai/gpt-4o",
  "prompt": "Continue this conversation...",
  "system_message": "You are helpful"
}
```

**After:**
```json
{
  "model": "openai/gpt-4o", 
  "messages": [
    {"role": "system", "content": "You are helpful"},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"},
    {"role": "user", "content": "Continue this conversation..."}
  ]
}
```

## Best Practices

1. **Always include a system message** for consistent behavior
2. **Manage conversation length** to avoid token limits
3. **Store conversation state** on the client side
4. **Handle errors gracefully** when validation fails
5. **Use appropriate models** for your use case (some models work better for conversations)

## Limitations

1. **No server-side persistence** - conversations are not stored
2. **Token limits apply** - long conversations may exceed model limits
3. **Images in multi-turn** are currently attached to the last user message only
4. **Provider differences** - some providers have role mapping limitations
