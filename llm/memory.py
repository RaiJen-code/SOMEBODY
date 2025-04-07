"""
Memory management for SOMEBODY LLM
"""

import logging
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class Message:
    """Message in a conversation"""
    
    def __init__(self, role: str, content: str, timestamp: Optional[datetime] = None):
        """Initialize message
        
        Args:
            role: Role of the message sender (user/assistant/system)
            content: Content of the message
            timestamp: Time when message was created (default: now)
        """
        self.role = role
        self.content = content
        self.timestamp = timestamp or datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary
        
        Returns:
            Dictionary representation of message
        """
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary
        
        Args:
            data: Dictionary representation of message
            
        Returns:
            Message instance
        """
        timestamp = datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else None
        return cls(data["role"], data["content"], timestamp)


class ConversationMemory:
    """Memory for storing conversation history"""
    
    def __init__(self, max_messages: int = 100):
        """Initialize conversation memory
        
        Args:
            max_messages: Maximum number of messages to store
        """
        self.messages: List[Message] = []
        self.max_messages = max_messages
    
    def add_message(self, role: str, content: str) -> Message:
        """Add message to memory
        
        Args:
            role: Role of the message sender
            content: Content of the message
            
        Returns:
            The added message
        """
        message = Message(role, content)
        self.messages.append(message)
        
        # Trim if exceeds maximum
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
            
        return message
    
    def get_history(self, max_messages: Optional[int] = None) -> List[Message]:
        """Get conversation history
        
        Args:
            max_messages: Maximum number of messages to retrieve (default: all)
            
        Returns:
            List of messages
        """
        if max_messages is None:
            return self.messages
        return self.messages[-max_messages:]
    
    def clear(self) -> None:
        """Clear conversation history"""
        self.messages = []
    
    def save(self, file_path: str) -> bool:
        """Save conversation history to file
        
        Args:
            file_path: Path to save file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(
                    {
                        "messages": [msg.to_dict() for msg in self.messages]
                    },
                    f,
                    ensure_ascii=False,
                    indent=2
                )
            logger.info(f"Saved conversation to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving conversation to {file_path}: {str(e)}")
            return False
    
    @classmethod
    def load(cls, file_path: str) -> Optional['ConversationMemory']:
        """Load conversation history from file
        
        Args:
            file_path: Path to load file
            
        Returns:
            ConversationMemory instance, or None if failed
        """
        try:
            if not os.path.exists(file_path):
                logger.warning(f"Conversation file {file_path} not found")
                return None
                
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            memory = cls()
            memory.messages = [Message.from_dict(msg) for msg in data.get("messages", [])]
            logger.info(f"Loaded conversation from {file_path} with {len(memory.messages)} messages")
            return memory
        except Exception as e:
            logger.error(f"Error loading conversation from {file_path}: {str(e)}")
            return None
    
    def format_for_prompt(self, max_messages: Optional[int] = None) -> str:
        """Format conversation history for inclusion in a prompt
        
        Args:
            max_messages: Maximum number of messages to include
            
        Returns:
            Formatted conversation history string
        """
        history = self.get_history(max_messages)
        result = []
        
        for msg in history:
            prefix = "User: " if msg.role == "user" else "SOMEBODY: "
            result.append(f"{prefix}{msg.content}")
            
        return "\n".join(result)