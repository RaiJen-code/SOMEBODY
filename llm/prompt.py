"""
Prompt management for SOMEBODY LLM
"""

import logging
from string import Template
from typing import Dict, Any

logger = logging.getLogger(__name__)

class PromptTemplate:
    """Template for generating prompts"""
    
    def __init__(self, template: str):
        """Initialize prompt template
        
        Args:
            template: String template with placeholders for variables
        """
        self.template = Template(template)
        logger.debug(f"Initialized prompt template: {template[:50]}...")
    
    def format(self, **kwargs) -> str:
        """Format template with provided variables
        
        Args:
            **kwargs: Variables to substitute in template
            
        Returns:
            Formatted prompt string
        """
        try:
            return self.template.substitute(**kwargs)
        except KeyError as e:
            logger.error(f"Missing variable in prompt template: {str(e)}")
            # Replace missing variables with placeholder
            return self.template.safe_substitute(**kwargs)

# Define some default templates
DEFAULT_TEMPLATES = {
    "base_conversation": Template(
        "You are SOMEBODY, a helpful AI assistant. $context\n\n"
        "User: $user_input\n"
        "SOMEBODY:"
    ),
    "image_analysis": Template(
        "You are SOMEBODY, a helpful AI assistant. "
        "I'm looking at an image and will describe what I see. $image_description\n\n"
        "User: $user_input\n"
        "SOMEBODY:"
    ),
    "command_execution": Template(
        "You are SOMEBODY, a helpful AI assistant. "
        "I will help you execute commands and report results. $command_result\n\n"
        "User: $user_input\n"
        "SOMEBODY:"
    )
}

def get_template(template_name: str) -> PromptTemplate:
    """Get a prompt template by name
    
    Args:
        template_name: Name of template to retrieve
        
    Returns:
        PromptTemplate instance
    """
    template = DEFAULT_TEMPLATES.get(template_name)
    if template:
        return PromptTemplate(template.template)
    else:
        logger.warning(f"Template '{template_name}' not found, using base_conversation")
        return PromptTemplate(DEFAULT_TEMPLATES["base_conversation"].template)