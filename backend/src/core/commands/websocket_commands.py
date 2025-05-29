from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pydantic import BaseModel
from enum import Enum

class CommandType(str, Enum):
    """WebSocket command types"""
    PAUSE = "pause"
    RESUME = "resume"
    STOP = "stop"
    CLEAR_HISTORY = "clear_history"
    SET_CONFIG = "set_config"
    GET_STATUS = "get_status"
    SET_PROMPT = "set_prompt"  # New command type

class Command(ABC):
    """Base command interface"""
    
    @abstractmethod
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the command"""
        pass
    
    @abstractmethod
    def validate(self) -> bool:
        """Validate command parameters"""
        pass

class PauseCommand(Command):
    """Pause audio streaming command"""
    
    def __init__(self, reason: Optional[str] = None):
        self.reason = reason
        
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Pause audio streaming"""
        pipeline = context.get("pipeline")
        if pipeline:
            await pipeline.pause()
            
        return {
            "command": CommandType.PAUSE,
            "status": "paused",
            "reason": self.reason
        }
        
    def validate(self) -> bool:
        return True

class ResumeCommand(Command):
    """Resume audio streaming command"""
    
    def __init__(self, reason: Optional[str] = None):
        self.reason = reason
        
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Resume audio streaming"""
        pipeline = context.get("pipeline")
        if pipeline:
            await pipeline.resume()
            
        return {
            "command": CommandType.RESUME,
            "status": "resumed",
            "reason": self.reason
        }
        
    def validate(self) -> bool:
        return True

class StopCommand(Command):
    """Stop pipeline command"""
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Stop the pipeline"""
        pipeline = context.get("pipeline")
        if pipeline:
            await pipeline.stop()
            
        return {
            "command": CommandType.STOP,
            "status": "stopped"
        }
        
    def validate(self) -> bool:
        return True

class ClearHistoryCommand(Command):
    """Clear conversation history command"""
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Clear LLM conversation history"""
        pipeline = context.get("pipeline")
        if pipeline and pipeline.llm:
            pipeline.llm.clear_history()
            
        return {
            "command": CommandType.CLEAR_HISTORY,
            "status": "cleared",
            "type": "command_response"
        }
        
    def validate(self) -> bool:
        return True

class SetPromptCommand(Command):
    """Set system prompt command"""
    
    def __init__(self, prompt: str):
        self.prompt = prompt
        
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Set the system prompt for the LLM"""
        pipeline = context.get("pipeline")
        if pipeline and pipeline.llm:
            # Set the system prompt in the LLM provider
            pipeline.llm.set_system_prompt(self.prompt)
            
        return {
            "command": CommandType.SET_PROMPT,
            "status": "updated",
            "prompt": self.prompt,
            "type": "command_response"
        }
        
    def validate(self) -> bool:
        return isinstance(self.prompt, str) and len(self.prompt.strip()) > 0

class SetConfigCommand(Command):
    """Update configuration command"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Update configuration dynamically"""
        # Validate and apply config changes
        # This is a simplified example
        return {
            "command": CommandType.SET_CONFIG,
            "status": "updated",
            "config": self.config
        }
        
    def validate(self) -> bool:
        return isinstance(self.config, dict)

class GetStatusCommand(Command):
    """Get pipeline status command"""
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get current pipeline status"""
        pipeline = context.get("pipeline")
        
        status = {
            "command": CommandType.GET_STATUS,
            "is_processing": pipeline.is_processing if pipeline else False,
            "providers": {}
        }
        
        if pipeline:
            status["providers"] = {
                "stt": {"connected": pipeline.stt.is_connected},
                "llm": {
                    "history_length": len(pipeline.llm.conversation_history),
                    "system_prompt": getattr(pipeline.llm, '_system_prompt', 'Default prompt')
                },
                "tts": {"active": True}
            }
            
        return status
        
    def validate(self) -> bool:
        return True

class CommandFactory:
    """Factory for creating commands from JSON"""
    
    _command_map = {
        CommandType.PAUSE: PauseCommand,
        CommandType.RESUME: ResumeCommand,
        CommandType.STOP: StopCommand,
        CommandType.CLEAR_HISTORY: ClearHistoryCommand,
        CommandType.SET_CONFIG: SetConfigCommand,
        CommandType.GET_STATUS: GetStatusCommand,
        CommandType.SET_PROMPT: SetPromptCommand,  # Add new command
    }
    
    @classmethod
    def create(cls, command_data: Dict[str, Any]) -> Command:
        """Create command from JSON data"""
        command_type = command_data.get("command")
        
        if not command_type or command_type not in cls._command_map:
            raise ValueError(f"Unknown command type: {command_type}")
            
        command_class = cls._command_map[command_type]
        
        # Extract parameters
        params = {k: v for k, v in command_data.items() if k != "command"}
        
        # Create command instance
        if params:
            command = command_class(**params)
        else:
            command = command_class()
            
        if not command.validate():
            raise ValueError(f"Invalid command parameters for {command_type}")
            
        return command