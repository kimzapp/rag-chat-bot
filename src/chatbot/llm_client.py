from typing import Dict, Any, List, Generator, AsyncGenerator, Union
from configs import LLMConfig
import os
from dotenv import load_dotenv
from abc import ABC, abstractmethod
from configs import get_config
import google.generativeai as genai


load_dotenv()


class LLMClient(ABC):
    def __init__(self):
        self.config: LLMConfig = get_config().get_llm_config()
    
    @abstractmethod
    def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        stream: bool = False
    ) -> Union[Dict[str, Any], Generator[str, None, None]]:
        """Gọi API để tạo chat completion"""
        pass

    @abstractmethod
    async def async_chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        stream: bool = False
    ) -> Union[Dict[str, Any], AsyncGenerator[str, None]]:
        """Gọi API để tạo chat completion"""
        pass
    
    @abstractmethod
    def _stream_completion(self, payload: Dict[str, Any]) -> Generator[str, None, None]:
        """Xử lý streaming response"""
        pass

    @abstractmethod
    async def _async_stream_completion(self, payload: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Xử lý streaming response"""
        pass
    
    @abstractmethod
    async def close(self):
        """Đóng client connection"""
        pass

    def build_messages(
        self, 
        query: str, 
        context: str = "", 
        chat_history: List[Dict[str, str]] = None
    ) -> List[Dict[str, str]]:
        """Xây dựng messages cho API call"""
        messages = []

        # System message với context
        system_content = "Bạn là một AI assistant hữu ích."
        if context:
            system_content += f"\n\nContext thông tin:\n{context}"
        
        messages.append({
            "role": "system",
            "content": system_content
        })
        
        # Thêm chat history
        if chat_history:
            messages.extend(chat_history)
        
        # Thêm query hiện tại
        messages.append({
            "role": "user",
            "content": query
        })
        
        return messages


class GeminiClient(LLMClient):
    def __init__(self):
        super().__init__()
        # Gemini-specific initialization if needed
        api_key = os.getenv('GOOGLE_API_KEY_GEMINI', None)
        genai.configure(api_key=api_key)

        # initialize model
        self.client = genai.GenerativeModel(
            model_name=self.config.model_name,
            generation_config=genai.GenerationConfig(
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_tokens,
            )
        )

    def chat_completion(
        self,
        query: str,
        context: str = "",
        chat_history: List[Dict[str, str]] = None,
        stream: bool = False
    ) -> Union[Dict[str, Any], Generator[str, None, None]]:
        """Gọi API để tạo chat completion"""
        try:
            messages = self.build_messages(query, context, chat_history)
            # Convert OpenAI format messages to Gemini format
            prompt = self._convert_messages_to_prompt(messages)
            
            if stream:
                return self._stream_completion(prompt)
            else:
                response = self.client.generate_content(prompt)
                return self._convert_to_openai_format(response)
        except Exception as e:
            print(f"Error in chat completion: {e}")
            raise

    async def async_chat_completion(
        self, 
        query: str,
        context: str = "",
        chat_history: List[Dict[str, str]] = None,
        stream: bool = False
    ) -> Dict[str, AsyncGenerator[str, None]]:
        """Gọi API để tạo chat completion"""
        try:
            # Convert OpenAI format messages to Gemini format
            messages = self.build_messages(query, context, chat_history)
            prompt = self._convert_messages_to_prompt(messages)
            
            if stream:
                return self._async_stream_completion(prompt)
            else:
                response = await self.client.generate_content_async(prompt)
                return self._convert_to_openai_format(response)
            
        except Exception as e:
            print(f"Error in chat completion: {e}")
            raise
    
    def _stream_completion(self, prompt: Dict[str, Any]) -> Generator[str, None, None]:
        """Xử lý streaming response cho Gemini"""
        try:
            response = self.client.generate_content(
                prompt,
                stream=True
            )
            
            for chunk in response:
                if chunk.text:
                    yield chunk.text
                    
        except Exception as e:
            print(f"Error in streaming: {e}")
            raise

    async def _async_stream_completion(self, prompt: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Xử lý streaming response cho Gemini"""
        try:
            response = await self.client.generate_content_async(
                prompt, 
                stream=True
            )
            
            async for chunk in response:
                if chunk.text:
                    yield chunk.text
                    
        except Exception as e:
            print(f"Error in streaming: {e}")
            raise

    def _convert_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert OpenAI messages format to simple prompt"""
        prompt_parts = []
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        return "\n\n".join(prompt_parts)
    
    def _convert_to_openai_format(self, response) -> Dict[str, Any]:
        """Convert Gemini response to OpenAI format"""
        try:
            return {
                "choices": [{
                    "message": {
                        "content": response.text,
                        "role": "assistant"
                    },
                    "finish_reason": "stop"
                }]
            }
        except Exception as e:
            print(f"Error converting response: {e}")
            return {
                "choices": [{
                    "message": {
                        "content": "Sorry, I couldn't generate a response.",
                        "role": "assistant"
                    },
                    "finish_reason": "error"
                }]
            }

    async def close(self):
        """Đóng client connection"""
        pass


class LLMClientFactory:
    @staticmethod
    def create_llm_client() -> LLMClient:
        config: LLMConfig = get_config().get_llm_config()
        
        if config.provider == "google":
            return GeminiClient()
        else:
            raise ValueError(f"Unsupported LLM provider: {config.provider}")