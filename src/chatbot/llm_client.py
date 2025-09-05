from typing import Dict, Any, List, AsyncGenerator
import httpx
import json
from configs import LLMConfig
import os
from dotenv import load_dotenv
from abc import ABC, abstractmethod
from configs import get_config


load_dotenv()


class LLMClient(ABC):
    def __init__(self):
        self.config: LLMConfig = get_config().get_llm_config()
    
    @abstractmethod
    async def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        stream: bool = False
    ) -> Dict[str, Any]:
        """Gọi API để tạo chat completion"""
        pass
    
    @abstractmethod
    async def _stream_completion(self, payload: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Xử lý streaming response"""
        pass
    
    @abstractmethod
    async def close(self):
        """Đóng client connection"""
        pass


class GeminiClient(LLMClient):
    def __init__(self):
        super().__init__()
        # Gemini-specific initialization if needed
        self.api_key = os.getenv('GOOGLE_API_KEY_GEMINI', None)
        self.client = httpx.AsyncClient(
            timeout=self.config.timeout,
            headers={
                "X-goog-api-key": self.api_key,
                "Content-Type": "application/json"
            }
        )

    async def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        stream: bool = False
    ) -> Dict[str, Any]:
        """Gọi API để tạo chat completion"""
        try:
            # Convert OpenAI format messages to Gemini format
            gemini_payload = self._convert_to_gemini_format(messages)
            
            if stream:
                return await self._stream_completion(gemini_payload)
            else:
                url = f"{self.config.base_url}/v1beta/models/{self.config.model_name}:generateContent"
                response = await self.client.post(url, json=gemini_payload)
                response.raise_for_status()
                
                # Convert Gemini response back to OpenAI format
                gemini_response = response.json()
                return self._convert_gemini_response(gemini_response)
                
        except httpx.HTTPError as e:
            print(f"HTTP error occurred: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise
    
    async def _stream_completion(self, payload: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Xử lý streaming response cho Gemini"""
        url = f"{self.config.base_url}/v1beta/models/{self.config.model_name}:streamGenerateContent?key={self.api_key}"
        
        async with self.client.stream("POST", url, json=payload) as response:
            response.raise_for_status()
            
            async for line in response.aiter_lines():
                if line.strip():
                    try:
                        # Remove "data: " prefix if present
                        if line.startswith("data: "):
                            line = line[6:]
                        
                        chunk = json.loads(line)
                        
                        # Extract text from Gemini streaming response
                        if "candidates" in chunk and len(chunk["candidates"]) > 0:
                            candidate = chunk["candidates"][0]
                            if "content" in candidate and "parts" in candidate["content"]:
                                for part in candidate["content"]["parts"]:
                                    if "text" in part:
                                        yield part["text"]
                                        
                    except json.JSONDecodeError:
                        continue
                    except (KeyError, IndexError):
                        continue

    def _convert_gemini_response(self, gemini_response: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Gemini response to OpenAI format"""
        try:
            content = gemini_response["candidates"][0]["content"]["parts"][0]["text"]
            return {
                "choices": [{
                    "message": {
                        "content": content,
                        "role": "assistant"
                    },
                    "finish_reason": "complete"
                }]
            }
        except (KeyError, IndexError) as e:
            print(f"Error parsing Gemini response: {e}")
            return {
                "choices": [{
                    "message": {
                        "content": "Sorry, I couldn't generate a response.",
                        "role": "assistant"
                    },
                    "finish_reason": "error"
                }]
            }
        
    def _convert_to_gemini_format(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Convert OpenAI messages format to Gemini format"""
        contents = []
        system_instruction = None
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                system_instruction = {"parts": [{"text": content}]}
            elif role == "user":
                contents.append({
                    "role": "user",
                    "parts": [{"text": content}]
                })
            elif role == "assistant":
                contents.append({
                    "role": "model",
                    "parts": [{"text": content}]
                })
        
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.config.temperature,
                "maxOutputTokens": self.config.max_tokens,
            }
        }
        
        if system_instruction:
            payload["systemInstruction"] = system_instruction
            
        return payload
    
    async def close(self):
        """Đóng client connection"""
        await self.client.aclose()


class LLMClientFactory:
    @staticmethod
    def create_llm_client() -> LLMClient:
        config: LLMConfig = get_config().get_llm_config()
        
        if config.provider == "google":
            return GeminiClient()
        else:
            raise ValueError(f"Unsupported LLM provider: {config.provider}")