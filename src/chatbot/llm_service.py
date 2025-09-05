from typing import List, Dict
from .llm_client import LLMClientFactory
from configs import LLMConfig


class LLMService:
    def __init__(self):
        self.client = LLMClientFactory.create_llm_client()
    
    async def generate_response(
        self, 
        query: str, 
        context: str = "", 
        chat_history: List[Dict[str, str]] = None
    ) -> str:
        """Tạo response từ LLM với context và chat history"""
        try:
            messages = self._build_messages(query, context, chat_history)
            response = await self.client.chat_completion(messages)
            
            return response["choices"][0]["message"]["content"]
            
        except Exception as e:
            print(f"Error generating response: {e}")
            raise
    
    async def generate_stream_response(
        self, 
        query: str, 
        context: str = "", 
        chat_history: List[Dict[str, str]] = None
    ):
        """Tạo streaming response từ LLM"""
        try:
            messages = self._build_messages(query, context, chat_history)
            async for chunk in await self.client.chat_completion(messages, stream=True):
                yield chunk
                
        except Exception as e:
            print(f"Error generating stream response: {e}")
            raise
    
    def _build_messages(
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
    
    async def close(self):
        """Đóng service"""
        await self.client.close()