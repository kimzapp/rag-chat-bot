from typing import Dict, List
from ollama import chat, ChatResponse
from .chat_history import ChatHistory


class OllamaClient:
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    def answer(self, query: str, chat_history: ChatHistory=None, stream=False, think=None) -> ChatResponse:
        messages = self._build_messages(query=query, chat_history=chat_history)
        return chat(model=self.model_name, messages=messages, stream=stream, think=think)    

    def _build_messages(
        self, 
        query: str, 
        context: str="", 
        chat_history: ChatHistory=None
    ) -> List[Dict[str, str]]:
        """Xây dựng message từ chat history và context bổ sung"""
        messages = []

        # System message với context
        if context:
            system_content += f"\n\nContext thông tin:\n{context}"
        
        # Thêm chat history
        if chat_history:
            messages.extend(chat_history)
        
        # Thêm query hiện tại
        messages.append({
            "role": "user",
            "content": query
        })
        
        return messages
