from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from OpenAIService import OpenAIService

app = FastAPI()
openai_service = OpenAIService()

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    model: Optional[str] = "gpt-4o"

@app.post("/api/chat")
async def chat(request: ChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages are required")

    try:
        model_context_length = 128000
        max_output_tokens = 50
        input_tokens = await openai_service.count_tokens(request.messages, request.model)

        if input_tokens + max_output_tokens > model_context_length:
            raise HTTPException(
                status_code=400,
                detail=f"No space left for response. Input tokens: {input_tokens}, Context length: {model_context_length}"
            )

        print(f"Input tokens: {input_tokens}, Max tokens: {max_output_tokens}, "
              f"Model context length: {model_context_length}, "
              f"Tokens left: {model_context_length - (input_tokens + max_output_tokens)}")

        full_response = await openai_service.continuous_completion({
            "messages": request.messages,
            "model": request.model,
            "max_tokens": max_output_tokens
        })

        return {
            "role": "assistant",
            "content": full_response
        }

    except Exception as error:
        print('Error:', str(error))
        raise HTTPException(status_code=500, detail="An error occurred while processing your request")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000) 