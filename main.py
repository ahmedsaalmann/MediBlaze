"""
🏥 MediBlaze FastAPI Backend
Advanced Medical AI Assistant with RAG and Web Search
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, AsyncGenerator
import logging
import os
from pathlib import Path
import markdown
from datetime import datetime
import json
import asyncio

from langchain_core.messages import HumanMessage
from agent.agent import agent

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="🏥 MediBlaze API",
    description="Advanced Medical AI Assistant with RAG and Web Search capabilities",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
# check_dir=False so the app doesn't crash on startup if the folder is empty/missing
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static", check_dir=False), name="static")

# Pydantic models
class ChatMessage(BaseModel):
    message: str
    timestamp: datetime = None

class ChatResponse(BaseModel):
    response: str
    response_html: str
    timestamp: datetime
    processing_time: float
    tools_used: List[str] = []

class HealthCheck(BaseModel):
    status: str
    timestamp: datetime
    version: str
    services: Dict[str, str]

# Store conversation history (in production, use proper session management)
conversation_history: Dict[str, List[Dict]] = {}

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """🏠 Serve the main MediBlaze interface"""
    try:
        html_file = Path("templates/index.html")
        if html_file.exists():
            # Read with UTF-8 encoding to handle special characters
            with open(html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Replace Flask template syntax with static paths
            html_content = html_content.replace(
                "{{ url_for('static', filename='styles.css') }}", 
                "/static/styles.css"
            )
            html_content = html_content.replace(
                "{{ url_for('static', filename='doctor-avatar.png') }}", 
                "/static/doctor-avatar.png"
            )
            html_content = html_content.replace(
                "{{ url_for('static', filename='user-avatar.png') }}", 
                "/static/user-avatar.png"
            )
            
            return HTMLResponse(content=html_content, status_code=200)
        else:
            return HTMLResponse(
                content="""
                <!DOCTYPE html>
                <html>
                    <head>
                        <title>🏥 MediBlaze</title>
                        <meta charset="UTF-8">
                        <style>
                            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
                            .container { max-width: 800px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                            h1 { color: #2c5aa0; }
                            .status { padding: 20px; background: #e8f5e8; border-radius: 5px; margin: 20px 0; }
                        </style>
                    </head>
                    <body>
                        <div class="container">
                            <h1>🏥 MediBlaze Medical Assistant</h1>
                            <div class="status">
                                <h2>✅ API Status: Running</h2>
                                <p>The MediBlaze API is running successfully!</p>
                                <ul>
                                    <li><strong>Chat API:</strong> <code>POST /chat</code></li>
                                    <li><strong>Health Check:</strong> <a href="/health">GET /health</a></li>
                                    <li><strong>API Documentation:</strong> <a href="/docs">GET /docs</a></li>
                                    <li><strong>ReDoc:</strong> <a href="/redoc">GET /redoc</a></li>
                                </ul>
                            </div>
                            <p><strong>Note:</strong> Frontend template not found. Place your HTML template in <code>templates/index.html</code></p>
                        </div>
                    </body>
                </html>
                """,
                status_code=200
            )
    except Exception as e:
        logger.error(f"❌ [MediBlaze API] Error serving root: {str(e)}")
        return HTMLResponse(
            content=f"""
            <!DOCTYPE html>
            <html>
                <head><title>🏥 MediBlaze - Error</title><meta charset="UTF-8"></head>
                <body style="font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5;">
                    <div style="max-width: 600px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px;">
                        <h1 style="color: #d32f2f;">⚠️ MediBlaze Error</h1>
                        <p>There was an error loading the application.</p>
                        <p><strong>Error:</strong> {str(e)}</p>
                        <p>Try accessing the <a href="/docs">API documentation</a> instead.</p>
                    </div>
                </body>
            </html>
            """,
            status_code=500
        )

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """🔍 Health check endpoint for MediBlaze services"""
    try:
        # Check if agent is available
        agent_status = "✅ Online" if agent else "❌ Offline"
        
        # Check environment variables
        env_status = "✅ Configured" if os.getenv("PINECONE_API_KEY") and os.getenv("GOOGLE_API_KEY") else "❌ Missing Keys"
        
        return HealthCheck(
            status="🟢 Healthy",
            timestamp=datetime.now(),
            version="1.0.0",
            services={
                "agent": agent_status,
                "environment": env_status,
                "api": "✅ Online"
            }
        )
    except Exception as e:
        logger.error(f"❌ [MediBlaze API] Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Health check failed")

@app.post("/chat", response_model=ChatResponse)
async def chat_with_mediblaze(message: ChatMessage):
    """
    💬 Main chat endpoint for MediBlaze medical assistance
    Processes medical queries using RAG and web search capabilities
    """
    start_time = datetime.now()
    tools_used = []
    
    try:
        logger.info(f"💬 [MediBlaze API] Processing message: {message.message[:100]}...")
        
        # Create a session ID (simple approach - in production use proper session management)
        session_id = "default_session"
        
        # Initialize conversation history if needed
        if session_id not in conversation_history:
            conversation_history[session_id] = []
        
        # Process the message with the agent
        response = agent.invoke({
            "messages": [HumanMessage(content=message.message)]
        })
        
        # Extract the final response
        if response and "messages" in response:
            final_message = response["messages"][-1]
            response_text = final_message.content
            
            # Check for tool usage in the conversation
            for msg in response["messages"]:
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        if tool_call["name"] not in tools_used:
                            tools_used.append(tool_call["name"])
        else:
            response_text = "I apologize, but I'm having trouble processing your request right now. Please try again."
        
        # Convert markdown to HTML for frontend display
        response_html = markdown.markdown(
            response_text,
            extensions=['extra', 'codehilite', 'toc']
        )
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Store in conversation history
        conversation_history[session_id].append({
            "user_message": message.message,
            "bot_response": response_text,
            "timestamp": start_time,
            "tools_used": tools_used
        })
        
        # Limit conversation history (keep last 10 exchanges)
        if len(conversation_history[session_id]) > 10:
            conversation_history[session_id] = conversation_history[session_id][-10:]
        
        logger.info(f"✅ [MediBlaze API] Response generated in {processing_time:.2f}s using tools: {tools_used}")
        
        return ChatResponse(
            response=response_text,
            response_html=response_html,
            timestamp=datetime.now(),
            processing_time=processing_time,
            tools_used=tools_used
        )
    
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        error_message = f"❌ I encountered an error while processing your medical query. Please try rephrasing your question or try again in a moment."
        
        logger.error(f"❌ [MediBlaze API] Error processing message: {str(e)}")
        
        return ChatResponse(
            response=error_message,
            response_html=f"<p>{error_message}</p>",
            timestamp=datetime.now(),
            processing_time=processing_time,
            tools_used=[]
        )

@app.post("/chat/stream")
async def stream_chat_with_mediblaze(message: ChatMessage):
    """
    💬 Streaming chat endpoint for real-time MediBlaze responses
    Streams response tokens as they are generated with tool usage indicators
    """
    async def generate_response() -> AsyncGenerator[str, None]:
        try:
            logger.info(f"💬 [MediBlaze Stream] Processing: {message.message[:100]}...")
            
            # Create a session ID
            session_id = "default_session"
            
            # Initialize conversation history if needed
            if session_id not in conversation_history:
                conversation_history[session_id] = []
            
            # Send initial status
            yield f"data: {json.dumps({'type': 'start', 'status': 'Processing your medical query...'})}\n\n"
            
            # Show tool indicators during processing
            yield f"data: {json.dumps({'type': 'tool_start', 'tool_name': 'rag_tool', 'message': '🤔Thinking'})}\n\n"
            await asyncio.sleep(0.5)  # Simulate search time
            
            # Process with agent
            response = agent.invoke({
                "messages": [HumanMessage(content=message.message)]
            })
            
            # Check if web search was used by examining tool calls
            tools_used = []
            if response and "messages" in response:
                for msg in response["messages"]:
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            tool_name = tool_call["name"]
                            if tool_name not in tools_used:
                                tools_used.append(tool_name)
                                if "web" in tool_name.lower():
                                    yield f"data: {json.dumps({'type': 'tool_start', 'tool_name': 'web_search', 'message': 'Searching web for latest medical information...'})}\n\n"
                                    await asyncio.sleep(1.0)  # Simulate web search time
            
            # End tool usage
            yield f"data: {json.dumps({'type': 'tool_end'})}\n\n"
              # Extract the response
            if response and "messages" in response:
                final_message = response["messages"][-1]
                response_text = final_message.content
                
                # Send response start
                yield f"data: {json.dumps({'type': 'response_start'})}\n\n"
                
                # Split response by sentences and meaningful chunks to preserve markdown structure
                # This ensures headers, lists, and formatting stay intact
                lines = response_text.split('\n')
                current_chunk = ""
                
                for line in lines:
                    current_chunk += line + '\n'
                    
                    # Send chunk if we have a complete line or it's getting long
                    if line.strip() == '' or len(current_chunk) > 100 or line.startswith('##') or line.startswith('**') or line.startswith('-'):
                        if current_chunk.strip():
                            yield f"data: {json.dumps({'type': 'content', 'content': current_chunk.strip()})}\n\n"
                            await asyncio.sleep(0.1)  # Slightly longer delay for readability
                        current_chunk = ""
                
                # Send any remaining content
                if current_chunk.strip():
                    yield f"data: {json.dumps({'type': 'content', 'content': current_chunk.strip()})}\n\n"
                
                # Send completion
                yield f"data: {json.dumps({'type': 'complete', 'tools_used': tools_used})}\n\n"
                
            else:
                error_msg = "I apologize, but I'm having trouble processing your request right now."
                yield f"data: {json.dumps({'type': 'error', 'content': error_msg})}\n\n"
                
        except Exception as e:
            logger.error(f"❌ [MediBlaze Stream] Error: {str(e)}")
            yield f"data: {json.dumps({'type': 'error', 'content': 'Sorry, there was an error processing your request.'})}\n\n"
        
        finally:
            yield f"data: {json.dumps({'type': 'end'})}\n\n"
    
    return StreamingResponse(
        generate_response(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        }
    )

@app.get("/conversation/{session_id}")
async def get_conversation_history(session_id: str = "default_session"):
    """📚 Get conversation history for a session"""
    try:
        if session_id not in conversation_history:
            return JSONResponse(content={"history": []})
        
        return JSONResponse(content={
            "history": conversation_history[session_id],
            "total_messages": len(conversation_history[session_id])
        })
    
    except Exception as e:
        logger.error(f"❌ [MediBlaze API] Error retrieving conversation: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving conversation history")

@app.delete("/conversation/{session_id}")
async def clear_conversation_history(session_id: str = "default_session"):
    """🗑️ Clear conversation history for a session"""
    try:
        if session_id in conversation_history:
            conversation_history[session_id] = []
        
        return JSONResponse(content={"message": f"Conversation history cleared for session: {session_id}"})
    
    except Exception as e:
        logger.error(f"❌ [MediBlaze API] Error clearing conversation: {str(e)}")
        raise HTTPException(status_code=500, detail="Error clearing conversation history")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"message": "🔍 Resource not found", "detail": "The requested resource was not found on this server."}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"message": "⚠️ Internal server error", "detail": "An internal error occurred. Please try again later."}
    )

if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 [MediBlaze API] Starting MediBlaze FastAPI server...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )