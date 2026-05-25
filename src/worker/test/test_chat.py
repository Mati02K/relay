"""
Interactive chat test with the llama model
Verify the server is working by having a real conversation
"""
import asyncio
from pathlib import Path
import sys
import os

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from worker.inference.llamacpp import LlamaCppEngine


async def chat_with_model() -> bool:
    """Interactive chat with the model"""
    
    print("\n" + "="*70)
    print("INTERACTIVE CHAT TEST - Gemma Model")
    print("="*70 + "\n")
    
    # ========================================================================
    # Your paths
    # ========================================================================
    
    model_path = os.getenv("LLAMA_MODEL_PATH")
    llama_server_path = os.getenv("LLAMA_SERVER_BIN")
    if not model_path or not llama_server_path:
        print(" Error: LLAMA_MODEL_PATH or LLAMA_SERVER_BIN is not set")
        return False
    
    print(f"Model: {model_path}")
    print(f"Server: {llama_server_path}\n")
    
    # Initialize engine
    try:
        engine = LlamaCppEngine(
        model_path=model_path,
        server_binary=llama_server_path,
        port=9081,
        n_gpu_layers=-1,
        n_ctx=4096,
        n_threads=8,
        verbose=False,
        )
        print("Engine initialized\n")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return False
    
    try:
        # Start server
        print("Starting server...")
        await engine.start()
        print("Server started\n")
        
        # Check health
        health = await engine.health()
        if not health.status:
            print(f"Server not healthy: {health.detail}")
            return False
        
        print("Server is healthy\n")
        print("="*70)
        print("Chat with the Model (type 'quit' to exit)")
        print("="*70 + "\n")
        
        # Chat loop
        conversation_history = []
        
        while True:
            # Get user input
            user_message = input("\nYou: ").strip()
            
            if user_message.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if not user_message:
                print("(empty message, try again)")
                continue
            
            # Add to history
            conversation_history.append({
                "role": "user",
                "content": user_message
            })
            
            # Create request
            request = {
                "messages": conversation_history,
                "max_tokens": 256,
                "temperature": 0.7,
                "top_p": 0.9,
                "model": "gemma-3-4b-it",
            }
            
            print("\nModel: ", end="", flush=True)
            
            # Stream response
            full_response = ""
            async for line in engine.generate(request):
                # Parse SSE format
                if line.startswith("data: "):
                    import json
                    try:
                        payload = line.removeprefix("data: ").strip()
                        if payload and payload != "[DONE]":
                            obj = json.loads(payload)
                            choices = obj.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    print(content, end="", flush=True)
                                    full_response += content
                    except json.JSONDecodeError:
                        pass
            
            print()  # New line after response
            
            # Add to history
            if full_response:
                conversation_history.append({
                    "role": "assistant",
                    "content": full_response
                })
            
            # Show telemetry
            telemetry = await engine.get_engine_telemetry()
            print(f"\nEngine telemetry: Queue={telemetry.qw}, Memory={telemetry.mw:.1%}")
        
        return True
        
    except KeyboardInterrupt:
        print("\n\nChat interrupted")
        return False
    except Exception as e:
        print(f"\nError: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        print("\nStopping server...")
        await engine.stop()
        print("Server stopped")


if __name__ == "__main__":
    success = asyncio.run(chat_with_model())
    exit(0 if success else 1)
