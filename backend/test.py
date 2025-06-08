"""
Test script to verify websockets extra_headers support
"""
import asyncio
import websockets
import inspect

async def test_websockets():
    # Print websockets version
    print(f"Websockets version: {websockets.__version__}")
    
    # Check if connect supports extra_headers
    sig = inspect.signature(websockets.connect)
    print(f"\nwebsockets.connect signature: {sig}")
    
    # Check parameters
    params = sig.parameters
    if 'extra_headers' in params:
        print("\n✓ extra_headers parameter is supported")
    else:
        print("\n✗ extra_headers parameter is NOT supported")
        print("Available parameters:", list(params.keys()))
    
    # Try to create a connection with extra_headers (to a test server)
    try:
        # This will fail to connect but should accept the parameter
        async with websockets.connect(
            "wss://echo.websocket.org",
            extra_headers={"test": "header"}
        ) as ws:
            print("\n✓ Connection created with extra_headers")
    except TypeError as e:
        if "extra_headers" in str(e):
            print(f"\n✗ TypeError with extra_headers: {e}")
        else:
            print(f"\n✓ extra_headers accepted, connection failed for other reason: {e}")
    except Exception as e:
        print(f"\n✓ extra_headers accepted, connection failed for other reason: {e}")

if __name__ == "__main__":
    asyncio.run(test_websockets())