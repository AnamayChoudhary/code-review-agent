import asyncio
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

CHUNK_MARKER = "<<<CHUNK>>>"

async def main():
    mcp_url = "http://localhost:8000/mcp"
    async with streamablehttp_client(mcp_url, {}, timeout=120, terminate_on_close=False) as (read_stream, write_stream, _,):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool("analyze_git_repo", {
                "repo_url": "https://github.com/manning-lp/AnamayChoudhary-configure-observability-lp"
            })
            # graceful parsing of chunked response
            if isinstance(result, str) and CHUNK_MARKER in result:
                parts = result.split(CHUNK_MARKER)
                for i, part in enumerate(parts, start=1):
                    print(f"\n--- chunk {i}/{len(parts)} ---\n")
                    print(part)
            else:
                print(result)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("Client error:", e)