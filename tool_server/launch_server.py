import argparse
import asyncio
from functools import wraps
import time
from quart import Quart, jsonify, request
from collections import defaultdict
from web_search import search, parse_url, find

parser = argparse.ArgumentParser()
parser.add_argument("--serp_api_key", type=str, default="--", help="Serp API key")
parser.add_argument("--jina_api_key", type=str, default=None, help="Jina API key")
parser.add_argument("--http_proxy", type=str, default=None)
parser.add_argument("--port", type=int, default=7230)
args = parser.parse_args()

app = Quart(__name__)
session2sandbox = defaultdict(dict)

def log_tool_call_every_15_seconds(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        stop_event = asyncio.Event()
        start_time = time.time()
        
        tool_call = await request.get_json()

        async def log_task():
            while not stop_event.is_set():
                elapsed_time = time.time() - start_time
                print(f"[Periodic Log] tool_call: {tool_call}, runtime: {elapsed_time:.2f} seconds")
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=15)
                except asyncio.TimeoutError:
                    continue

        initial_elapsed_time = time.time() - start_time
        print(f"[Initial Log] tool_call: {tool_call}, runtime: {initial_elapsed_time:.2f} seconds")

        task = asyncio.create_task(log_task())
        try:
            result = await func(tool_call=tool_call, *args, **kwargs)
        finally:
            stop_event.set()
            await task

            final_elapsed_time = time.time() - start_time
            print(f"[Final Log] tool_call: {tool_call}, runtime: {final_elapsed_time:.2f} seconds")

        return result
    return wrapper

@app.route("/", methods=["POST"])
@log_tool_call_every_15_seconds
async def call_tool(tool_call):
    
    session_id, func_name, arguments, remote_env_info = (
        tool_call["session_id"],
        tool_call["name"],
        tool_call["arguments"],
        tool_call["remote_env_info"],
    )
    sandbox = session2sandbox[session_id]
    
    if func_name == "start_session":
        result = "Sucess start session"

    elif func_name == "close_session":
        if session_id in session2sandbox:
            del session2sandbox[session_id]  # Remove sandbox
        result = f"Sucessully close session {session_id}"
        
    elif func_name == "browser.search":
        query = arguments.get("query", "")
        num = arguments.get("num", 10)
        forbidden_strs = remote_env_info.get("search_forbidden_strs", [])
        if not query:
            result = "No query provided for search."
        else:
            result, idx2url = await search(query, num=num, forbidden_strs=forbidden_strs, proxy=args.http_proxy, serp_api_key=args.serp_api_key)
            sandbox['idx2url'] = idx2url
            if not result:
                result = "No results found."

    elif func_name == "browser.open":
        idx2url = sandbox.get("idx2url", {})
        idx = arguments.get("id", None)
        forbidden_strs = remote_env_info.get("search_forbidden_strs", [])
        if idx is None or (isinstance(idx, int) and idx not in idx2url):
            result = "Must provide a valid id(>=0 or a string) for opening. Other arguments are unvalid for now."
        else:
            url = idx2url.get(idx) if isinstance(idx, int) else idx
            if not url:
                result = "No URL found for the given id."
            else:
                result = await parse_url(url=url, forbidden_strs=forbidden_strs, proxy=args.http_proxy, jina_api_key=args.jina_api_key)
                if not result:
                    result = "Failed to fetch URL content."
                else:
                    sandbox["cur_web_content"] = result
        result = result[:10000]
            
    elif func_name == "browser.find":
        pattern = arguments.get("pattern", "")
        cur_web_content = sandbox.get("cur_web_content", "")
        if not pattern:
            result = "No pattern provided for finding."
        elif not cur_web_content:
            result = ""
        else:
            find_results = find(
                pattern=pattern,
                parse_content=cur_web_content,
                max_results=20,
                context_length=200,
                word_overlap_threshold=0.8,
            )
            result = "\n\n".join(
                [f"[Match {i+1}]\n{ctx}" for i, ctx in enumerate(find_results)]
            )
    else:
        result = f"Undefined function: {func_name}"

    return jsonify({"output": result})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=args.port, debug=False, use_reloader=False)