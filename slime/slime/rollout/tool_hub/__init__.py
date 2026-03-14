import asyncio
import json

import aiohttp
from aiohttp import ClientTimeout


def parse_arguments(json_value, force_dict=True):
    try:
        parsed_value = json.loads(json_value)
        return parsed_value, isinstance(parsed_value, dict) if force_dict else True
    except Exception:
        return json_value, False


def get_argument_type(func_name: str, arg_key: str, defined_tools: list):
    name2tool = {tool["name"]: tool for tool in defined_tools}
    if func_name not in name2tool:
        return None
    tool = name2tool[func_name]
    if arg_key not in tool["parameters"].get("properties", {}):
        return None
    if "type" in tool["parameters"]["properties"][arg_key]:
        return tool["parameters"]["properties"][arg_key]["type"]
    elif "anyOf" in tool["parameters"]["properties"][arg_key]:
        types = []
        for x in tool["parameters"]["properties"][arg_key]["anyOf"]:
            if "type" in x:
                types.append(x["type"])
        return types
    return "Any"


def check_tool_call(tool_call, defined_tools):
    func_name2arguments = {tool["name"]: tool["parameters"] for tool in defined_tools}
    func_name = tool_call["name"]
    arguments = tool_call["arguments"]

    if func_name not in func_name2arguments:
        return False, f'Undefined function "{func_name}". Only use function in {list(func_name2arguments.keys())}'
    defined_arguments = func_name2arguments[func_name].get("properties", {})
    required_arg_names = func_name2arguments[func_name].get("required", [])
    for arg_name in arguments:
        if arg_name not in defined_arguments:
            return False, f'Undefined argument "{arg_name}" for functinon "{func_name}". Allowed arguments: "{list(defined_arguments.keys())}"'
    for arg_name in required_arg_names:
        if arg_name not in arguments:
            return False, f'Missing required argument "{arg_name}" for functinon "{func_name}"'

    return True, None


def get_str_output(tool_call_output):
    if isinstance(tool_call_output, str):
        return tool_call_output
    try:
        return json.dumps(tool_call_output, ensure_ascii=False)
    except Exception:
        return str(tool_call_output)


remote_env_sem = None
remote_env_client = None
remote_env_max_proc = 2400


async def run_tool(args, tool_call, defined_tools=[], remote_env_info=None, need_check=True):
    assert tool_call["session_id"] is not None and tool_call["tool_call_id"] is not None, tool_call
    parsed_arguments, is_good_json = parse_arguments(tool_call["arguments"])
    assert is_good_json, tool_call
    tool_call["arguments"] = parsed_arguments
    tool_call["remote_env_info"] = remote_env_info

    is_valid = True
    tool_output = {
        "session_id": tool_call["session_id"],
        "tool_call_id": tool_call["tool_call_id"],
    }
    if need_check:
        is_valid, error_message = check_tool_call(tool_call, defined_tools)
        if not is_valid:
            tool_output["output"] = {"erorr": error_message}

    if is_valid:
        if remote_env_info is not None and "url" in remote_env_info:
            global remote_env_sem, remote_env_client
            if remote_env_sem is None:
                remote_env_sem = asyncio.Semaphore(remote_env_max_proc)
            if remote_env_client is None:
                connector = aiohttp.TCPConnector(limit=remote_env_max_proc + 100)
                remote_env_client = aiohttp.ClientSession(connector=connector)
            url = remote_env_info["url"]
            retries = 0
            timeout = ClientTimeout(total=args.tool_timeout)
            while retries < args.tool_max_retry:
                try:
                    async with remote_env_sem:
                        async with remote_env_client.post(url, json=tool_call, timeout=timeout) as resp:
                            try:
                                result = await resp.json()
                                tool_output.update(result)
                                break
                            except Exception:
                                tool_output["output"] = await resp.text()
                                break
                except Exception as e:
                    retries += 1
                    if retries == args.tool_max_retry:
                        tool_output["output"] = f"Connection error after {args.tool_max_retry} retries {type(e).__name__}: {str(e)}"
                        break
                    await asyncio.sleep(1)
        else:
            raise NotImplementedError("Only support remote tool for now")

    assert "output" in tool_output, tool_output
    tool_output["output"] = get_str_output(tool_output["output"])
    return tool_output

