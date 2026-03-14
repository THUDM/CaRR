import json
import re
import uuid

from slime.rollout.tool_hub import get_argument_type

from .base_message_processor import MessageProcessor


class Qwen3MessageProcessor(MessageProcessor):
    def __init__(self, tokenizer, drop_previous_turn_thinking=True, **kwargs):
        self.tokenizer = tokenizer
        self.drop_previous_turn_thinking = drop_previous_turn_thinking

    def apply_chat_template(
        self,
        messages: list,
        tools: list = None,
        stop_token: str = "\n<|im_start|>assistant\n<think>\n",
        add_special_tokens: bool = True,
    ):
        prompt = ""

        if tools:
            system_prompt = None
            if messages[0]["role"] == "system":
                system_prompt = messages[0]["content"]
            prompt += "<|im_start|>system\n"
            if system_prompt:
                prompt += system_prompt + "\n\n"

            prompt += (
                "# Tools\n\n"
                "You may call one or more functions to assist with the user query.\n\n"
                "You are provided with function signatures within <tools></tools> XML tags:\n"
                "<tools>"
            )
            for tool in tools:
                prompt += "\n" + json.dumps(tool, ensure_ascii=False)
            prompt += (
                "\n</tools>\n\n"
                "For each function call, return a json object with function name and arguments "
                "within <tool_call></tool_call> XML tags:\n"
                "<tool_call>\n"
                '{"name": <function-name>, "arguments": <args-json-object>}\n'
                "</tool_call><|im_end|>\n"
            )

        last_turn_begin_idx = -1
        truncated = False
        aborted = False
        for idx, item in enumerate(messages):
            if item["role"] == "user":
                last_turn_begin_idx = idx
        for idx, item in enumerate(messages):
            role = item["role"]
            if role == "system" and idx == 0:
                continue

            if role == "system" or role == "user":
                prompt += f"<|im_start|>{role}\n{item['content']}<|im_end|>\n"

            elif role == "assistant":
                prompt += "<|im_start|>assistant\n<think>\n"
                if "truncated_content" in item:
                    prompt += item["truncated_content"]
                    truncated = True
                elif "aborted_content" in item:
                    prompt += item["aborted_content"]
                    aborted = True
                elif "ill_formed_content" in item:
                    prompt += item["ill_formed_content"] + "\n"
                else:
                    if "reasoning_content" in item:
                        if not self.drop_previous_turn_thinking or idx > last_turn_begin_idx:
                            prompt += item["reasoning_content"] + "\n</think>\n\n"
                    if "content" in item:
                        prompt += item["content"]
                    if "tool_calls" in item:
                        for tc_idx, tool_call in enumerate(item["tool_calls"]):
                            if (tc_idx == 0 and "content" in item) or tc_idx > 0:
                                prompt += "\n"
                            func_name = tool_call["name"]
                            arguments = json.loads(tool_call["arguments"])
                            prompt += (
                                "<tool_call>\n"
                                f'{{"name": "{func_name}", "arguments": {json.dumps(arguments, ensure_ascii=False)}}}\n'
                                "</tool_call>"
                            )
                    prompt += "<|im_end|>\n"

            elif role == "tool":
                prompt += "<|im_start|>user"
                for tool_response in item["content"]:
                    prompt += "\n<tool_response>\n" + tool_response["output"] + "\n</tool_response>"
                prompt += "<|im_end|>\n"

        if not truncated and not aborted:
            if prompt.endswith("<|im_end|>\n") or prompt.endswith("<|endoftext|>\n"):
                prompt = prompt.removesuffix("\n")
            if stop_token == "\n<|im_start|>assistant\n<think>\n":
                prompt += stop_token
        if not add_special_tokens:
            prompt = "\n" + prompt
        return prompt

    def parse_model_response(self, response: str, tools: list = [], truncated: bool = False, aborted: bool = False, session_id: str = None):
        raw_response = response
        stop_token = None
        if not truncated and not aborted:
            for special_token in ["<|im_end|>", "<|endoftext|>"]:
                if response.endswith(special_token):
                    stop_token = special_token
                    response = response.removesuffix(stop_token)
                    break
            assert stop_token is not None, f"NOT FOUND STOP TOKEN: {response}"

        text = response
        func_name2arguments = {tool["name"]: tool["parameters"] for tool in tools}
        reasoning_content = None
        content = None
        tool_calls = []
        ill_formed_reasons = []
        finish_turn = False
        if truncated:
            message = {"role": "assistant", "truncated_content": text}
            return [message], stop_token, False, finish_turn
        if aborted:
            message = {"role": "assistant", "aborted_content": text}
            return [message], stop_token, False, finish_turn

        if "\n</think>\n\n" not in text or text.count("\n</think>\n\n") != 1:
            ill_formed_reasons.append("Wrong thinking format.")

        if "</think>" in text:
            reasoning_content, text = text.rsplit("</think>", 1)
            reasoning_content = reasoning_content.removesuffix("\n")
            text = text.removeprefix("\n\n")
        else:
            reasoning_content = text
            text = ""

        if tools and "<tool_call>" in text:
            index = text.find("<tool_call>")
            content_str = text[:index]
            text = text[index:]
        else:
            content_str = text
            text = ""
        if content_str != "":
            content = content_str.removesuffix("\n")

        tool_call_strs = re.findall(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
        for call in tool_call_strs:
            ill_formed_fc_reasons = []
            try:
                raw_tool_call = json.loads(call.strip())
            except Exception:
                raw_tool_call = {}
            func_name = raw_tool_call.get("name", None)
            arguments = raw_tool_call.get("arguments", None)
            if not isinstance(func_name, str) or not isinstance(arguments, dict):
                ill_formed_fc_reasons.append(
                    "Error function call format. "
                    "For each function call, return a json object with function name and arguments in the following format:\n"
                    '{"name": <function-name>, "arguments": <args-json-object>}\n'
                )
            else:
                if func_name not in func_name2arguments:
                    ill_formed_fc_reasons.append(
                        f'Undefined function "{func_name}". Only use function in {list(func_name2arguments.keys())}'
                    )

                if func_name in func_name2arguments:
                    for arg_key in arguments:
                        arg_type = get_argument_type(func_name, arg_key, defined_tools=tools)
                        if not arg_type:
                            ill_formed_fc_reasons.append(
                                f'Undefined argument "{arg_key}" for functinon "{func_name}".'
                            )
                    required_arg_names = func_name2arguments[func_name].get("required", [])
                    for arg_name in required_arg_names:
                        if arg_name not in arguments:
                            ill_formed_fc_reasons.append(
                                f'Missing required argument "{arg_name}" for functinon "{func_name}"'
                            )

            tool_call = {
                "session_id": session_id,
                "tool_call_id": "tool-call-" + str(uuid.uuid4()),
                "name": func_name,
                "arguments": json.dumps(arguments, ensure_ascii=False),
            }
            if ill_formed_fc_reasons:
                tool_call["ill_formed_reasons"] = ill_formed_fc_reasons
                ill_formed_reasons.extend(ill_formed_fc_reasons)
            tool_calls.append(tool_call)

        message = {"role": "assistant"}
        if reasoning_content is not None:
            message["reasoning_content"] = reasoning_content
        if content is not None:
            message["content"] = content
        if len(tool_calls) > 0:
            message["tool_calls"] = tool_calls
        if reasoning_content is None:
            ill_formed_reasons.append("Missing reasoning_content.")
        if content is None and len(tool_calls) == 0:
            ill_formed_reasons.append("Missing response or tool_calls.")

        if not ill_formed_reasons:
            reconstructed_response = self.apply_chat_template([message], stop_token="").removeprefix(
                "<|im_start|>assistant\n<think>\n"
            )
            if reconstructed_response != raw_response:
                ill_formed_reasons.append(
                    f"Response unmatched:\nSGLANG_RESPONSE:\n{raw_response}**END**\n\nMESSAGE_RPOCESSOR_RESPONSE:\n{reconstructed_response}**END**"
                )

        if ill_formed_reasons:
            message.update({"ill_formed_content": raw_response, "ill_formed_reasons": ill_formed_reasons})

        if stop_token == "<|endoftext|>":
            ill_formed_reasons.append(f"Error stop token: {stop_token}")

        if len(tools) == 0 or len(tool_calls) == 0 or stop_token == "<|endoftext|>":
            finish_turn = True

        return [message], stop_token, len(ill_formed_reasons) > 0, finish_turn
