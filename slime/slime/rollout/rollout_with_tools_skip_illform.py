import json
import uuid
from copy import deepcopy

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

from .tool_hub import run_tool


async def generate_with_tool(args, sample: Sample, sampling_params, evaluation=False):
    generate_state = GenerateState(args)
    source = sample.source
    metadata = sample.metadata
    session_id = metadata["session_id"]
    initial_messages = metadata["input_messages"]
    tools = sample.tools
    message_processor = sample.message_processor
    tokenizer = message_processor.tokenizer
    remote_env_info = sample.metadata["remote_env_info"]
    max_env_interact_steps = remote_env_info.get("max_steps", None) if remote_env_info else None
    max_context_length = args.rollout_max_context_len if not evaluation else args.eval_max_context_len
    max_new_tokens = args.rollout_max_response_len if not evaluation else args.eval_max_response_len
    initial_prompt = sample.prompt
    initial_prompt_token_ids = tokenizer.encode(initial_prompt, add_special_tokens=False)
    initial_prompt_length = len(initial_prompt_token_ids)

    if args.partial_rollout and sample.status == Sample.Status.ABORTED and len(sample.response) > 0:
        messages = sample.metadata["history"]
        prompt = sample.prompt + sample.response
        prompt_token_ids = sample.tokens
        prompt_length = len(prompt_token_ids)
        response_loss_mask = sample.loss_mask
        rollout_log_probs = sample.rollout_log_probs
        env_interact_steps = sample.metadata["env_interact_steps"]
        ill_formed = sample.metadata["ill_formed"]
        use_tool = sample.metadata["ill_formed"]
        reward = sample.reward

        prompt_by_mp = message_processor.apply_chat_template(messages, tools)
        assert prompt_by_mp == prompt
    else:
        if remote_env_info and "url" in remote_env_info:
            tool_call = {
                "session_id": session_id,
                "tool_call_id": "tool-call-" + str(uuid.uuid4()),
                "name": "start_session",
                "arguments": json.dumps({}),
            }
            start_session_info = await run_tool(
                args, tool_call=tool_call, defined_tools=tools, remote_env_info=remote_env_info, need_check=False
            )
            if "messages" in start_session_info:
                initial_messages.extend(start_session_info["messages"])
                sample.prompt = message_processor.apply_chat_template(initial_messages, tools=tools)

        messages = deepcopy(initial_messages)
        prompt = initial_prompt
        prompt_token_ids = initial_prompt_token_ids
        prompt_length = initial_prompt_length
        response_loss_mask = []
        rollout_log_probs = []
        env_interact_steps = 0
        ill_formed = False
        use_tool = False
        reward = None

    while True:
        current_sampling_params = deepcopy(sampling_params)
        current_sampling_params["max_new_tokens"] = min(max_new_tokens, max_context_length - prompt_length)

        sglang_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
        payload = {
            "input_ids": prompt_token_ids,
            "sampling_params": current_sampling_params,
            "return_logprob": True,
        }

        sglang_output = await post(sglang_url, payload)

        if "output_token_logprobs" in sglang_output["meta_info"]:
            try:
                response_token_ids = [item[1] for item in sglang_output["meta_info"]["output_token_logprobs"]]
                response_token_logprobs = [item[0] for item in sglang_output["meta_info"]["output_token_logprobs"]]
            except Exception:
                response_token_ids = []
                response_token_logprobs = []
        else:
            response_token_ids = []
            response_token_logprobs = []
        response = sglang_output["text"]
        truncated = sglang_output["meta_info"]["finish_reason"]["type"] == "length"
        aborted = sglang_output["meta_info"]["finish_reason"]["type"] == "abort"
        if args.partial_rollout and messages and "aborted_content" in messages[-1]:
            complete_response = messages[-1]["aborted_content"] + response
            assistant_messages, stop_token, is_ill_formed, finish_turn = message_processor.parse_model_response(
                complete_response, tools=tools, truncated=truncated, aborted=aborted, session_id=session_id
            )
            messages = messages[:-1] + assistant_messages
        else:
            complete_response = response
            assistant_messages, stop_token, is_ill_formed, finish_turn = message_processor.parse_model_response(
                complete_response, tools=tools, truncated=truncated, aborted=aborted, session_id=session_id
            )
            messages.extend(assistant_messages)
        if is_ill_formed:
            ill_formed = True

        next_prompt = prompt + response
        rollout_log_probs += response_token_logprobs
        next_prompt_token_ids = prompt_token_ids + response_token_ids
        next_prompt_by_mp = message_processor.apply_chat_template(messages, tools=tools, stop_token=stop_token)
        next_prompt_length = len(next_prompt_token_ids)
        response_length = next_prompt_length - prompt_length

        assert next_prompt_by_mp == next_prompt
        assert next_prompt_by_mp[len(prompt) :] == response

        response_loss_mask += [1] * response_length
        prompt, prompt_token_ids, prompt_length = next_prompt, next_prompt_token_ids, next_prompt_length
        if truncated or aborted or finish_turn:
            break
        if args.stop_once_illform and ill_formed and not evaluation:
            break

        tool_calls = sum([message.get("tool_calls", []) for message in assistant_messages], [])
        env_interact_steps += 1
        use_tool = True
        if max_env_interact_steps is not None and env_interact_steps > max_env_interact_steps:
            break
        tool_outputs = []
        if tool_calls:
            for tool_call in tool_calls:
                if tool_call.get("ill_formed_reasons"):
                    tool_output = {
                        "output": f'Error: {tool_call["ill_formed_reasons"][0]} Please check your function call format.'
                    }
                else:
                    tool_output = await run_tool(
                        args,
                        tool_call=deepcopy(tool_call),
                        defined_tools=tools,
                        remote_env_info=remote_env_info,
                        need_check=True,
                    )
                assert "output" in tool_output and isinstance(tool_output["output"], str), tool_output
                if tool_output.get("finish_turn", False):
                    finish_turn = True
                if tool_output.get("reward", None) is not None:
                    reward = tool_output["reward"]
                tool_outputs.append(tool_output)
        else:
            tool_outputs = [{"output": "Error: no function call found. Please check your function call format."}]
        tool_output_message = {"role": "tool", "content": tool_outputs}
        messages.append(tool_output_message)
        if finish_turn:
            break

        tool_output_prompt = message_processor.apply_chat_template([tool_output_message], add_special_tokens=False).removeprefix(stop_token)
        next_prompt = prompt + tool_output_prompt
        next_prompt_token_ids = prompt_token_ids + tokenizer.encode(tool_output_prompt, add_special_tokens=False)
        next_prompt_by_mp = message_processor.apply_chat_template(messages, tools=tools)
        next_prompt_length = len(next_prompt_token_ids)
        observation_length = next_prompt_length - prompt_length

        assert next_prompt_by_mp == next_prompt

        if next_prompt_length > max_context_length - 1:
            break

        response_loss_mask += [0] * observation_length
        rollout_log_probs += [0.0] * observation_length
        prompt, prompt_token_ids, prompt_length = next_prompt, next_prompt_token_ids, next_prompt_length

        if generate_state.aborted:
            aborted = True
            break

    assert prompt[: len(initial_prompt)] == initial_prompt and prompt_token_ids[:initial_prompt_length] == initial_prompt_token_ids
    assert len(response_loss_mask) == prompt_length - initial_prompt_length
    assert len(rollout_log_probs) == prompt_length - initial_prompt_length

    if remote_env_info and "url" in remote_env_info:
        if not (args.partial_rollout and aborted and len(sample.response) > 0):
            tool_call = {
                "session_id": session_id,
                "tool_call_id": "tool-call-" + str(uuid.uuid4()),
                "name": "close_session",
                "arguments": json.dumps({}),
            }
            await run_tool(args, tool_call=tool_call, defined_tools=tools, remote_env_info=remote_env_info, need_check=False)

    sample.response = prompt[len(initial_prompt) :]
    sample.tokens = prompt_token_ids
    assert len(prompt_token_ids) <= max_context_length, len(prompt_token_ids)
    sample.rollout_log_probs = rollout_log_probs
    sample.loss_mask = response_loss_mask
    sample.response_length = prompt_length - initial_prompt_length
    sample.reward = reward
    if aborted:
        sample.status = Sample.Status.ABORTED
    elif truncated:
        sample.status = Sample.Status.TRUNCATED
    else:
        sample.status = Sample.Status.COMPLETED

    task_unfinished = not finish_turn
    sample.metadata.update(
        {
            "history": messages,
            "loss_token_num": sum(response_loss_mask),
            "use_tool": use_tool,
            "env_interact_steps": env_interact_steps,
            "task_unfinished": task_unfinished,
            "ill_formed": ill_formed,
            "is_evaluation": evaluation,
        }
    )
    return sample
