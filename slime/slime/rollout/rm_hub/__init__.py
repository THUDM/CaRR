import asyncio
import logging
import random

import aiohttp

logger = logging.getLogger(__name__)

from slime.utils.misc import load_function
from slime.utils.types import Sample

from .deepscaler import get_deepscaler_rule_based_reward
from .f1 import f1_score
from .gpqa import compute_gpqa_reward
from .math_dapo_utils import compute_score as compute_score_dapo
from .math_utils import extract_answer as extract_boxed_answer
from .math_utils import grade_answer_verl

_shared_session: aiohttp.ClientSession | None = None


def _get_shared_session() -> aiohttp.ClientSession:
    global _shared_session
    if _shared_session is None or _shared_session.closed:
        connector = aiohttp.TCPConnector(
            limit=64,
            enable_cleanup_closed=True,
        )
        timeout = aiohttp.ClientTimeout(total=120)
        _shared_session = aiohttp.ClientSession(connector=connector, timeout=timeout)
    return _shared_session


async def remote_rm(
    args,
    sample: Sample,
    max_retries: int = 10,
    remote_url: str | None = None,
    input_keys: list[str] | None = None,
    proxy: str | None = None,
):
    if input_keys:
        payload = {}
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        for key in input_keys:
            if hasattr(sample, key):
                payload[key] = getattr(sample, key)
            elif key in metadata:
                payload[key] = metadata[key]
            else:
                raise RuntimeError(f"sample has no key '{key}'")
    else:
        payload = {
            "prompt": sample.prompt,
            "response": sample.response,
            "label": sample.label,
        }
    url = remote_url or args.rm_url
    if not url:
        raise ValueError("remote_rm requires `args.rm_url` or reward_model_info.kwargs.remote_url`.")
    session = _get_shared_session()
    for attempt in range(max_retries):
        try:
            async with session.post(url, json=payload, proxy=proxy) as resp:
                resp.raise_for_status()
                return await resp.json()
        except Exception as e:
            if attempt + 1 >= max_retries:
                logger.warning(f"remote_rm failed after {attempt + 1} attempts: {e}")
                raise
            backoff = min(2**attempt, 30) + random.random()
            logger.info(f"remote_rm: {type(e).__name__}, retrying in {backoff:.1f}s ({attempt + 1}/{max_retries})")
            await asyncio.sleep(backoff)


async def async_rm(args, sample: Sample, **kwargs):
    if args.custom_rm_path is not None:
        rm_function = load_function(args.custom_rm_path)
        return await rm_function(args, sample, **kwargs)

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    if "reward_model_info" in metadata:
        if not getattr(args, "no_punish_for_illform", False):
            is_ill_formed = metadata.get("ill_formed", False)
            is_evaluation = metadata.get("is_evaluation", False)
            if is_ill_formed and not is_evaluation:
                return {"reward": 0, "outcome_reward": 0, "rubric_reward": 0}

        reward_model_info = metadata["reward_model_info"]
        rm_function_name = reward_model_info["function"]
        reward_kwargs = reward_model_info.get("kwargs", {})
        if rm_function_name == "remote_rm":
            return await remote_rm(args, sample, **reward_kwargs)
        rm_function = load_function(rm_function_name)
        return await rm_function(sample, **reward_kwargs)

    rm_type = (metadata.get("rm_type") or args.rm_type or "").strip()
    response = sample.response
    label = sample.label
    if rm_type.startswith("boxed_"):
        response = extract_boxed_answer(response) or ""
        rm_type = rm_type[len("boxed_") :]

    # This function is intended for remote or time-consuming reward model evaluation.
    # Implement the actual logic as needed.
    if rm_type == "remote_rm":
        return await remote_rm(args, sample)
    elif rm_type == "deepscaler":
        return get_deepscaler_rule_based_reward(response, label)
    elif rm_type == "dapo":
        return compute_score_dapo(response, label)
    elif rm_type == "math":
        return 1 if grade_answer_verl(response, label) else 0
    elif rm_type == "f1":
        return f1_score(response, label)[0]
    elif rm_type == "gpqa":
        return compute_gpqa_reward(response, label, metadata=metadata)
    elif rm_type == "ifbench":
        from .ifbench import compute_ifbench_reward

        return compute_ifbench_reward(response, label, metadata=metadata)
    elif rm_type == "random":
        return random.randint(0, 1)
    elif rm_type:
        raise NotImplementedError(f"Rule-based RM for {rm_type} is not implemented.")
    else:
        raise NotImplementedError("Rule-based RM type is not specified.")


async def batched_async_rm(
    args,
    samples: list[Sample],
    **kwargs,
) -> list[int | float]:
    if args.custom_rm_path is not None:
        # Ensure the custom reward function is implemented in batch mode
        rm_function = load_function(args.custom_rm_path)
        return await rm_function(args, samples, **kwargs)
    tasks = [async_rm(args, sample, **kwargs) for sample in samples]
    rewards = await asyncio.gather(*tasks)
    return rewards
