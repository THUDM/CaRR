# Using CaRR as Remote Reward Model Server 

This folder contains scripts for running **CaRR** as a remote reward model server.

## 📦 Step 1: Install Dependencies

```bash
cd ./deepsearch_rm_with_rubrics
pip install -r requirements.txt
```

## 🚀 Step2: Launch Server

We use DeepSeek official API as a reference. You can also use locally deployed LLMs with VLLM or SGlang or other APIs by changing the `base_url` and `api_key`.

```bash
python3 launch_server.py --port 8888 --model_name deepseek-chat --base_url https://api.deepseek.com --api_key <your DeepSeek API key>
```

## ✅ Step 3: Test the Server
Once the server is running, verify it with the provided test script:

```bash
python3 test.py
```

The output should be like:
```json
{'reward': 0.8, 'outcome_reward': 1.0, 'rubric_reward': 0.6, 'rubric_scores': {'0': {'raw_rubric': '<E0> is an MMA event that occurred before 2022.', 'filled_rubric': 'UFC 219: Cyborg vs. Holm is an MMA event that occurred before 2022.', 'entity_values': {'E0': 'UFC 219: Cyborg vs. Holm'}, 'all_entity_identified': True, 'is_supported': True, 'connected_to_answer': True, 'score': 1}, ...}}
```
where the `reward` is a wegithed sum of `outcome_reward` and `rubric_reward`. 

In C-GRPO, we first obtain the rewards for all rollouts in a group, then use each `outcome_reward` and `rubric_reward` to recompute the final reward based on the formula described in our paper.