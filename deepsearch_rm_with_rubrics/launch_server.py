import httpx
import asyncio
from openai import AsyncOpenAI
import json
import asyncio
import re
import time
import traceback
from fastapi import FastAPI, Request, HTTPException
import uvicorn
import argparse
from collections import defaultdict, deque

OUTCOME_REWARD_JUDGE_PROMPT = open("./prompts/get_outcome_reward.txt").read()
IDENTIFY_ENTITY_PROMPT = open("./prompts/identify_entity.txt").read()
JUDGE_RUBRIC_PROMPT = open("./prompts/judge_rubric.txt").read()

app = FastAPI()

class GPTModel:
    def __init__(self, model_name, base_url, api_key):
        super().__init__()
        self.model_name = model_name
        self.client = AsyncOpenAI(
            api_key=api_key, 
            base_url=base_url,
            http_client=httpx.AsyncClient(proxy=None)
        )
        
    async def get_resp(self, message_list):
        for i in range(3):
            try:
                chat_completion = await self.client.chat.completions.create(
                    messages=message_list,
                    model=self.model_name,
                )
                print(chat_completion)
                output = chat_completion.choices[0].message.content
                return output
            except Exception as e:
                print(f"[LLM Judge Internal Error] Attempt {i+1}/3. Exception: {e}\nTraceback: {traceback.format_exc()}")
                time.sleep(1)
                continue
        print(f"[LLM Judge Internal Error] All request failed, last exception: {e if 'e' in locals() else ''}")
        return ''

async def get_outcome_reward(response, question, answer):
    prompt = OUTCOME_REWARD_JUDGE_PROMPT.format(question=question, correct_answer=answer, response=response)
    judge_messages = [{"role": "user", "content": prompt}]
    tries = 0
    judgement = ""
    correctness, extracted_final_answer = None, None
    correctness_pattern = r"(?i)\*{0,2}correct\*{0,2}\s*:\s*(no|yes)"
    extracted_pattern = r"(?i)\*{0,2}extracted_final_answer\*{0,2}\s*:\s*(.+)"
    while tries < 3:
        judgement = await reward_model.get_resp(judge_messages)
        match = re.search(correctness_pattern, judgement, flags=re.IGNORECASE)
        correctness = match.group(1) if match else None
        match = re.search(extracted_pattern, judgement, flags=re.IGNORECASE)
        extracted_final_answer = match.group(1) if match else None
        if correctness and extracted_final_answer:
            try:
                correctness = correctness.lower()
                break
            except:
                pass
        tries += 1
    accuracy = 1.0 if correctness == "yes" else 0
    
    res = {}
    res["extracted_final_answer"] = extracted_final_answer
    res["golden_answer"] = answer
    res["judgement"] = judgement
    res["reward"] = accuracy
    return res


def extract_entity_ids(text):
    pattern = r"<(E\d+)>"
    matches = re.findall(pattern, text)
    unique_matches = sorted(set(matches), key=lambda x: int(x[1:]))
    return unique_matches

def extract_json_block(model_output: str) -> dict:
    match = re.search(r'```json\s*(.*?)\s*```', model_output, re.DOTALL | re.IGNORECASE)
    if not match:
        return {}

    json_str = match.group(1).strip()
    
    try:
        data = json.loads(json_str)
        return data
    except json.JSONDecodeError as e:
        print(f"[Error] JSON Parsing Failed: {e}")
        return {}

def check_identified_entities(all_entity_values, all_entity_ids):
    if not isinstance(all_entity_values, dict):
        return False
    if set(all_entity_values.keys()) != set(all_entity_ids):
        return False
    for id, entity in all_entity_values.items():
        if entity is not None and not isinstance(entity, str):
            return False
        if isinstance(entity, str) and entity.lower().strip() in ["null", "none"]:
            return False
    return True

def fill_rubric_with_entity_value(rubric, all_entity_values):
    pattern = r"<(E\d+)>"
    entities_in_line = re.findall(pattern, rubric)

    filled_rubric = rubric
    all_filled = True
    current_entity_values = {}
    for key in entities_in_line:
        value = all_entity_values.get(key, None)
        current_entity_values[key] = value
        if value is not None:
            filled_rubric = filled_rubric.replace(f"<{key}>", str(value))
        else:
            all_filled = False
    
    return filled_rubric, all_filled, current_entity_values

def extract_search_results(text, query):
    pattern = re.compile(r"""
    \[(\d+)\]\sTitle:\s(.+?)\s*
    \[\1\]\sURL\sSource:\s(.+?)\s*
    \[\1\]\sDescription:\s(.+?)\s*
    (?:\[\1\]\sDate:\s(.+?))?\s*(?=\[\d+\]\sTitle:|\Z)
    """, re.DOTALL | re.VERBOSE)
    search_results = []
    for m in pattern.finditer(text):
        idx = int(m.group(1).strip())
        title = m.group(2).strip()
        url = m.group(3).strip()
        snippet = m.group(4).strip() if m.group(4) else ""
        date_published = m.group(5).strip() if m.group(5) else ""
        search_results.append({
            "idx": idx,
            "query": query,
            "url": url,
            "title": title,
            "snippet": snippet,
            "published_time": date_published,
        })
    return search_results

def extract_open_results(text):
    # Title
    title_match = re.search(r'Title:\s*(.+)', text)
    title = title_match.group(1).strip() if title_match else ""

    # URL Source
    url_match = re.search(r'URL Source:\s*(\S+)', text)
    url = url_match.group(1).strip() if url_match else ""

    # Published Time
    time_match = re.search(r'Published Time:\s*([^\n]+)', text)
    published_time = time_match.group(1).strip() if time_match else ""

    # Markdown Content
    md_match = re.search(r'Markdown Content:\s*(.*)', text, flags=re.S)
    markdown_content = md_match.group(1).strip() if md_match else ""

    return {
        "title": title,
        "url": url,
        "published_time": published_time,
        "content": markdown_content
    }

def extract_citation_content(response, history, max_citation_num=20):
    pattern = r'\[\d+\]\((?:view-source:)?(https?://.+?)\)[\],]'
    citation_urls = re.findall(pattern, response)
    deduplicated_citation_urls = []
    url_set = set()
    for url in citation_urls:
        if "https://r.jina.ai/" in url:
            url = url.replace("https://r.jina.ai/", "")
        if "http://r.jina.ai/" in url:
            url = url.replace("http://r.jina.ai/", "")
        if "view-source:" in url:
            url = url.replace("view-source:", "")
        if url not in url_set:
            url_set.add(url)
            deduplicated_citation_urls.append(url)
    print(f"[Info] Cited URLS: {json.dumps(deduplicated_citation_urls, indent=2, ensure_ascii=False)}")
    if len(deduplicated_citation_urls) == 0:
        return ""
    url_contents = {}
    tool_calls = {}
    current_url = None
    # print(f"[Info] History: {json.dumps(history, indent=2, ensure_ascii=False)}")
    for item in history:
        if item["role"] == "assistant" and "tool_calls" in item:
            for tool_call in item["tool_calls"]:
                tool_calls[tool_call["tool_call_id"]] = tool_call
        if item["role"] == "tool":
            for tool_output in item["content"]:
                if "tool_call_id" not in tool_output:
                    continue
                tool_call_id = tool_output["tool_call_id"]
                tool_output_content = tool_output["output"]
                if tool_output_content.strip() == "":
                    continue
                tool_call = tool_calls[tool_call_id]
                tool_name = tool_call["name"]
                arguments = json.loads(tool_call["arguments"])
                # search
                if tool_name == "browser.search":
                    query = arguments["query"]
                    search_results = extract_search_results(tool_output_content, query)
                    for block in search_results:
                        url = block["url"]
                        if url not in url_contents:
                            url_contents[url] = defaultdict(list)
                            if block["title"]:
                                url_contents[url]["title"] = block["title"]
                            if block["published_time"]:
                                url_contents[url]["published_time"] = block["published_time"]
                        url_contents[url]["search_results"].append(block)
                elif tool_name == "browser.open":
                    block = extract_open_results(tool_output_content)
                    url = block["url"]
                    current_url = url
                    if url not in url_contents:
                        url_contents[url] = defaultdict(list)
                        if block["title"]:
                            url_contents[url]["title"] = block["title"]
                        if block["published_time"]:
                            url_contents[url]["published_time"] = block["published_time"]
                    url_contents[url]["open_results"].append(block)
                elif tool_name == "browser.find":
                    keyword = arguments["pattern"]
                    url = current_url
                    if url not in url_contents:
                        url_contents[url] = defaultdict(list)
                    block = {"keyword": keyword, "matches": tool_output_content}
                    url_contents[url]["find_results"].append(block)

    citation_content = ""
    snippet_set = set()
    open_content_set = set()
    find_match_set = set()
    url_cnt = 0
    for url in deduplicated_citation_urls:
        if url in url_contents:
            url_cnt += 1
            citation_content += f"[Webpage {url_cnt}]\nURL Source: {url}\n"
            title = url_contents[url].get("title", None)
            published_time = url_contents[url].get("published_time", None)
            if title:
                citation_content += f"Title: {title}\n"
            if published_time:
                citation_content += f"Published Time: {published_time}\n"
            citation_content += "\n"
            for block in url_contents[url].get("search_results", []):
                snippet = block["snippet"].strip()
                if snippet not in snippet_set:
                    citation_content += f"Snippet:\n{block['snippet']}\n\n"
                    snippet_set.add(snippet)
            for block in url_contents[url].get("open_results", []):
                open_content = block['content'].strip()
                if open_content not in open_content_set:
                    citation_content += f"Content:\n{block['content']}\n\n"
                    open_content_set.add(open_content)
            for block in url_contents[url].get("find_results", []):
                keyword, matches = block["keyword"], block["matches"]
                if matches not in find_match_set:
                    citation_content += f'Search Result of Keyword "{keyword}":\n{matches}\n\n'
                    find_match_set.add(matches)
            if url_cnt == max_citation_num:
                break
    return citation_content

def check_rubric_judgement(judge_rubric_results, rubric_idx_map):
    if not isinstance(judge_rubric_results, dict):
        return False
    if set(judge_rubric_results.keys()) != set(rubric_idx_map.keys()):
        return False
    for id, result in judge_rubric_results.items():
        if not isinstance(result, bool):
            return False
    return True

async def get_rubric_reward(response, question, answer, history, rubrics, max_tries=5):
    res = {}
    if len(rubrics) == 0:
        print(f"Error: No rubrics.\nQuestion: {question}")
        res = {
            "error_reason": "No rubrics",
            "reward": 0,
        }
        return res

    # 1. Identify entities
    rubric_text = ""
    for idx, rubric in enumerate(rubrics):
        rubric_text += f"C{idx+1}. {rubric}\n"
    all_entity_ids = extract_entity_ids(rubric_text)
    identify_entity_prompt = IDENTIFY_ENTITY_PROMPT.format(question=question, constraints=rubric_text.strip(), response=response)
    identify_entity_messages = [{"role": "user", "content": identify_entity_prompt}]
    tries = 0
    while tries < max_tries:
        identify_entity_model_output = await reward_model.get_resp(message_list=identify_entity_messages)
        print(f"[Info] Identify entities. Attempt {tries}/{max_tries}.\n")
        print(f"Identify Entity Prompt:\n{identify_entity_prompt}\n\nIdentify Entity Model Output:\n{identify_entity_model_output}")
        all_entity_values = extract_json_block(identify_entity_model_output)
        res.update({
            "identify_entity_model_output": identify_entity_model_output,
            "all_entity_values": all_entity_values,
        })
        if check_identified_entities(all_entity_values, all_entity_ids):
            break
        tries += 1
    else:
        print("[Error] LLM judge fails to extract entity identities. last output:")
        print(identify_entity_model_output)
        res.update({
            "error_reason": "LLM judge fails to extract entity identities",
            "reward": 0
        })
        return res
    
    # 2. Judge rubrics
    # 2.1 fill entities
    identified_rubrics = []
    rubric_scores = {}
    for idx, rubric in enumerate(rubrics):
        filled_rubric, all_filled, entity_values = fill_rubric_with_entity_value(rubric, all_entity_values)
        if not all_filled:
            rubric_scores[idx] = {
                "raw_rubric": rubric, 
                "filled_rubric": filled_rubric, 
                "entity_values": entity_values, 
                "all_entity_identified": False, 
                "is_supported": False,
                "connected_to_answer": False,
                "score": 0,
            }
        else:
            rubric_scores[idx] = {
                "raw_rubric": rubric,
                "filled_rubric": filled_rubric, 
                "entity_values": entity_values, 
                "all_entity_identified": True, 
                "is_supported": False,
                "connected_to_answer": False,
                "score": 0
            }
            identified_rubrics.append((idx, filled_rubric))
    res["rubric_scores"] = rubric_scores
    if len(identified_rubrics) == 0:
        res["reward"] = 0
        return res
    
    # 2.2 extract citation context
    citation_content = extract_citation_content(response, history, max_citation_num=20)
    res.update({"citation_content": citation_content})
    if citation_content.strip() == "":
        print(f"[Warning] The model response No citation content:\n{response}")
        res["reward"] = 0
        return res

    # 2.3 judge all filled rubrics
    rubric_text = ""
    rubric_idx_map = {}
    for i, (idx, rubric) in enumerate(identified_rubrics):
        rubric_text += f"S{i+1}. {rubric}\n"
        rubric_idx_map[f"S{i+1}"] = idx
    # print(rubric_text)
    judge_rubric_prompt = JUDGE_RUBRIC_PROMPT.format(context=citation_content, statements=rubric_text.strip())
    judge_rubric_messages = [{"role": "user", "content": judge_rubric_prompt}]
    tries = 0
    while tries < max_tries:
        judge_rubric_model_output = await reward_model.get_resp(message_list=judge_rubric_messages)
        print(f"[Info] Judge rubrics. Attempt {tries}/{max_tries}.\n")
        print(f"Judge Rubric Prompt:\n{judge_rubric_prompt}\n\nJudge Rubric Model Output:\n{judge_rubric_model_output}")
        judge_rubric_results = extract_json_block(judge_rubric_model_output)
        res.update({
            "judge_rubric_model_output": judge_rubric_model_output,
            "judge_rubric_results": judge_rubric_results,
        })
        if check_rubric_judgement(judge_rubric_results, rubric_idx_map):
            break
        tries += 1
    else:
        print("[Error] LLM judge fails to judge rubrics. last output:")
        print(judge_rubric_model_output)
        res.update({
            "error_reason": "LLM judge fails to judge rubrics",
            "reward": 0
        })
        return res
    for id, judge_result in judge_rubric_results.items():
        idx = rubric_idx_map[id]
        rubric_scores[idx]["is_supported"] = judge_result

    # 3. Use BFS to judge whether the supported rubric can be arrived from E0
    entity2rubrics = defaultdict(set)
    for idx, rubric in rubric_scores.items():
        if rubric["is_supported"] == 1:
            for entity in rubric["entity_values"]:
                entity2rubrics[entity].add(idx)
    arrivable_entites = {"E0"}
    arrivable_rubrics = set()
    queue = deque(["E0"])
    while queue:
        entity = queue.popleft()
        for idx in entity2rubrics[entity]:
            arrivable_rubrics.add(idx)
            for other_entity in rubric_scores[idx]["entity_values"]:
                if other_entity not in arrivable_entites:
                    arrivable_entites.add(other_entity)
                    queue.append(other_entity)
    total_score = 0
    for idx, rubric in rubric_scores.items():
        if idx in arrivable_rubrics:
            rubric["connected_to_answer"] = True
            rubric["score"] = 1
            total_score += 1
    res["rubric_scores"] = rubric_scores
    res["reward"] = total_score / len(rubric_scores)
    return res

async def get_reward(response, question, answer, history, rubrics, rubric_reward_ratio=0):

    response_text_judge = response
    if '## References' in response_text_judge:
        response_text_judge = response_text_judge.rsplit('## References', 1)[0].strip()

    outcome_reward = 0
    rubric_reward = 0
    if rubric_reward_ratio < 1:
        outcome_reward_result = await get_outcome_reward(response_text_judge, question, answer)
        outcome_reward = outcome_reward_result["reward"]
    if rubric_reward_ratio > 0:
        rubric_reward_result = await get_rubric_reward(response_text_judge, question, answer, history, rubrics)
        rubric_reward = rubric_reward_result["reward"]
    final_reward = (1 - rubric_reward_ratio) * outcome_reward + rubric_reward_ratio * rubric_reward
    
    res = {
        "question": question,
        "response_text": response,
        "response_text_judge": response_text_judge,
        "label": answer,
        "rubric_reward_ratio": rubric_reward_ratio,
    }
    if rubric_reward_ratio < 1:
        res["outcome_reward_result"] = outcome_reward_result
    if rubric_reward_ratio > 0:
        res["rubric_reward_result"] = rubric_reward_result
    res["reward"] = final_reward
    return res
    
@app.post("/evaluate")
async def evaluate(request: Request):
    try:
        data = await request.json()
        
        # check reuqired arguments
        for key in ["history", "label", "task_unfinished", "remote_env_info"]:
            if key not in data:
                raise HTTPException(status_code=400, detail=f"Miss arguments: {key}")
            
        task_unfinished = data["task_unfinished"]
        if task_unfinished:
            return {
                "reward": 0,
                "outcome_reward": 0,
                "rubric_reward": 0,
                "rubric_scores": {}
            }
        answer = data["label"]
        history = data["history"]
        remote_env_info = data["remote_env_info"]
        question = remote_env_info["search_forbidden_strs"][0]
        rubrics = remote_env_info.get("rubrics", [])
        rubric_reward_ratio = remote_env_info.get("rubric_reward_ratio", 0)
        if history[-1]['role'] != 'assistant' or 'content' not in history[-1]:
            return {
                "reward": 0,
                "outcome_reward": 0,
                "rubric_reward": 0,
                "rubric_scores": {}
            }
        response = history[-1]['content']
        
        result = await asyncio.wait_for(get_reward(response, question, answer, history, rubrics, rubric_reward_ratio), timeout=600)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        # if ("outcome_reward_result" in result and not result["outcome_reward_result"].get("judgement")) or \
        #     ("rubric_reward_result" in result and not result["rubric_reward_result"].get("identify_entity_model_output")):
        #     raise HTTPException(
        #         status_code=500,  
        #         detail="LLM request failed。Please check your LLM server or token usage"
        #     )
        reward = result['reward']
        return {
            "reward": reward,
            "outcome_reward": result.get("outcome_reward_result", {}).get("reward", 0),
            "rubric_reward": result.get("rubric_reward_result", {}).get("reward", 0),
            "rubric_scores": result.get("rubric_reward_result", {}).get("rubric_scores", {}),
        }
        
    except Exception as e:
        print(f"Error when Processing Request: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Server Internal Error: {str(e)}")
    
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="deepseek-chat")
    parser.add_argument("--base_url", type=str, default="https://api.deepseek.com")
    parser.add_argument("--api_key", type=str, default="sk-xxx")
    
    return parser.parse_args()

@app.on_event("startup")
async def startup_event():
    global reward_model
    
    args = get_args()
    reward_model = GPTModel(model_name=args.model_name, base_url=args.base_url, api_key=args.api_key)

if __name__ == "__main__":
    args = get_args()
    uvicorn.run(app, host=args.host, port=args.port, reload=False) 