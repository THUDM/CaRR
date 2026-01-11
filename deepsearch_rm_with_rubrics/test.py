import requests
import json
import time

js = json.load(open("/mnt/o1_alicloud/personal/zjj/dev/dr-rubric/deepsearch_rm_with_rubrics/4b_bc_examples.json"))[0]
history = js["history"][0]
question = history[0]["content"].split("\n\nYour response should be")[0]
label = js["label"][0]
task_unfinished = js["task_unfinished"][0]
rubrics = ['<E0> is an MMA event that occurred before 2022.', '<E0> featured a featherweight bout between <E1> and <E2>.', '<E1> was the loser of the featherweight bout in <E0>.', '<E1> landed 14 significant strikes out of 83 attempted in <E0>, resulting in a 16.87% significant strikes percentage.', '<E1> failed to land any takedowns out of 4 attempts in <E0>.', 'Both <E1> and <E2> were under 35 years old at the time of <E0>.', '<E1> and <E2> shared the same nationality.', 'The nickname of <E1> is a synonym for "swordsman".', '<E3> was the referee who officiated the featherweight bout in <E0>.', '<E3> worked his first event for the same MMA promotion as <E0> in 1994.']
print(rubrics)

# Define Request Data
data = {
    "history": history,
    "task_unfinished": task_unfinished,
    "label": label,
    "remote_env_info": {
        "search_forbidden_strs": [question],
        "rubrics": rubrics,
        "rubric_reward_ratio": 0.5,
    }
}

start_time = time.time()
try:
    resp = requests.post("http://127.0.0.1:8888/evaluate", json=data)
    print("Status Code:", resp.status_code)
    print("Response:")
    print(resp.json())
    print(resp.text)
except Exception as e:
    print("Error:", e)
end_time = time.time()
print(end_time - start_time)