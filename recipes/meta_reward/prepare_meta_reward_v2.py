import pyarrow as pa
import pyarrow.parquet as pq
import random
import re
import json
import numpy as np


planning_prompt = """### You are an expert Evaluation Architect.

### Objective: Analyze the provided Query and Response A to extract highly discriminative rubrics. These rubrics must capture both explicit and implicit intents and establish clear quality benchmarks based on Response A's performance.

### Instructions:
- **Intent Analysis**: Identify the critical features (logic, style, factual density, structure) that determine the quality of the response to this specific query.
- **Criteria Translation**: Convert observations into objective, descriptive requirements.
- **Blind Evaluation**: Do NOT mention "Response A" or "the candidate" by name; describe specific "content behaviors" or "patterns."

### Rubric Component Requirements: Each <rubric> tag must follow this internal structure:
- [Criterion Name]: Formal name of the evaluation aspect.
- [Definition]: Core logic of this evaluation dimension.
- Tip 1 (Baseline): The mandatory, quantifiable, or hard standard any response must meet.
- Tip 2 (Advanced): Soft standards or "value-add" features that further enhance quality.
- Tip 3 (Superior Pattern): Specific positive behaviors or strengths observed in Response A.
- Tip 4 (Failure Pattern): Specific omissions, errors, or areas for improvement identified in Response A.

### Input Data:
Query: {query}

<Response A>: {response_1} </Response A>

### Output Format (Output exactly 10 rubrics):
- Format: One rubric per line, wrapped in <rubric>...</rubric> tags. No introductory text, no conclusion, and no numbering outside the tags.
- Internal Structure: [Name]: [Definition]. Tip 1: [Content]; Tip 2: [Content]; Tip 3: [Content]; Tip 4: [Content].

### Output Example: <rubric>Technical Precision: Evaluates accuracy and robustness. Tip 1: Must provide executable code for all mentioned functions; Tip 2: Prefer responses that include time complexity analysis; Tip 3: Reward the use of modular functions as seen in high-quality implementations; Tip 4: Penalize the lack of input validation or error-handling logic.</rubric>"""

select_prompt = """### You are a Strategic Evaluation Architect.

### Objective: Analyze the newly provided Response B against the existing Query and Response A. Select the single most discriminative rubric from the provided list that best exposes the "Critical Quality Gap" between the two candidates.

### Selection Criteria:
- Core Intent Alignment: The rubric must target the primary intent or critical constraints of the Query.
- Delta Maximization: Choose the rubric where the "Tips" (1-4) create the widest scoring gap between A and B based on their observed performance.
- Decisive Actionability: The rubric must provide unambiguous "decision rules" that allow a judge to declare a clear winner without subjective doubt.

### Input Data:
<Response B>: {response_2} </Response B>

### Output Format:
Selected Rubric from previous rubric list: <rubric>[Insert the full, original tag and content here]</rubric>"""


judge_prompt = """### You are a Precision Evaluation Engine

### Objective: Perform a rigorous, rubric-grounded comparison between Response A and Response B. You are a neutral judge; the provided rubric is the sole legal framework for your decision.

### Evaluation Protocol:
- Evidence Extraction: Pinpoint specific segments, phrases, or omissions in both responses that trigger the "Tips" within the Rubric.
- Tip-Grounded Comparison: Every evaluative claim must be explicitly linked to a Tip ID (e.g., "Satisfies Tip 3" or "Triggers Failure Pattern in Tip 4").

### Scoring Scale (1-10):
9-10 (Exceptional): Flawless; exceeds expectations; fully embodies all "Superior Patterns."
7-8 (Strong): Clear adherence to most tips; high utility with only negligible omissions.
5-6 (Satisfactory): Meets basic requirements and "Baseline Tips"; functional but lacks "Advanced" brilliance.
3-4 (Poor): Significant violations of tips; missing core depth or containing notable errors.
1-2 (Failure): Completely misses the query intent or violates almost all rubric guidelines.

### Output Format (Analysis Section):
Step 1: Evidence Analysis
Response A Evidence: [Quote/Reference vs. Tip ID]
Response B Evidence: [Quote/Reference vs. Tip ID]
---
Step 2: Final Synthesis
[Analyze the decisive "delta" (quality gap). Explain exactly why one candidate outperformed the other based on the rubric's hierarchy of tips.]
---
Final Results Section
(Ensure the following labels appear exactly as shown at the end of your response)
SCORE_A: [Integer 1-10]
SCORE_B: [Integer 1-10]
VERDICT: [[A]] or [[B]]"""


def build_criteria_TTS_conversation(idx, query, response_1, response_list, ground_truth, ground_truth_list):

    prompt = planning_prompt.format(query=query, response_1=response_1)    
    data = {
        "data_source": "HelpSteer3",
        "agent_name": "criteria_TTS",
        "prompt": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        'reward_model': {
            'ground_truth': ground_truth,
        },
        "extra_info": {
            "idx":  idx,
            "interaction_kwargs": {
                "name": "criteria_TTS_interaction",
                "mode": "meta_reward_golden",
                "query": query,
                "golden_rubric": "",
                "response_list": response_list,
                'ground_truth': ground_truth,
                "ground_truth_list": ground_truth_list
            },
        }
    }
    # print(ground_truth, ground_truth_list)
    # import pdb; pdb.set_trace()
    return data

def _extract(text):
    input_part = re.search(r"(?<=## INPUT)([\s\S]*?)(?=## RESPONSE 1)", text)
    resp1_part = re.search(r"(?<=## RESPONSE 1)([\s\S]*?)(?=## RESPONSE 2)", text)
    resp2_part = re.search(r"(?<=## RESPONSE 2)([\s\S]*?)(?=### EVALUATION)", text)

    return input_part.group(1).strip(), resp1_part.group(1).strip(), resp2_part.group(1).strip()


def build_lang_data():

    lang_train_data = []
    ground_truth_0 = ground_truth_1 = 0
    count = 0
    with open("/workspace/mnt/lxb_work/GRM/GRM-omni-save/bon_helpsteer3/Skywork-Reward-V2-no-thinking-dis_judge-bo12x2/dis_judge.jsonl") as f:

        for line in f:
            TOTAL_SAMPLES_TO_PICK = 5
            picked_indices = list([22, 23])

            json_item = json.loads(line)
            data = np.array(json_item['judge_list'])
            
            thresholds = np.percentile(data, np.linspace(0, 100, 11)[1:-1])
            segment_ids = np.digitize(data, thresholds, right=True)
            used_segments = {segment_ids[i] for i in picked_indices}
            remaining_indices = [i for i in range(len(data)) if i not in [22, 23]]
            
            segment_to_indices = {}
            for i in remaining_indices:
                seg_id = segment_ids[i]
                if seg_id not in segment_to_indices:
                    segment_to_indices[seg_id] = []
                segment_to_indices[seg_id].append(i)

            # 优先从没被占用的段位里选
            available_segments = list(segment_to_indices.keys())
            random.shuffle(available_segments) # 增加随机性


            anchor_seg = segment_ids[22]
            picked_segments = {int(segment_ids[22]), int(segment_ids[23])}
            needed = TOTAL_SAMPLES_TO_PICK - len(picked_indices)
            
            # 将所有索引按 segment_id 分组
            seg_map = {}
            for i in remaining_indices:
                if i == 23: continue
                s_id = int(segment_ids[i])
                if s_id not in seg_map: seg_map[s_id] = []
                seg_map[s_id].append(i)

            # 计算所有段位与 anchor 的距离并排序 (从远到近)
            sorted_segs = sorted(seg_map.keys(), key=lambda x: abs(x - anchor_seg), reverse=True)
            
            # 设定最小间距阈值 (例如不希望选相邻的，设为 2)
            MIN_GAP = 2 

            # 第一轮：尝试寻找满足间距要求的远端段位
            for s_id in sorted_segs:
                if len(picked_indices) >= TOTAL_SAMPLES_TO_PICK:
                    break
                
                # 检查当前 s_id 是否与已选的所有段位都保持了足够的距离
                if all(abs(s_id - p) >= MIN_GAP for p in picked_segments):
                    idx = random.choice(seg_map[s_id])
                    picked_indices.append(idx)
                    picked_segments.add(s_id)

            # 第二轮：如果因为间距太严苛没选满，则放宽要求补齐
            if len(picked_indices) < TOTAL_SAMPLES_TO_PICK:
                for s_id in sorted_segs:
                    if len(picked_indices) >= TOTAL_SAMPLES_TO_PICK:
                        break
                    if s_id not in picked_segments:
                        idx = random.choice(seg_map[s_id])
                        picked_indices.append(idx)
                        picked_segments.add(s_id)

            # # 计算所有段位与 anchor 的距离并排序
            # sorted_segs = sorted(seg_map.keys(), key=lambda x: abs(x - anchor_seg), reverse=True)
            
            # # 依次从距离最远的段位中各挑一个索引
            # for s_id in sorted_segs:
            #     if len(picked_indices) >= TOTAL_SAMPLES_TO_PICK:
            #         break
            #     # 优先挑这个段位里还没被选过的索引
            #     idx = random.choice(seg_map[s_id])
            #     picked_indices.append(idx)

            # # 第一轮：选没用过的段位
            # for seg_id in available_segments:
            #     if len(picked_indices) >= TOTAL_SAMPLES_TO_PICK:
            #         break
            #     if seg_id not in used_segments:
            #         idx = random.choice(segment_to_indices[seg_id])
            #         picked_indices.append(idx)
            #         used_segments.add(seg_id)

            # # 第二轮：如果还没满（比如数据太集中在某些段），随机补齐
            # if len(picked_indices) < TOTAL_SAMPLES_TO_PICK:
            #     remaining_all = [i for i in remaining_indices if i not in picked_indices]
            #     random.shuffle(remaining_all)
            #     picked_indices.extend(remaining_all[:TOTAL_SAMPLES_TO_PICK - len(picked_indices)])
            
            final_samples = [{"index": i, "score": data[i], "segment": int(segment_ids[i])} for i in picked_indices]
            
            if (json_item['answer'] == 1 and segment_ids[22] < segment_ids[23]) or (json_item['answer'] == 0 and segment_ids[22] > segment_ids[23]):
                
                count += 1
                query = json_item['list_data']['query']
                if json_item['answer'] == 0:
                    ground_truth_0 += 1
                    response_1 = json_item['list_data']['response_list'][22]
                    response_2 = json_item['list_data']['response_list'][23]
                elif json_item['answer'] == 1:
                    ground_truth_1 += 1
                    response_1 = json_item['list_data']['response_list'][23]
                    response_2 = json_item['list_data']['response_list'][22]
                
                response_list = []
                ground_truth_list = []
                for i in picked_indices:
                    if json_item['list_data']['response_list'][i] == response_1:
                        ground_truth = int(segment_ids[i])
                        continue
                    
                    response_list.append(json_item['list_data']['response_list'][i])
                    ground_truth_list.append(int(segment_ids[i]))
                
                if len(response_list) < 4:
                    continue
                if len(ground_truth_list) < 4:
                    continue
                if len(ground_truth_list) != len(response_list):
                    continue
                
                data = build_criteria_TTS_conversation(str(len(lang_train_data)), query, response_1, response_list, ground_truth, ground_truth_list)
                lang_train_data.append(data)

    print(count)
    print(ground_truth_0, ground_truth_1)
    random.shuffle(lang_train_data)
    print(len(lang_train_data))
    table = pa.Table.from_pylist(lang_train_data)

    pq.write_table(table, "/workspace/mnt/lxb_work/GRM/Tau-omni/data/rubric_rm/meta_reward_train_data.parquet")

def main():

    build_lang_data()

if __name__ == "__main__":
    main()