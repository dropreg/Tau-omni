import pyarrow as pa
import pyarrow.parquet as pq
import random
import re
import json


planning_prompt = """###### You are an expert Evaluation Architect. Your goal is to create a set of highly discriminative, self-contained rubrics that can accurately rank candidate answers to a specific query. 

### Objective
The rubrics must be designed to highlight the specific quality gap between the provided responses without explicitly naming them. They must be general enough to judge future responses but specific enough to capture the unique nuances, strengths, and failures observed in the current candidates.

### Instructions
1. **Analyze the Delta:** Compare the provided candidate responses. Identify the specific features, reasoning steps, or stylistic choices that make one superior to the other.
2. **Translate to Criteria:** Turn those specific observations into objective, descriptive requirements. 
    * *Example:* If one response uses a table and the other is a wall of text, the rubric should reward "Structural Clarity" and specify "Preference for tabular data over dense prose."
3. **Draft Rubrics:** Ensure each is "response-aware" but "blind" (do NOT mention "Response A/B" or "the first/second response").

### Rubric Component Requirements
Each `<rubric>` tag must contain:
- **Criterion Name:** A formal name for the evaluation aspect.
- **Discriminative Tips:** At least 3 specific, verifiable tips based on the observed differences:
    - **Tip 1 (Superior Pattern):** Describe a specific positive behavior or detail present in the better candidate.
    - **Tip 2 (Failure Pattern):** Describe a specific omission or error observed in the weaker candidate.
    - **Tip 3 (General Quality Bar):** A measurable standard that applies to any response for this query.

### Constraint: Blind Evaluation
**CRITICAL:** Describe the *content* or *behavior* (e.g., "Prefer answers that define technical terms" or "Penalize responses that ignore the budget constraint").

### Input Data
- **Original Query:** {query}
- **Candidate Responses:**
    - <Response_A>{response_1}</Response_A>
    - <Response_B>{response_2}</Response_B>

### Output Format Requirements
- Output ten rubrics.
- Format: One rubric per line, wrapped in `<rubric>...</rubric>` tags.
- No introductory text, no conclusion, and no numbering outside the tags.
- **Internal Structure:** [Criterion Name]: [Definition]. Tip 1: [Observation and Pattern 1]; Tip 2: [Observation and Pattern 2]; Tip 3: [Observation and Pattern 3].

### Output Example
<rubric>Technical Precision and Edge-Case Handling: Evaluates the accuracy of the technical solution and its robustness. Tip 1: Reward responses that provide specific error-handling logic for null inputs; Tip 2: Penalize solutions that provide high-level conceptual summaries without executable steps; Tip 3: Verify that all mathematical formulas mentioned are applied correctly to the variables provided in the query.</rubric>"""

select_prompt = """### You are a Strategic Evaluation Architect. Your task is to analyze a query and two candidate responses to select the single most discriminative rubric from a provided list.

### Objective
Identify the rubric that best captures the "Critical Quality Gap" between Response A and Response B. The chosen rubric should highlight the specific strengths or fatal flaws that make one response clearly superior to the other for this specific query.

### Selection Criteria
Select the rubric that meets these requirements:
1. **High Relevance:** Directly addresses the core intent or constraints of the query.
2. **Maximum Discrimination:** The "Tips" within the rubric must clearly apply to the observed differences between A and B (e.g., if one has code and the other doesn't, select a rubric focused on implementation).
3. **Actionability:** The rubric's tips must provide clear "decision rules" for a judge to follow.

### Output Format
**Selected Rubric:** Output the full, original `<rubric>...</rubric>` tag of your choice.

---

### Output Example
**Selected Rubric:** <rubric>Technical Implementation Depth: ... Tip 1: ... Tip 2: ...</rubric>"""


def build_criteria_TTS_conversation(idx, query, response_1, response_2, ground_truth):

    prompt = planning_prompt.format(query=query, response_1=response_1, response_2=response_2)

    assert ground_truth == 0 or ground_truth == 1
    
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
                "rejected": response_1 if ground_truth == 1 else response_2,
                "chosen": response_1 if ground_truth == 0 else response_2,
                "golden_rubric": "",
                "response_1": response_1,
                "response_2": response_2, 
                "ground_truth": ground_truth
            },
        }
    }
    
    return data

def _extract(text):
    input_part = re.search(r"(?<=## INPUT)([\s\S]*?)(?=## RESPONSE 1)", text)
    resp1_part = re.search(r"(?<=## RESPONSE 1)([\s\S]*?)(?=## RESPONSE 2)", text)
    resp2_part = re.search(r"(?<=## RESPONSE 2)([\s\S]*?)(?=### EVALUATION)", text)

    return input_part.group(1).strip(), resp1_part.group(1).strip(), resp2_part.group(1).strip()


def build_lang_data():

    file_list = [
        # "/workspace/mnt/lxb_work/GRM/GRM-omni-data/grm_omni_dataset/helpsteer3/part1/rubric_judge.jsonl",
        # "/workspace/mnt/lxb_work/GRM/GRM-omni-data/grm_omni_dataset/helpsteer3/part2/rubric_judge.jsonl",
        # "/workspace/mnt/lxb_work/GRM/GRM-omni-data/grm_omni_dataset/helpsteer3/sample_paired_raw_rl_new_score/rubric_judge.jsonl",
        "/workspace/mnt/lxb_work/GRM/GRM-omni-save/auto_rubric/skywork/rubric_rm-no-thining-fennec_v0-prompt-pairwise_1/rubric_judge.jsonl"
    ]

    lang_train_data = []
    ground_truth_0 = ground_truth_1 = 0
    for file in file_list:
        for line in open(file).readlines():
            
            json_item = json.loads(line)
            raw_id = json_item['paired_data']['meta']['raw_id']
            idx = json_item['paired_data']['id']
            query = json_item['paired_data']['query']
            chosen = json_item['paired_data']['chosen']['text_content']
            reject = json_item['paired_data']['rejected']['text_content']
            rubric_list = json_item['rubric_list']
            
            if json_item['answer'] == 0:
                response_1 = chosen
                response_2 = reject
            elif json_item['answer'] == 1:
                response_1 = reject
                response_2 = chosen

            if (json_item['judge_list'][0] == 'B' and json_item['answer'] == 0) or (json_item['judge_list'][0] == 'A' and json_item['answer'] == 1) or random.random() > 0.8:

                    ground_truth = json_item['answer']
                
                    data = build_criteria_TTS_conversation(idx, query, response_1, response_2, ground_truth)
                
                    lang_train_data.append(data)
                    
                    if ground_truth == 1:
                        ground_truth_1 += 1
                    else:
                        ground_truth_0 += 1
                    
            # for rubric, judge_score, judge, response in zip(rubric_list, json_item['judge_score_list'],, json_item['llm_response']):
                
            #     if (judge == 'A' and json_item['answer'] == 0 and json_item['paired_data']['meta']['chosen_likert_scores'] - json_item['paired_data']['meta']['reject_likert_scores'] <= int(judge_score[0]) - int(judge_score[1])) or (judge == 'B' and json_item['answer'] == 1 and json_item['paired_data']['meta']['chosen_likert_scores'] - json_item['paired_data']['meta']['reject_likert_scores'] <= int(judge_score[1]) - int(judge_score[0])):
                    
            #         ground_truth = json_item['answer']
            #         data = build_criteria_TTS_conversation(idx, query, response_1, response_2, ground_truth)
            #         lang_train_data.append(data)
                    
            #         if ground_truth == 1:
            #             ground_truth_1 += 1
            #         else:
            #             ground_truth_0 += 1
                    
            #         break
    
    print(ground_truth_0, ground_truth_1)
    random.shuffle(lang_train_data)
    print(len(lang_train_data))
    table = pa.Table.from_pylist(lang_train_data)

    pq.write_table(table, "/workspace/mnt/lxb_work/GRM/Tau-omni/data/rubric_rm/meta_reward_golden_train_data_v2.parquet")

    # id2rubric = {}
    # for file in file_list:
    #     for line in open(file).readlines():

    #         json_item = json.loads(line)
    #         raw_id = json_item['paired_data']['meta']['raw_id']

    #         if raw_id not in id2rubric:
    #             id2rubric[raw_id] = json_item['rubric_list']
    #         else:
    #             id2rubric[raw_id].extend(json_item['rubric_list'])
    
    # lang_train_data = []
    # ground_truth_0 = ground_truth_1 = 0
    # for file in file_list:
    #     for line in open(file).readlines():
            
    #         json_item = json.loads(line)
    #         raw_id = json_item['paired_data']['meta']['raw_id']
    #         if raw_id not in id2rubric:
    #             continue
    #         idx = json_item['paired_data']['id']
    #         query = json_item['paired_data']['query']
    #         chosen = json_item['paired_data']['chosen']['text_content']
    #         reject = json_item['paired_data']['rejected']['text_content']
    #         rubric_list = json_item['rubric_list']

    #         if json_item['answer'] == 0:
    #             response_1 = chosen
    #             response_2 = reject
    #         elif json_item['answer'] == 1:
    #             response_1 = reject
    #             response_2 = chosen
            
    #         rubric_set = set()
    #         _rubric_list = [r['content'] for r in id2rubric[raw_id]]
    #         random.shuffle(_rubric_list)
    #         for r in _rubric_list:

    #             if len(rubric_set) > 14:
    #                 break
    #             rubric_set.add(r)
            
    #         for r in rubric_list:
    #             rubric_set.add(r['content'])
    #         _rubric_list = list(rubric_set)
    #         _golden_rubric_list = []
            
    #         for rubric, judge, response in zip(rubric_list, json_item['judge_list'], json_item['llm_response']):
    #             if (judge == 'A' and json_item['answer'] == 0) or (judge == 'B' and json_item['answer'] == 1):
    #                 _golden_rubric_list.append(rubric['content'])
            
    #         if len(_golden_rubric_list):
    #             ground_truth = json_item['answer']
    #             data = build_criteria_TTS_conversation(idx, query, response_1, response_2, _rubric_list, _golden_rubric_list, ground_truth)
    #             lang_train_data.append(data)
                
    #             if ground_truth == 1:
    #                 ground_truth_1 += 1
    #             else:
    #                 ground_truth_0 += 1
    
    # print(ground_truth_0, ground_truth_1)
    # random.shuffle(lang_train_data)
    # print(len(lang_train_data))
    # table = pa.Table.from_pylist(lang_train_data)

    # pq.write_table(table, "/workspace/mnt/lxb_work/GRM/Tau-omni/data/rubric_rm/meta_reward_golden_train_data.parquet")

def main():

    build_lang_data()

if __name__ == "__main__":
    main()