import pyarrow as pa
import pyarrow.parquet as pq
import random
import re
import json
import uuid


THINKING_PRMOPT="""You are an objective, impartial, and unbiased content evaluator. Given a user query and two candidate assistant responses, produce a rigorous, evidence-based comparison and a single final verdict indicating which response better fulfills the user’s intent.\n\n\n### Important constraints (must follow exactly):\n1. Work only from the provided, (Assistant A), and (Assistant B). Do not introduce outside facts or assumptions.\n2. Output only the structured block described below and nothing else (no preamble, no postscript).\n3. Use exactly three evaluation criteria (no more, no fewer). The criteria must be distinct (non-overlapping) and focused on observable differences between the two responses.\n3. For each criterion, provide: A short name, A one-sentence explanation of what it measures and why it matters. Each of analyses must explicitly identify the response’s strengths AND weaknesses, especially the concrete defects relevant to that criterion.\n4. In each analysis: When pointing out a defect, explain why it is a defect with respect to the criterion (e.g., “fails to answer X”, “contradicts user intent”, “includes factual error”, “provides irrelevant content”). If a response lacks relevant content, clearly state: “Response A/B lacks evidence for X.” Every Judge analysis must include: (a) what works, (b) what fails, (c) concrete evidence, and (d) criterion-based impact.\n5. After the three criteria blocks, give a single final verdict line containing exactly [[A]] or [[B]] (choose the response that better meets the query overall).\n\n\n### Required output format (produce exactly this structure — replace placeholders with real content):\nReasoning Step xxx.\nThe final verdict is [[A]] or [[B]]\n\n\n### Input:[User Question]:\n{query}\n\n[The Start of Assistant A's Answer]:\n{response_1}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]: \n{response_2}\n[The End of Assistant B's Answer]\nPlease output your analysis and final verdict:"""


def build_criteria_TTS_conversation(idx, query, chosen, rejected):

    if random.random() > 0.5:
        ground_truth = 1
        prompt = THINKING_PRMOPT.format(query=query, response_1=rejected, response_2=chosen)
    else:
        ground_truth = 0
        prompt = THINKING_PRMOPT.format(query=query, response_1=chosen, response_2=rejected)
    
    data = {
        "data_source": "R1_Reward",
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
            "idx":  idx
        }
    }
    
    return data, ground_truth

lang_train_data = []

def messages_to_context_and_query(messages):
    assert messages[-1]["role"] == "user", "Last message must be from user"
    query = messages[-1]["content"].strip()
    context_parts = []
    for msg in messages[:-1]:
        role = msg["role"].capitalize()
        content = msg["content"].strip()
        context_parts.append(f"{role}: {content}")

    context = "\n\n".join(context_parts)
    return context, query

def messages_to_judge_input(messages):
    context, query = messages_to_context_and_query(messages)
    return f"""[Context]
    {context}

    [Query]
    {query}
    """

count = 0
ground_truth_0 = ground_truth_1 = 0
for line in open("/workspace/mnt/lxb_work/hf_dir/hf_dataset/HelpSteer3/preference/train.jsonl").readlines():
    
    json_item = json.loads(line)
    try:
        
        query = messages_to_judge_input(json_item['context'])
        if json_item['overall_preference'] < 0:
            chosen = json_item['response1']
            rejected = json_item['response2']
        elif json_item['overall_preference'] > 0:
            chosen = json_item['response2']
            rejected = json_item['response1']
        else:
            continue

        count += 1
        data, ground_truth = build_criteria_TTS_conversation(count, query, chosen, rejected)
        lang_train_data.append(data)
        
        if ground_truth == 0:
            ground_truth_0 += 1
        elif ground_truth == 1:
            ground_truth_1 += 1
    except:
        continue

print(ground_truth_0, ground_truth_1)
print(count, len(lang_train_data))
random.shuffle(lang_train_data)
table = pa.Table.from_pylist(lang_train_data)
pq.write_table(table, "/workspace/mnt/lxb_work/MindMirror/GRM-omni-train-v1/data/criteria_TTS/helpsteer_filted_train_data_lang.parquet")
