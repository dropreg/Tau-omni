import json
import random

def random_argmax(scores):
    max_score = max(scores)
    max_indices = [i for i, score in enumerate(scores) if score == max_score]
    return random.choice(max_indices)

PROMPT = """You are an objective, impartial, and unbiased content evaluator. Given a user query and two candidate assistant responses, produce a rigorous, evidence-based comparison and a single final verdict indicating which response better fulfills the user’s intent.\n\n\n###Output Format\nThe final verdict is [[A]] or [[B]]\n\n\n### Input:[User Question]:\n{query}\n\n[The Start of Assistant A's Answer]:\n{response_1}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]: \n{response_2}\n[The End of Assistant B's Answer]\nPlease output your analysis and final verdict:"""

count = 0
with open("/workspace/mnt/lxb_work/Tau-omni/data/helpsteer_qwen8b_convert_sft/sft.jsonl", 'w') as f:
    
    for line in open("/workspace/mnt/lxb_work/GRM-omni-save/rar_science/skywork-reward-v2-llama-no-thinking-gen-judge-bo8/gen_judge.jsonl"):
        
        json_item = json.loads(line)

        scores = []
        for response in json_item["llm_response"]:
            scores.append(json.loads(response.replace("```json", "").replace("```", ""))['rating'])

        max_index = random_argmax(scores)
        min_index = scores.index(min(scores))
        chosen = json_item["response_list"][max_index]
        rejected = json_item["response_list"][min_index]
        
        
        if random.random() > 0.5:
            prompt = PROMPT.format(query=json_item['paired_data']['query'],response_1=chosen,response_2=rejected)
            response = "The final verdict is [[A]]."
        else:
            prompt = PROMPT.format(query=json_item['paired_data']['query'],response_1=rejected,response_2=chosen)
            response = "The final verdict is [[B]]."

        data = {"conversations": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]}

        count += 1
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

print(count)
