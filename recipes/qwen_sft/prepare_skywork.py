import json


with open("/workspace/mnt/lxb_work/Tau-omni/data/skywork_qwen32b_thinking_sft/sft.jsonl", 'w') as f:

    count = 0
    for line in open("/workspace/mnt/lxb_work/GRM-omni-save/pairwise/skywork/Qwen32-8B-thining-fennec_v0-prompt-pairwise_5/pairwise_judge.jsonl").readlines():

        json_item = json.loads(line)
        # import pdb; pdb.set_trace()
        # json_item['llm_response']
        if len(set(json_item['judge_list'])) <= 1:
            continue
        
        for idx, conv, judge in zip(range(3), json_item['conversations'], json_item['judge_list']):
            if (judge == "A" and json_item['answer'] == 0) or  (judge == "B" and json_item['answer'] == 1):
                
                if len(conv) == 2:
                    data = {"conversations": [
                        {"role": "user", "content": conv[0]['content']},
                        {"role": "assistant", "content": conv[1]['content']}
                    ]}
                else:
                    data = {"conversations": [
                        {"role": "user", "content": conv[0]['content']},
                        {"role": "assistant", "content": conv[idx + 1]['content']}
                    ]}
                count += 1
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
                break
    
    # 9162/39413
    print(count)
