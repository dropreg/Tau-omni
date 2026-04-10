import json


def build_pairwise_judge_prompt_fennec_v0(context, query, response_1, response_2):
    
    return f"""You are an exceptionally rigorous, objective, and logical AI Content Evaluator. Your mission is to conduct a deep-dive, evidence-based comparison between two assistant responses and deliver a definitive verdict on which response better fulfills the user's intent.

    ### Core Evaluation Principles (Strict Adherence Required)
    1.  **Evidence-Based Judgment**: Every claim must be supported by specific textual evidence from the provided responses. If a response contains hallucinations, factual errors, or misses instructions, you must pinpoint them exactly.
    2.  **Comparative Analysis**: In your feedback, focus on the *relative* performance (e.g., why Response 1 is superior to Response 2 in a specific area) rather than just listing features in isolation.
    3.  **Instruction Primacy**: Prioritize explicit constraints provided in the user's query (e.g., word count, formatting, tone, or technical requirements).
    4.  **Zero External Bias**: Rely solely on the provided input. Do not bring in outside knowledge or personal interpretations of what the user "might" have meant beyond the text.

    ### Rubric Definition
    You must dynamically generate three **mutually exclusive and independent** evaluation rubrics based on the nature of the User Query.

    # Output Format
    <rubrics>
        <rubric>[Name]: [Description of what this rubric measures and why it matters for this specific query].
        <feedback_1>[Analysis for Assistant 1. Must include: (a) strengths with evidence, (b) flaws/omissions with evidence, and (c) the resulting impact on this score.]<\feedback_1>
        <feedback_2>[Analysis for Assistant 2. Must include: (a) strengths with evidence, (b) flaws/omissions with evidence, and (c) the resulting impact on this score.]<\feedback_2><rubric>
        <rubric>...
        <feedback_1>...<\feedback_1>
        <feedback_2>...<\feedback_2><rubric>
        <rubric>...
        <feedback_1>...<\feedback_1>
        <feedback_2>...<\feedback_2><rubric>
        <judge>A synthesis of the three rubrics explaining the 'Why' behind the winner. Highlight any 'deal-breakers' (e.g., factual errors or failed constraints) that heavily influenced the decision. The final verdict is [[1]] or [[2]]</judge>
    </rubrics>
    ---

    ### Input Data
    [History Context]: 
    {context}

    [User Query]: 
    {query}

    [Assistant 1's Response]:
    {response_1}

    [Assistant 2's Response]:
    {response_2}

    ---
    ### Adjudication Start
    Provide your analysis and final verdict:"""

with open("/workspace/mnt/lxb_work/Tau-omni/data/helpsteer_qwen8b_convert_sft/sft.jsonl", 'w') as f:
    
    count = 0
    for line in open("/workspace/mnt/lxb_work/GRM-omni-save/fennec/qwen32-8b-preference/direct_synth.jsonl").readlines():
        
        json_item = json.loads(line)
        
        if ("[[1]]" in json_item['conversations'][0][1]['content'] and json_item['answer'] == 0) or ("[[2]]" in json_item['conversations'][0][1]['content'] and json_item['answer'] == 1):
            
            prompt = build_pairwise_judge_prompt_fennec_v0(json_item['paired_data']['context']['text_content'], json_item['paired_data']['query']['text_content'], json_item['paired_data']['response_1']['text_content'], json_item['paired_data']['response_2']['text_content'])
            
            data = {"conversations": [
                {"role": "user", "content": prompt},
                json_item['conversations'][0][1]
            ]}

            count += 1
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    
    # 9162/39413
    print(count)
