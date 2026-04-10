
from typing import Any, Optional
from uuid import uuid4
import logging
import os
import re
import random
import requests
import torch
import io
import base64
import random

from verl.interactions.base import BaseInteraction


logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class CriteriaTTSInteraction(BaseInteraction):

    def __init__(self, config: dict):
        super().__init__(config)

    async def start_interaction(
        self, instance_id: Optional[str] = None, **kwargs
    ) -> str:

        if instance_id is None:
            instance_id = str(uuid4())
        return instance_id

    def build_select_message(self):
        
        select_prompt="""### You are a Strategic Evaluation Architect. Your task is to analyze a query and two candidate responses to select the single most discriminative rubric from a provided list.

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
        
        user_msg = {
            "role": "user",
            "content": select_prompt
        }
        return user_msg

    def build_select_message_v2(self, response_2):
        
        select_prompt= f"""### You are a Strategic Evaluation Architect.

        ### Objective: Analyze the newly provided Response B against the existing Query and Response A. Select the single most discriminative rubric from the provided list that best exposes the "Critical Quality Gap" between the two candidates.

        ### Selection Criteria:
        - Core Intent Alignment: The rubric must target the primary intent or critical constraints of the Query.
        - Delta Maximization: Choose the rubric where the "Tips" (1-4) create the widest scoring gap between A and B based on their observed performance.
        - Decisive Actionability: The rubric must provide unambiguous "decision rules" that allow a judge to declare a clear winner without subjective doubt.

        ### Input Data:
        <Response B>: {response_2} </Response B>

        ### Output Format:
        Selected Rubric from previous rubric list: <rubric>[Insert the full, original tag and content here]</rubric>"""
        
        user_msg = {
            "role": "user",
            "content": select_prompt
        }
        return user_msg
    
    def build_judge_message(self):
        
        judge_prompt="""You are a Precision Evaluation Engine. Your task is to perform a rigorous, rubric-grounded comparison between Response A and Response B. You must act as a neutral judge, using the **provided rubric** as the sole legal framework for your decision.

        ### Evaluation Protocol
        1. **Evidence Extraction:** Scan both responses to identify specific segments, phrases, or omissions that trigger the "Tips" within the Rubric.
        2. **Tip-Grounded Comparison:** For every claim, you must explicitly link it to a specific Tip (e.g., "Violates Tip 1" or "Satisfies Tip 2").
        3. **Scoring Scale (1-5):** Assign a score to each response based on the following:
            - **5 (Exceptional):** Perfect adherence to all tips; fully embodies "Superior Patterns" with no flaws.
            - **4 (Good):** Strong adherence; minor omissions or slight stylistic issues that do not affect overall utility.
            - **3 (Satisfactory):** Meets basic requirements but misses some "Superior Patterns" or exhibits minor "Failure Patterns."
            - **2 (Poor):** Significant violations of tips; lacks required depth, clarity, or accuracy.
            - **1 (Major Flaws/Failure):** Fails to address the core query or violates almost all tips; provides little to no value.

        ---

        ### Output Format
        (Produce the analytical reasoning first, followed by the fixed-label results.)

        ### Analysis Section
        #### Reasoning Step 1: Evidence Analysis
        - **Evidence for A:** [Direct reference vs Tip ID]
        - **Evidence for B:** [Direct reference vs Tip ID]

        #### Reasoning Step 2: Final Synthesis
        [Explain the decisive "delta" and why one candidate outperformed the other based on the rubric.]

        ---

        ### Final Results Section
        (Ensure the following labels appear exactly as shown at the end of your response)
        SCORE_A: [Integer 1-5]
        SCORE_B: [Integer 1-5]
        VERDICT: [[A]] or [[B]]"""
        
        user_msg = {
            "role": "user",
            "content": judge_prompt
        }
        return user_msg
        
    def build_judge_message_v2(self):
        
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
        
        user_msg = {
            "role": "user",
            "content": judge_prompt
        }
        return user_msg



    def split_into_three_parts(self, text: str):

        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        criteria_match = re.search(
            r"<Rubric-Judge>(.*?)(?=<Judge A>)",
            text,
            flags=re.DOTALL
        )
        judge_a_match = re.search(
            r"<Judge A>(.*?)(?=</Judge A>)",
            text,
            flags=re.DOTALL
        )
        judge_b_match = re.search(
            r"<Judge B>(.*?)(?=</Judge B>)",
            text,
            flags=re.DOTALL
        )
        criteria = criteria_match.group(1).strip() if criteria_match else ""
        judge_a = judge_a_match.group(1).strip() if judge_a_match else ""
        judge_b = judge_b_match.group(1).strip() if judge_b_match else ""

        return criteria, judge_a, judge_b

    def parse(self, think_content, single=True):
        
        results = {"rubric_judge": [], "verdict": None}
        try:
            if single:
                criteria_matches = re.findall(r"<Rubric-Judge>(.*?)</Rubric-Judge>", think_content, re.S)
            else:
                criteria_matches = re.findall(r"<Rubric-Judge\s*\d+>(.*?)</Rubric-Judge\s*\d+>", think_content, re.S)
            
            for crit in criteria_matches:
                crit_before_A = crit.split("<Judge A>")[0].strip()
                # 提取 Judge A
                judge_A_match = re.search(r"<Judge A>(.*?)</Judge A>", crit, re.S)
                judge_A = judge_A_match.group(1).strip() if judge_A_match else None

                # 提取 Judge B
                judge_B_match = re.search(r"<Judge B>(.*?)</Judge B>", crit, re.S)
                judge_B = judge_B_match.group(1).strip() if judge_B_match else None
                
                criterion_dict={
                    "criterion": crit_before_A,
                    "judge_A": judge_A,
                    "judge_B": judge_B,
                }

                if criterion_dict["criterion"] is None or criterion_dict['judge_A'] is None or criterion_dict['judge_B'] is None:
                    raise Exception(f"Parse Criterion failed. Text:[{crit}]. will skip it.")
                
                results["rubric_judge"].append(criterion_dict)
            assert len(results["rubric_judge"]) > 0
        except Exception as e:
            criteria, judge_a, judge_b = self.split_into_three_parts(think_content)
            results["rubric_judge"].append({
                "criterion": criteria,
                "judge_A": judge_a,
                "judge_B": judge_b,
            })

        pred = None
        pattern = r'\[{1,2}([abcABC])\]{1,2}'
        matches = re.findall(pattern, think_content)
        if len(matches) > 0:
            pred = matches[-1]
        else:
            match = re.search(r"<Final Verdict>(.*?)</Final Verdict>", think_content)
            pred = match.group(1) if match else None
        
        if isinstance(pred, str):
            pred = pred.upper()
        results["verdict"] = pred

        return results

    async def generate_response(
        self, agent_data, interaction_mode: str, messages: list[dict[str, Any]], markov_messages, **kwargs
    ) -> tuple[bool, str, float, dict]:
        
        instance_id = agent_data.request_id
        if interaction_mode == "grm":
            return await self.generate_grm_response(instance_id, messages, markov_messages, **kwargs)
        elif interaction_mode == "meta_reward_golden":
            return await self.generate_meta_reward_golden_response(agent_data, messages, markov_messages, **kwargs)
        elif interaction_mode == "meta_reward":
            return await self.generate_meta_reward_response(agent_data, messages, markov_messages, **kwargs)
        else:
            raise Exception(f"Not Support This Mode {interaction_mode}")

    async def generate_grm_response(self, instance_id, messages, markov_messages, **kwargs):
        
        content = ""
        for i in range(len(messages) - 1, -1, -1):
            item = messages[i]
            if item.get("role") == "assistant":
                content = item.get("content")
                break
        
        raw_query = kwargs['query']
        print(f"[debug] process messages {len(messages)}....")

        breakpoint()

        if len(messages) == 4:
            user_msg = self.build_judge_message()
            return False, user_msg, None
        elif len(messages) == 6:
            return True, None, await self.calculate_judge_score(content, kwargs['ground_truth'])
        else:
            raise Exception("runtime error!")
    
    async def generate_meta_reward_golden_response(self, agent_data, messages, markov_messages, **kwargs):
        
        content = ""
        for i in range(len(messages) - 1, -1, -1):
            item = messages[i]
            if item.get("role") == "assistant":
                content = item.get("content")
                break
        
        raw_query = kwargs['query']
        if len(messages) == 2:
            
            access_index, _ = self._sync_rubric_list(agent_data, agent_data.request_id)
            agent_data.access_index = access_index
            user_msg = self.build_select_message_v2(agent_data.response_list[agent_data.access_index])
            # should_terminate user_msg, reward, content
            return False, user_msg, None, None
        
        elif len(messages) == 4:
            user_msg = self.build_judge_message_v2()
            # should_terminate user_msg, reward, content
            return False, user_msg, None, content
        
        elif len(messages) == 6:

            return True, None, await self.calculate_judge_score_v2(content, kwargs['ground_truth'], kwargs['ground_truth_list'][agent_data.access_index]), None
        else:
            raise Exception("runtime error!")

    def get_candidate_rubric(self):
        return [
            "Clarity: Measures how clear and understandable the response is, including sentence structure, logical flow, and avoidance of ambiguity.",
            "Accuracy: Assesses whether the information in the response is factually correct and consistent with known knowledge or data.",
            "Relevance: Evaluates how well the response addresses the question or prompt, without deviating into unrelated content.",
            "Accuracy: Assesses whether the information in the response is factually correct and consistent with known knowledge or data.",
            "Completeness: Checks if the response covers all major aspects of the question or task, without missing important points.",
            "Conciseness: Measures whether the response communicates its message without unnecessary repetition or verbosity.",
            "Coherence: Evaluates the logical consistency of ideas within the response and how well the sentences and paragraphs connect.",
            "Engagement: Assesses how interesting or engaging the response is, including tone, style, and ability to hold the reader’s attention.",
            "Coherence: Evaluates the logical consistency of ideas within the response and how well the sentences and paragraphs connect.",
            "Creativity: Measures the originality or innovativeness of the ideas or solutions presented in the response.",
            "Politeness / Tone: Checks whether the response maintains an appropriate, respectful, and professional tone.",
        ]
    
    def _sync_rubric_list(self, agent_data, golden_idx=-1):
        
        resp_raw = requests.post(
            "http://127.0.0.1:8000/sync",
            json={"id": str(agent_data.instance_id), "golden_idx": golden_idx},
            proxies={"http": None, "https": None}
        )
        resp = resp_raw.json()
        return resp['access_index'], resp['golden_idx']
    
    async def generate_meta_reward_response(self, agent_data, messages, markov_messages, **kwargs):

        instance_id = agent_data.request_id
        content = ""
        for i in range(len(messages) - 1, -1, -1):
            item = messages[i]
            if item.get("role") == "assistant":
                content = item.get("content")
                break
        
        raw_query = kwargs['query']
        if random.random() > 0.8:
            print(f"[debug] process messages {len(messages)}....")
        
        if len(messages) == 2:
            try:
                rubric_list = content.replace("<think>", "").replace("</think>", "").split("\n")
                rubric_list = [r for r in rubric_list if r]
                if random.random() > 0.99:
                    print(f"[debug] access_index=> {agent_data.access_index} data =>{rubric_list[agent_data.access_index]}")
                user_msg = self.build_judge_part_message(rubric_list[agent_data.access_index])
            except Exception as e:
                print(f"[error] {e}")
                try:
                    user_msg = self.build_judge_part_message(self.get_candidate_rubric()[agent_data.access_index])
                except:
                    user_msg = self.build_judge_part_message(self.get_candidate_rubric()[random.randint(0, 9)])
            # user_msg = self.build_judge_message()
            return False, user_msg, None, False

        elif len(messages) == 6:
            
            if markov_messages['criteria_a'] == '' and markov_messages['criteria_b'] == '':
                parse_result = self.parse(content)
                
                criteria_a = f"{parse_result["rubric_judge"][0]['criterion']}\n<Judge>{parse_result["rubric_judge"][0]["judge_A"]}</Judge>"
                criteria_b = f"{parse_result["rubric_judge"][0]['criterion']}\n<Judge>{parse_result["rubric_judge"][0]["judge_B"]}</Judge>"

                markov_messages['criteria_a'] = criteria_a
                markov_messages['criteria_b'] = criteria_b
                
            if markov_messages['current_response'] == "":

                user_msg = self.bulid_correct_message(raw_query, markov_messages['response_a'], markov_messages['criteria_a'])

                return False, user_msg, await self.calculate_judge_score(content, kwargs['ground_truth']), True

            elif len(markov_messages['revised_response_a']) == 1 and len(markov_messages['revised_response_b']) == 1:
                user_msg = self.bulid_correct_message(raw_query, markov_messages['response_b'], markov_messages['criteria_b'])

                markov_messages['revised_response_a'].append({
                    "role": "assistant",
                    "content": markov_messages['current_response']
                })
                if random.random() > 0.99:
                    print(f"[debug] revised response a {markov_messages['current_response'][:20]}")
                return False, user_msg, None, True
            else:
                assert len(markov_messages['revised_response_b']) == 1 and len(markov_messages['revised_response_a']) == 2
                
                markov_messages['revised_response_b'].append({
                    "role": "assistant",
                    "content": markov_messages['current_response']
                })
                if random.random() > 0.99:
                    print(f"[debug] revised response b {markov_messages['current_response'][:20]}")
                return True, None, None, True
        else:
            raise Exception("runtime error!")

    async def calculate_judge_score(self, content, answer) -> float:
        def extract_omni_answer(response: str, answer: int):
            
            assert type(answer) == int, f"answer shoule be a interger. not [{type(answer)}]"
            assert answer in [0, 1], f"answer shoule be 0 or 1, not [{answer}]"

            pred = None
            pattern = r'\[{1,2}([abcABC])\]{1,2}'
            matches = re.findall(pattern, response)
            if len(matches) > 0:
                pred = matches[-1]
            else:
                pred = None
            if isinstance(pred, str):
                pred = pred.upper()
            
            correct = False
            if pred == 'A' and answer == 0:
                correct = True
            elif pred == 'B' and answer == 1:
                correct = True
            elif pred == 'C' and answer == 2:
                correct = True
            
            return correct, pred
        
        correct, pred = extract_omni_answer(content, answer)
        
        if random.random() > 0.95:
            print(f"[debug] judge the answer right or not: [{correct}], the right answer is [{answer}], the model judge answer is [{pred}]")

        if pred == "A":
            return 0
        elif pred == "B":
            return 1
        else:
            return -1
    
    async def calculate_judge_score_v2(self, content, answer_1, answer_2) -> float:

        def extract_omni_answer(response: str):

            pred = None
            pattern = r'\[{1,2}([abcABC])\]{1,2}'
            matches = re.findall(pattern, response)
            if len(matches) > 0:
                pred = matches[-1]
            else:
                pred = None
            if isinstance(pred, str):
                pred = pred.upper()
            
            return pred
        
        try:
            score_a = re.search(r"SCORE_A:\s*(\d+)",content).group(1)
            score_b = re.search(r"SCORE_B:\s*(\d+)", content).group(1)
            final_result = True
            final_reward = 0.5
        except Exception as e:
            print(f"SCORE PARESE ERROR... {e}")
            score_a = -1
            score_b = -1
            final_reward = 0

        pred = extract_omni_answer(content)
        if pred == "A" and answer_1 > answer_2:
            correct = True
            final_reward += 0.5
        elif pred == "B" and answer_1 < answer_2:
            correct = True
            final_reward += 0.5
        else:
            correct = False

        if random.random() > 0.95:
            print(f"[debug] judge the answer right or not: [{correct}], the right answer is [{answer_1}, {answer_2}], the model judge answer is [{pred}]")
        
        if final_reward == 1:
            return (final_reward, score_a, score_b)
        else:
            return (final_reward, 0, 0)
