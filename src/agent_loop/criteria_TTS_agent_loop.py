# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import copy
import json
import logging
import os
from enum import Enum
from typing import Any, Optional
from uuid import uuid4
import ray
import requests
import torch
import io
import base64
import random

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register, AgentLoopWorkerBase, _InternalAgentLoopOutput
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.interactions.base import BaseInteraction
from verl.interactions.utils.interaction_registry import initialize_interactions_from_config
from verl.tools.schemas import ToolResponse
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AgentState(Enum):

    PENDING = "pending"
    GENERATING = "generating"
    TERMINATED = "terminated"
    INTERACTING = "interacting"


class AgentData:
    """Encapsulates all state variables for the agent loop."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        instance_id: str,
        request_id: str,
        golden_rubric: list[str],
        response_list: list[str],
        interaction: Optional[BaseInteraction] = None,
        interaction_kwargs: Optional[dict[str, Any]] = None,
        markov_messages: dict = {},
        image_data: [] = [],
    ):

        self.messages = messages
        self.instance_id = instance_id
        self.request_id = request_id
        self.golden_rubric = golden_rubric
        self.response_list = response_list

        self.interaction = interaction
        self.interaction_kwargs = interaction_kwargs or {}
        
        self.access_index = -1
        self.markov_messages = markov_messages
        self.markov_judge = []
        self.markov_rubric = []
        self.image_data = image_data
        
        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []

        self.metrics = {}
        self.user_turns = 0
        self.assistant_turns = 0


@register("criteria_TTS")
class CriteriaTTSAgentLoop(AgentLoopBase):

    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level RewardZeroAgentLoop initialization")

        # Initialize tools from config file
        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.max_user_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns
        cls.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
        cls.max_parallel_calls = config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls
        
        cls.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length

        cls.interaction_config_file = config.actor_rollout_ref.rollout.multi_turn.interaction_config_path
        if cls.interaction_config_file:
            cls.interaction_map: dict[str, BaseInteraction] = cls._initialize_interactions(cls.interaction_config_file)

    def build_markov_message(self, interaction_mode, interaction_kwargs):
        
        if interaction_mode == "meta_reward" or interaction_mode == "grm" or interaction_mode == "meta_reward_golden":
            _markov_messages = {
                "response": ""
            }
        else:
            raise Exception(f"Not Support This Mode {interaction_mode}")
        
        return _markov_messages

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:

        messages = list(kwargs["raw_prompt"])
        request_id = uuid4().hex
        instance_id = kwargs['extra_info']['idx']

        interaction_kwargs = kwargs['extra_info']['interaction_kwargs']
        interaction_name = interaction_kwargs["name"]
        interaction_mode = interaction_kwargs["mode"]
        response_list = interaction_kwargs["response_list"]
        interaction = self.interaction_map[interaction_name]
        assert len(response_list) == len(interaction_kwargs['ground_truth_list'])

        if "golden_rubric" in interaction_kwargs:
            golden_rubric = interaction_kwargs['golden_rubric']
        else:
            golden_rubric = []
        
        await interaction.start_interaction(request_id)
        
        _markov_messages = self.build_markov_message(interaction_mode, interaction_kwargs)
        agent_data = AgentData(
            messages=messages,
            instance_id=instance_id,
            request_id=request_id,
            golden_rubric=golden_rubric,
            response_list=response_list,
            interaction=interaction,
            interaction_kwargs=interaction_kwargs,
            markov_messages=_markov_messages,
        )
        
        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, interaction_kwargs, interaction_mode, sampling_params)
            elif state == AgentState.INTERACTING:
                state = await self._handle_interacting_state(agent_data, interaction_mode)
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED
        
        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        
        # multi_modal_data = {"image": agent_data.image_data} if agent_data.image_data is not None else {}
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            multi_modal_data={},
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            response_logprobs=agent_data.response_logprobs[: self.response_length]
            if agent_data.response_logprobs
            else None,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics={},
            extra_fields={},
        )
        
        self._del_rubric_list(agent_data)
        if len(agent_data.markov_judge) != 1:
            print("[debug] some error happend... gent_data.markov_judge is null !!!")
            agent_data.markov_judge = [(0, 0, 0)]
        
        output.extra_fields.update({"access_index": agent_data.access_index})
        output.extra_fields.update({"markov_judge": agent_data.markov_judge})
        output.extra_fields.update({"ground_truth": interaction_kwargs['ground_truth']})
        output.extra_fields.update({"ground_truth_list": interaction_kwargs['ground_truth_list']})
        output.extra_fields.update({"interaction_mode": interaction_mode})
        return output
    
    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        
        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    agent_data.messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            model_inputs = self.processor(text=[raw_prompt], return_tensors="pt")
            agent_data.prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            agent_data.prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    agent_data.messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )
        return AgentState.GENERATING

    def _del_rubric_list(self, agent_data):
        resp = requests.delete(f"http://127.0.0.1:8000//delete/{agent_data.instance_id}", proxies={"http": None, "https": None})

    async def _handle_generating_state(
        self, agent_data: AgentData, interaction_kwargs: dict, interaction_mode: str, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:

        add_messages: list[dict[str, Any]] = []
        with simple_timer("generate_sequences", agent_data.metrics):
            _sync_output = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params
            )
            output = _sync_output

        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs
    
        # Check termination conditions
        if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
            return AgentState.TERMINATED
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            return AgentState.TERMINATED
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            return AgentState.TERMINATED
        
        if self.interaction_config_file:
            assistant_message = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
            )
            add_messages.append({"role": "assistant", "content": assistant_message})
            agent_data.messages.extend(add_messages)

        if self.interaction_config_file:
            return AgentState.INTERACTING
        else:
            return AgentState.TERMINATED

    async def _handle_interacting_state(self, agent_data: AgentData, interaction_mode: str) -> AgentState:
        """Handle the interacting state: get user input from interaction."""
        
        should_terminate_sequence, user_msg, reward, content = await agent_data.interaction.generate_response(agent_data, interaction_mode, agent_data.messages, agent_data.markov_messages, **agent_data.interaction_kwargs)
        
        if content is not None:
            agent_data.markov_rubric.append(content)
        if reward is not None:
            agent_data.markov_judge.append(reward)

        if user_msg is not None:
            
            add_messages: list[dict[str, Any]] = [user_msg]
            agent_data.messages.extend(add_messages) ##[change]
            agent_data.user_turns += 1
            _run_msg = add_messages
            
            if self.processor is not None:
                raw_user_response = await self.loop.run_in_executor(
                    None,
                    lambda: self.processor.apply_chat_template(
                        _run_msg,
                        add_generation_prompt=True,
                        tokenize=False,
                        **self.apply_chat_template_kwargs,
                    ),
                )
                model_inputs = self.processor(text=[raw_user_response], return_tensors="pt")
                response_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
            else:
                response_ids = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.apply_chat_template(_run_msg, add_generation_prompt=True, tokenize=True,**self.apply_chat_template_kwargs),
                )

            agent_data.prompt_ids += response_ids
            agent_data.response_mask += [0] * len(response_ids)
            if agent_data.response_logprobs:
                agent_data.response_logprobs += [0.0] * len(response_ids)

        if should_terminate_sequence:
            return AgentState.TERMINATED
        else:
            return AgentState.GENERATING

    @classmethod
    def _initialize_interactions(cls, interaction_config_file):
        """Initialize interactions from configuration.
        Returns:
            dict[str, BaseInteraction]: A dictionary mapping interaction names to interaction instances.
        """
        if interaction_config_file is None:
            return {}

        interaction_map = initialize_interactions_from_config(interaction_config_file)
        logger.info(f"Initialize interactions from configuration: interaction_map: {list(interaction_map.keys())}")
        return interaction_map
