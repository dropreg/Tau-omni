import asyncio
import sys

from ms_agent import LLMAgent
from ms_agent.config import Config

async def run_query(query: str):
    config = Config.from_task('ms-agent/simple_agent')
    # TODO change to your real api key https://modelscope.cn/my/myaccesstoken
    config.llm.modelscope_api_key = 'xxx'
    engine = LLMAgent(config=config)

    _content = ''
    generator = await engine.run(query, stream=True)
    async for _response_message in generator:
        new_content = _response_message[-1].content[len(_content):]
        sys.stdout.write(new_content)
        sys.stdout.flush()
        _content = _response_message[-1].content
    sys.stdout.write('\n')
    return _content


if __name__ == '__main__':
    query = 'Introduce yourself'
    asyncio.run(run_query(query))
Copy code