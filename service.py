'''
Designing an API request service to call LLM providers' APIs.

- The goal of this code it to ALWAYS generate text for a given prompt, as fast as possible.

Requirements:
- The service needs to make asynchronous requests to the LLM provider's API.
- If an API request fails, the service should retry the request
- It should fall back to another LLM provider if the first one fails.
- Retry
- Needs to generate a response within a specified time limit, if it cannot,
  redundant requests should be made to other LLM providers.

Needs:
- 2 LLM providers and 2 models each for redundancy. (Mixtral & Llama + Gpt-3.5 & Gpt-4)

===
Based on the above description, I feel following classes/functions should be present
===

* LLMParams: All the parameters required to change model's sampling behaviour.
  - max_tokens: number
  - temperature: decimal
  - top_p: decimal
  - top_k: number
  - stop_tokens: List[str]

* LLM: All the details required to make a request to an LLM provider's API, along with the request method.
  - url: string
  - api_key: string
  - model: string
  - session: Optional
  - llm_params: LLMParams
  + _agenerate: async function
  + _generate: function which calls the API with normal requests
  + __call__: based on if a session is present, it should call _agenerate or _generate

* GenerationMaster: All the details required to get generations fast.
  - llms: List[LLM]  # list of redundancy
  - input_text_list: List[str]
  - num_workers: number
  - session: asyncio.ClientSession with timeout
  - max_retries: number
  - _responses: List[str]
  - _todo: queue of tasks  # asyncio.Queue
  - _pbar: progress bar  # tqdm
  + run: async function to run the generation
  + _worker: async function to process one task and quit when (helper function)
  + _process_one: async function to generate text and retry if fails

'''


import aiohttp
import asyncio
import os
from typing import Any, Dict, List

from .class_model import LLM, GenerationMaster

# together ai config
url = "https://api.together.xyz/v1/completions"
api_key = os.environ['TOGETHER_API_KEY']
NUM_WORKERS = int(os.environ.get('LLM_NUM_WORKERS', 2))

async def _generate(
  inputs: List[List[Dict[str, Any]]],
  llm: List[LLM],
  max_retries: int,
  total_timeout: int,
) -> List[str]:
  '''Helper function to create a session, and run the GenerationMaster.'''

  timeout = aiohttp.ClientTimeout(total=total_timeout)
  async with aiohttp.ClientSession(timeout=timeout) as session:
    generator_instance = GenerationMaster(
      llms=llm,
      llm_inputs=inputs,
      num_workers=NUM_WORKERS,
      session=session,
      max_retries=max_retries,
    )
    return await generator_instance.run()

def generate(
  inputs: List[List[Dict[str, Any]]],
  llm: List[LLM],
  max_retries: int,
  total_timeout: int,
) -> List[str]:

  # validate
  # inputs
  if not isinstance(inputs, list): raise ValueError('inputs should be a list of list of dictionaries.')
  for inp in inputs:
    if not isinstance(inp, list): raise ValueError('inputs should be a list of list of dictionaries.')
  inp_keys = []
  for inp in inputs[0]:
    if not isinstance(inp, dict): raise ValueError('inputs should be a list of list of dictionaries.')
    inp_keys.append(set(inp.keys()))

  # llms
  if not isinstance(llm, list): raise ValueError('llm should be a list of LLM objects.')
  for l in llm:
    if not isinstance(l, LLM): raise ValueError('llm should be a list of LLM objects.')

  # inputs and llms
  for inp in inputs:
    assert len(inp) == len(llm), 'Length of inputs and llms should be same, because llms may have different prompt templates.'
  for inp in inputs[1:]:
    for i, x in enumerate(inp):
      if not isinstance(x, dict): raise ValueError('inputs should be a list of list of dictionaries.')
      assert set(x.keys()) == inp_keys[i], 'All inputs for a particular llm should have same keys.'


  # run the generation
  return asyncio.run(_generate(
    inputs,
    llm=llm,
    max_retries=max_retries,
    total_timeout=total_timeout,
  ))

if __name__ == '__main__':
  # testing service
  from class_model import LLMParams
  llm_params = LLMParams(
    max_tokens=1024,
    temperature=0.7,
    top_p=0.7,
    top_k=50,
    stop_tokens=['Human:', '[/INST]', '</s>'],
  )
  llms = [
    LLM(
      url,
      api_key,
      'something',
      llm_params,
      '<s>[INST] {query} [/INST]'
    ),
    LLM(
      url,
      api_key,
      'mistralai/Mistral-7B-Instruct-v0.2',
      llm_params,
      '<s>[INST] {query} [/INST]'
    ),
  ]
  inputs = [
    {'query': 'What is the capital of France?'},
  ]
  print(
    '\n---\nFinal answer:',
    generate(inputs, llms, 3, 20)[0]
  )
  exit()

