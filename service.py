'''
This module contains the service code for generating outputs using different LLM service providers.
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
  """
  Generate outputs using the given inputs and LLM objects.

  Args:
    inputs (List[List[Dict[str, Any]]]): A list of lists of dictionaries representing the input data.
    llm (List[LLM]): A list of LLM objects.
    max_retries (int): The maximum number of retries for generating outputs.
    total_timeout (int): The total timeout in seconds for generating outputs.

  Returns:
    List[str]: A list of generated outputs.

  Raises:
    ValueError: If the inputs or llm are not in the expected format.

  """
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

