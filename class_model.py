import asyncio
import aiohttp
import jinja2
import requests

from loguru import logger
from pydantic import BaseModel
from typing import Any, Dict, List, Optional, Union
from tqdm import tqdm


class LLMParams(BaseModel):
  max_tokens: int
  temperature: float
  top_p: float
  top_k: int
  stop: List[str]



class LLM:
  def __init__(
    self,
    url: str,
    api_key: str,
    model: str,
    llm_params: LLMParams,
    prompt_template: Union[str, jinja2.environment.Template],
    session: Optional[aiohttp.ClientSession]=None,
  ):
    self.url = url
    self.api_key = api_key
    self.model = model
    self.llm_params = llm_params
    self.prompt_template = prompt_template

    if session is None:
      self.generate = self._generate
    else:
      self.session = session
      self.generate = self._agenerate

  def __str__(self) -> str:
    return f"LLM: {self.model}. URL: {self.url}"

  async def _agenerate(self, inputs: Dict[str, Any]) -> str:
    data = self._prep_json(inputs)
    headers = {
      "accept": "application/json",
      "content-type": "application/json",
      "Authorization": "Bearer " + self.api_key
    }
    async with self.session.post(self.url, json=data, headers=headers) as response:
      self.response = response
      return response

  def _generate(self, inputs: Dict[str, Any], verbose=False) -> str:
    data = self._prep_json(inputs)
    if verbose:
      print('LLM Input>')
      if self.is_openai(): print(data['messages'][0]['content'])
      else: print(data['prompt'])

    headers = {
      "accept": "application/json",
      "content-type": "application/json",
      "Authorization": "Bearer " + self.api_key
    }
    res = requests.post(self.url, json=data, headers=headers)
    res.status = res.status_code
    self.response = res
    return res

  def _prep_json(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
    data = {
      "model": self.model,
      "n": 1,
      **self.llm_params.model_dump()
    }
    if self.is_openai():
      data['messages'] = [{'role': 'user', 'content': self._get_prompt(inputs)}]
      data.pop('top_k')
    else:
      data['prompt'] = self._get_prompt(inputs)
    return data
  
  def is_openai(self) -> bool:
    return 'openai' in self.url

  # TODO (rohan): Deprecate this method in v0.0.4
  def _get_prompt(self, inputs: Dict[str, Any]) -> str:
    if isinstance(self.prompt_template, str):
      # raise warning
      Warning("Using string as prompt_template is deprecated and removed in v0.0.4. Use jinja2.Template instead.")
      return self.prompt_template.format(**inputs)
    return self.prompt_template.render(**inputs)

  def get_text_from_response(self) -> str:
    if self.is_openai():
      return self.response.json()['choices'][0]['message']['content']
    return self.response.json()['choices'][0]['text']

class GenerationMaster:
  def __init__(
    self,
    llms: Union[LLM, List[LLM]],
    llm_inputs: List[Dict[str, Any]],
    num_workers: int,
    session: aiohttp.ClientSession,
    max_retries: int,
  ):

    # validate inputs
    if isinstance(llms, list):
      for llm in llms:
        if not isinstance(llm, LLM):
          raise ValueError("llms should be a list of LLM objects.")
    elif not isinstance(llms, LLM):
      raise ValueError("llms should be a LLM object.")

    if not isinstance(num_workers, int): raise ValueError("num_workers should be an integer.")
    if not isinstance(session, aiohttp.ClientSession): raise ValueError("session should be an aiohttp.ClientSession object.")
    if not isinstance(max_retries, int): raise ValueError("max_retries should be an integer.")

    # set attributes
    self.llms = llms if isinstance(llms, list) else [llms]
    self.llm_inputs = llm_inputs
    self.num_workers = min(num_workers, len(llm_inputs))
    self.session = session
    self.max_retries = max_retries

    # private attributes
    self._responses = []
    self._todo = asyncio.Queue()
    self._pbar = tqdm(total=len(llm_inputs), desc="Generating", leave=False)


  async def run(self) -> List[str]:
    outputs = []
    for inputs in self.llm_inputs: await self._todo.put(inputs)

    workers = [asyncio.create_task(self._worker()) for _ in range(self.num_workers)]
    await self._todo.join()
    for w in workers: w.cancel()
    
    for res in self._responses: outputs.append(res)
    return outputs

  async def _worker(self):
    while True:
      try: await self.process_one()
      except asyncio.CancelledError: return

  async def process_one(self):
    inputs = await self._todo.get()
    res = None
    raised_exc = None
    for _ in range(self.max_retries):
      for inp, llm in zip(inputs, self.llms):
        try:
          res = llm.generate(inp)
          if res.status == 200:
            res = res.json()
            self._responses.append(llm.get_text_from_response())
            self._pbar.update(1)
            self._todo.task_done()
            return
          else:
            raised_exc = res.text
            logger.warning(f'Failed to generate text using {llm}. Got status: {res.status}. Text: {res.text}')

        except aiohttp.ClientError as e:
          raised_exc = e
          logger.error(f'Client Error: {e}')
        except asyncio.TimeoutError as e:
          raised_exc = e
          logger.error(f'Timeout Error: {e}')
        except requests.JSONDecodeError as e:
          raised_exc = e
          logger.error(f'JSONDecodeError: {e} Got text: {res.text}')
        except Exception as e:
          raised_exc = e
          logger.error(f'Exception: {e}')


    self._pbar.update(1)
    self._todo.task_done()

    # code might not ever reach this
    if res is None: raise Exception(f"Got exception: {raised_exc}. Failed to generate text for inputs: {inputs}")
    else: self._responses.append(res)


