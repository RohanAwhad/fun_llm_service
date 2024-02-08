import asyncio
import aiohttp
import jinja2
import requests

from loguru import logger
from pydantic import BaseModel
from typing import Any, Dict, List, Optional, Union
from tqdm import tqdm


class LLMParams(BaseModel):
  """
  Parameters for the LLM model.

  Attributes:
    max_tokens (int): The maximum number of tokens to generate.
    temperature (float): The temperature value for token generation.
    top_p (float): The cumulative probability threshold for token generation.
    top_k (int): The number of top tokens to consider for token generation.
    stop (List[str]): A list of stop words to terminate token generation.
  """
  max_tokens: int
  temperature: float
  top_p: float
  top_k: int
  stop: List[str]


class LLMResponse(BaseModel):
  """
  Represents a response from the LLM service.

  Attributes:
    status (int): The status code of the response.
    data (Optional[Dict[str, Any]]): The data associated with the response, if any.
    text (Optional[str]): The error message associated with the response, if any.
  """
  status: int
  data: Optional[Dict[str, Any]] = None
  text: Optional[str] = None


class LLM:
  """
  Language Model class for generating text using LLM service.

  Args:
    url (str): The URL of the LLM service.
    api_key (str): The API key for accessing the LLM service.
    model (str): The name of the language model to use.
    llm_params (LLMParams): The parameters for the LLM model.
    prompt_template (Union[str, jinja2.environment.Template]): The template for generating prompts.

  Methods:
    set_session(session: aiohttp.ClientSession): Sets the session for making asynchronous requests.
    generate(inputs: Dict[str, Any]) -> str: Generates text using the LLM model based on the session attribute.
    get_text_from_response(response: Optional[LLMResponse]=None) -> str: Gets the data or message from the LLM response based on the status code.
  """
  def __init__(
    self,
    url: str,
    api_key: str,
    model: str,
    llm_params: LLMParams,
    prompt_template: Union[str, jinja2.environment.Template],
  ):
    self.url = url
    self.api_key = api_key
    self.model = model
    self.llm_params = llm_params
    self.prompt_template = prompt_template
    self.session = None

  def __str__(self) -> str: return f"LLM: {self.model}. URL: {self.url}"
  def set_session(self, session: aiohttp.ClientSession): self.session = session
  def _is_openai(self) -> bool: return 'openai' in self.url

  def generate(self, inputs: Dict[str, Any]) -> str:
    return self._generate(inputs) if self.session is None else self._agenerate(inputs)

  async def _agenerate(self, inputs: Dict[str, Any]) -> LLMResponse:
    '''Generate text using asynchronous requests.'''
    data = self._prep_json(inputs)
    headers = {
      "accept": "application/json",
      "content-type": "application/json",
      "Authorization": "Bearer " + self.api_key
    }
    async with self.session.post(self.url, json=data, headers=headers) as response:
      status = response.status
      if status == 200: self.response = LLMResponse(status=status, data=await response.json())
      else: self.response = LLMResponse(status=status, text=await response.text())
      return self.response

  def _generate(self, inputs: Dict[str, Any], verbose=False) -> str:
    '''Generate text using synchronous requests.'''
    data = self._prep_json(inputs)
    if verbose:
      print('LLM Input>')
      if self._is_openai(): print(data['messages'][0]['content'])
      else: print(data['prompt'])

    headers = {
      "accept": "application/json",
      "content-type": "application/json",
      "Authorization": "Bearer " + self.api_key
    }
    res = requests.post(self.url, json=data, headers=headers)
    status = res.status_code
    if status == 200: self.response = LLMResponse(status=status, data=res.json())
    else: self.response = LLMResponse(status=status, text=res.text)
    return self.response

  def _prep_json(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepares the JSON data for the LLM model request.

    Args:
      inputs (Dict[str, Any]): The input data for the LLM model.

    Returns:
      Dict[str, Any]: The prepared JSON data.
    """
    data = {
      "model": self.model,
      "n": 1,
      **self.llm_params.model_dump()
    }
    if self._is_openai():
      data['messages'] = [{'role': 'user', 'content': self._get_prompt(inputs)}]
      data.pop('top_k')
    else:
      data['prompt'] = self._get_prompt(inputs)
    return data
  

  # TODO (rohan): Deprecate this method in v0.0.4
  def _get_prompt(self, inputs: Dict[str, Any]) -> str:
    """
    Get the prompt based on the inputs provided.

    Args:
      inputs (Dict[str, Any]): A dictionary containing the input values.

    Returns:
      str: The generated prompt string.
    """
    if isinstance(self.prompt_template, str):
      # raise warning
      Warning("Using string as prompt_template is deprecated and removed in v0.0.4. Use jinja2.Template instead.")
      return self.prompt_template.format(**inputs)
    return self.prompt_template.render(**inputs)

  # TODO (rohan): Deprecate support response arg = None in v0.0.4
  def get_text_from_response(self, response: Optional[LLMResponse]=None) -> str:
    """
    Retrieves the text from the LLMResponse object.

    Args:
      response (Optional[LLMResponse]): The LLMResponse object to retrieve the text from. If not provided, the method will use the stored response.

    Returns:
      str: The text extracted from the LLMResponse object.
    """
    if response is None:
      Warning("Using get_text_from_response without passing response is deprecated and removed in v0.0.4. Use response.json instead.")
      response = self.response

    if self._is_openai(): return response.data['choices'][0]['message']['content']
    return response.data['choices'][0]['text']

class GenerationMaster:
  """
  A class that manages the generation of text using LLM models.

  Args:
    llms (Union[LLM, List[LLM]]): The LLM object(s) to use for text generation.
    llm_inputs (List[Dict[str, Any]]): The inputs for text generation.
    num_workers (int): The number of worker tasks to use for parallel generation.
    session (aiohttp.ClientSession): The aiohttp client session to use for HTTP requests.
    max_retries (int): The maximum number of retries for generating text.

  """

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

    # set sessions for llms
    for llm in self.llms: llm.set_session(session)


  async def run(self) -> List[str]:
    """
    Runs the text generation process.

    Returns:
      List[str]: The generated text outputs.
    """
    outputs = []
    for inputs in self.llm_inputs: await self._todo.put(inputs)

    workers = [asyncio.create_task(self._worker()) for _ in range(self.num_workers)]
    await self._todo.join()
    for w in workers: w.cancel()
    
    for res in self._responses: outputs.append(res)
    return outputs

  async def _worker(self):
    """
    Worker task for processing text generation.

    This task continuously processes inputs from the queue until all inputs are processed or the task is cancelled.
    """
    while True:
      try: await self.process_one()
      except asyncio.CancelledError: return

  async def process_one(self):
    """
    Processes a single input for text generation.

    This method attempts to generate text using the LLM models and handles retries and exceptions.
    """
    inputs = await self._todo.get()
    res = None
    raised_exc = None
    for _ in range(self.max_retries):
      for inp, llm in zip(inputs, self.llms):
        try:
          '''
          attributes needed of res:
          - status
          - json()
          - text
          '''
          res = await llm.generate(inp)
          if res.status == 200:
            self._responses.append(llm.get_text_from_response(res))
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


