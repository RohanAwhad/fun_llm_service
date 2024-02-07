Designing an API request service to call LLM providers' APIs.


- The goal of this code it to ALWAYS generate text for a given prompt, as fast as possible.

---

### Currently supports following providers:
- Together AI
- OpenAI

---


Requirements:
- The service needs to make asynchronous requests to the LLM provider's API.
- If an API request fails, the service should retry the request
- It should fall back to another LLM provider if the first one fails.
- Retry
- Needs to generate a response within a specified time limit, if it cannot,
  redundant requests should be made to other LLM providers.

Needs:
- 2 LLM providers and 2 models each for redundancy. (Mixtral & Llama + Gpt-3.5 & Gpt-4)

---
Based on the above description, I feel following classes/functions should be present [Out of Sync with implementation]


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
  - llm_params: LLMParams
  + session: Optional[asyncio.ClientSession]
  + set_session: function
  + _agenerate: async function
  + _generate: function which calls the API with normal requests
  + generate: based on if a session is present, it should call _agenerate or _generate

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
