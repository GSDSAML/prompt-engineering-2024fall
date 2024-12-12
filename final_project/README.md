# Final Project Folder for Prompt Engineering Course 2024 Fall

Topic: Financial QA - Agentic RAG with Multiple Tools

**File list**
- Final_Project_Skeleton_v1.ipynb: Skeleton code for the project
- db.zip: Compressed vector DB for financial reports (built on train.json in FinQA, so you should change the DB with the test.json in FinQA)
- test_db.zip: Compressed vector DB for financial reports, built on test.json in FinQA. You can use this file for the accuracy test.

# Documentation

## ChatOpenAI

OpenAI chat model integration.

### Arguments

- **model: str**

  - Name of OpenAI model to use.

- **temperature: float**

  - Sampling temperature.

- **max_tokens: Optional[int]**

  - Max number of tokens to generate.

- **api_key: Optional[str]**

  - OpenAI API key. If not passed in will be read from env var OPENAI_API_KEY.

### Example Usage

```
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=OPENAI_API_KEY)
```

## Tool

Make tools out of functions, can be used with or without arguments.

### Parameters

- name_or_callable – Optional name of the tool or the callable to be converted to a tool. Must be provided as a positional argument.
- runnable – Optional runnable to convert to a tool. Must be provided as a positional argument.
- return_direct – Whether to return directly from the tool rather than continuing the agent loop. Defaults to False.
- args_schema – optional argument schema for user to specify. Defaults to None.
- infer_schema – Whether to infer the schema of the arguments from the function’s signature. This also makes the resultant tool accept a dictionary input to its run() function. Defaults to True.
- response_format – The tool response format. If “content” then the output of the tool is interpreted as the contents of a ToolMessage. If “content_and_artifact” then the output is expected to be a two-tuple corresponding to the (content, artifact) of a ToolMessage. Defaults to “content”.
- parse_docstring – if infer_schema and parse_docstring, will attempt to parse parameter descriptions from Google Style function docstrings. Defaults to False.
- error_on_invalid_docstring – if parse_docstring is provided, configure whether to raise ValueError on invalid Google Style docstrings. Defaults to True.

### Example Usage

The @tool decorator is the simplest way to define a custom tool. The decorator uses the function name as the tool name by default, but this can be overridden by passing a string as the first argument. Additionally, the decorator will use the function's docstring as the tool's description - so a docstring **MUST** be provided.

```
from langchain_core.tools import tool

@tool
def rewrite2fiscal(question: str) -> tuple[str, int]:
  """Extract the company and convert relative time information in the question to the actual fiscal year of the company.
     It uses today function for the date information.

    Args:
        question: Question to be extracted.

    Returns:
        A tuple of ticker of the company and the fiscal year
  """
  ticker = extract_ticker_chain.invoke({"question": question}).ticker
  today = datetime.now().strftime("%Y-%m-%d")
  response = convert_chain.invoke({"original_question": question, "fiscal_info": get_fiscal_info_with_ticker(ticker), "today": today})
  fy = extract_fiscal_chain.invoke({'question': response}).fy
  return ticker, fy
```

### Pre-defined Tool List

- `rewrite2fiscal(question: str) -> tuple[str, int]`: Extract the company and convert relative time information in the `question` to the actual fiscal year of the company.
   - **Arguments**:
     - question: Question to be extracted.
   - **Returns**:
     - A tuple of ticker of the company and the fiscal year

- `calculate_eps(net_income: float, outstanding_shares: int)`: Calculate the EPS of the company using net income and outstanding share
   - **Arguments**:
      - net_income: Net income value of the company
      - outstanding_shares: Total stock held by the company's shareholders
   - **Returns**:
      - EPS value or None if there is no value for arguments

- `calculate_cashflowfromoperations(net_income: float, non_cash_items: float, changes_in_working_capital: float)`: Calculate the cash flow from operations of the company using net income, non cash items and change in working capital
   - **Arguments**:
      - net_income: Net income value of the company
      - non_cash_items: Financial transactions or events that are recorded in a company's financial statements but do not involve the exchange of cash
      - changes_in_working_capital: Difference in a company's working capital between two reporting periods
   - **Returns**:
      - Value of cash flow from operations

- `retrieve_factual_data(question:str, ticker: str, fy: int) -> str`: Search vector DB for the financial reports with the question and ticker and fiscal year
   - **Arguments**:
      - question: Question need to be answered
      - ticker: Ticker of the company for filtering the documents
      - fy: Fiscal year for filtering the documents
   - **Returns**:
      - A related document for the question.

## Agent with Tools

### bind_tools

`bind_tools(tools: Sequence[Union[Dict[str, Any], Type, Callable, BaseTool]], **kwargs: Any) → Runnable[LanguageModelInput, BaseMessage]`

- **Arguments**:
  - tools (Sequence[Union[Dict[str, Any], Type, Callable, BaseTool]])
  - kwargs (Any)
- **Returns**:
  - Runnable[LanguageModelInput, BaseMessage]

### Example Agent Usage

```
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

tools = [rewrite2fiscal, calculate_eps, calculate_cashflowfromoperations, retrieve_factual_data]
llm_with_tools = llm.bind_tools(tools)

system_query = 'You are the professional agent that can use tools to answer the financial question. Output answer without selecting tool if you can answer the question with chat history.'
user_query = """For the given question, you should decide which tool you should use and arguments for the tool to answer the question.
        Question: {question}"""

query = "What was the trend of revenue for Intel 10 years ago?"

messages = [SystemMessage(content=system_query), HumanMessage(content=user_query.format(question=query))]

ai_msg = llm_with_tools.invoke(messages)
print(ai_msg.content)
while ai_msg.content == '':
  messages.append(ai_msg)
  for tool_call in ai_msg.tool_calls:
    selected_tool = {"rewrite2fiscal": rewrite2fiscal,
                     "calculate_eps": calculate_eps,
                     "calculate_cashflowfromoperations": calculate_cashflowfromoperations,
                     "retrieve_factual_data": retrieve_factual_data}[tool_call["name"].lower()]
    tool_msg = selected_tool.invoke(tool_call)
    messages.append(tool_msg)
  ai_msg = llm_with_tools.invoke(messages)
  print(ai_msg.content)

for message in messages:
  print(message)
```
