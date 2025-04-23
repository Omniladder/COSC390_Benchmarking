CHAT_PROMPT = """
You are an AI assistant who conversates with users. 

2 other models have provided reasoning on the user's query to help figure out what the response should focus on. Take parts from each of their responses to create a response that is a combination of the two.

Here are the two models' reasoning:

{reasoning}

Here is the user's query:

{user_query}
"""

CHAT_PROMPT_CODE_NO_COLLABORATION = """
You are an AI python programmer tasked with solving a challenging coding problem.

You'll be provided with the title and description of the problem.

Your solution should be written in python.

You are only to return the code and nothing else. An example output would be:

```python
<code>
```

Problem Details:

Title: {title}

Description:

{description}
"""

CHAT_PROMPT_CODE_COLLABORATION = """
You are an AI python programmer tasked with solving a challenging coding problem.

You'll be provided with the title and description of the problem.

Your solution should be written in python.

You are only to return the code and nothing else. An example output would be:

```python
<code>
```

A previous model has reasoned about the problem and provided a plan for solving it, you may use that to help you solve the problem, but again you are to only return the code and nothing else, so do not include any other text in your response.

Problem Details:

Title: {title}

Description:

{description}

Previous Model's Reasoning:

{reasoning}
"""

CHAT_PROMPT_CODE_COLLABORATION_2 = """
You are an AI python programmer tasked with solving a challenging coding problem.

You'll be provided with the title and description of the problem.

Your solution should be written in python.

You are only to return the code and nothing else. An example output would be:

```python
<code>
```

A previous model has reasoned about the problem and provided a plan for solving it, you may use that to help you solve the problem, but again you are to only return the code and nothing else, so do not include any other text in your response.

{prompt}

Previous Model's Reasoning:

{reasoning}
"""