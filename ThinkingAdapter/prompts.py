from langchain_core.prompts import ChatPromptTemplate


reason_prompt = ChatPromptTemplate.from_template("""
    You are a reasoning model. Your task is to suggest specific logic, concepts, or tools that could support solving the problem or refining the approach. Your suggestions should guide thinking, not prescribe exact actions.

    Keep your response short (1–3 sentences) and do not repeat previous points.

    You will be given:
    - A **prompt** to reason about.
    - A **topic** to focus on.
    - Existing **reasoning** to build upon.

    Based on the information below, what concept, technique, or consideration could help next?

    Prompt:
    {prompt}

    Topic:
    {topic}

    Reasoning so far:
    {reason}

    Suggested next idea:
""")
topic_prompt = ChatPromptTemplate.from_template("""
You are an expert at generating short and relevant research topics.

You will receive a prompt and a list of existing topics. Your task is to suggest **one new topic**, using **only 1–2 words**, that gives a fresh angle.

Do not explain or elaborate. Just return the topic, nothing else.

**Prompt:**
{prompt}

**Existing Topics:**
{topics}

**New Topic (1–2 words only):**
""")

output_prompt = ChatPromptTemplate.from_template("""
Use the following information to inform your answer, but respond as if you're solving it directly without explaining the steps.

Context:
{reason}

Question:
{prompt}
""")
