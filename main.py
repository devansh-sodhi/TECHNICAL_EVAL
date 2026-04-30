import os
from openai import OpenAI, AsyncOpenAI
from agents import Agent, Runner, set_default_openai_client, set_default_openai_api, set_tracing_disabled
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

async_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

set_default_openai_client(async_client)
set_default_openai_api("chat_completions")
set_tracing_disabled(True)

agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant.",
    model="openai/gpt-oss-120b",
)


def chat(prompt, model="openai/gpt-oss-120b"):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def run_agent(prompt, history=None):
    input = history + [{"role": "user", "content": prompt}] if history else prompt
    result = Runner.run_sync(agent, input)
    return result.final_output, result.to_input_list()


if __name__ == "__main__":
    print(chat("Say hello in one sentence."))

    reply1, history = run_agent("My name is Devansh.")
    print(reply1)

    reply2, history = run_agent("What is my name?", history)
    print(reply2)
