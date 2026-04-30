import 'dotenv/config';
import OpenAI from 'openai';

const client = new OpenAI({
  baseURL: 'https://openrouter.ai/api/v1',
  apiKey: process.env.OPENROUTER_API_KEY,
});

async function chat(prompt, model = 'openai/gpt-4o') {
  const response = await client.chat.completions.create({
    model,
    messages: [{ role: 'user', content: prompt }],
  });
  return response.choices[0].message.content;
}

// Quick test
const result = await chat('Say hello in one sentence.');
console.log(result);
