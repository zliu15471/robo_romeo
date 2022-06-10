import os
from dotenv import load_dotenv, find_dotenv
import openai

env_path = find_dotenv()
load_dotenv(env_path)

openai.api_key = os.getenv('OPENAI_KEY')

#caption only hardcoded until we have output from models
caption = "a little girl in a red shirt and a skirt shirt is sitting on a park"

# def gpt3(prompt=f"write a love poem about {caption}:", engine='text-davinci-002',
#          temperature=0.7,top_p=1, max_tokens=256,
#         frequency_penalty=0, presence_penalty=0):
#     response = openai.Completion.create(engine=engine,
#                                         prompt=prompt,
#                                         temperature=temperature,
#                                         max_tokens=max_tokens,
#                                         top_p=top_p,
#                                         frequency_penalty=frequency_penalty,
#                                         presence_penalty=presence_penalty)
#     return response


# tests
response = openai.Completion.create(
  engine="text-davinci-002",
  prompt=f"write a shakesperian love poem about {caption}:",
  temperature=0.7,
  top_p=1,
  max_tokens=256,
  frequency_penalty=0.5,
  presence_penalty=0)


print(response)
