import os
from dotenv import load_dotenv, find_dotenv
import openai

env_path = find_dotenv()
load_dotenv(env_path)

openai.api_key = os.getenv('OPENAI_KEY')

#caption only hardcoded until we have output from models
caption = "man reading book on the dock near post"

def gpt3(prompt=f"write a love poem about {caption}:", engine='text-davinci-002',
         temperature=0.7,top_p=1, max_tokens=256,
        frequency_penalty=0, presence_penalty=0):
    response = openai.Completion.create(engine=engine,
                                        prompt=prompt,
                                        temperature=temperature,
                                        max_tokens=max_tokens,
                                        top_p=top_p,
                                        frequency_penalty=frequency_penalty,
                                        presence_penalty=presence_penalty)

    return response









# the below is test code for the API.

# response = openai.Completion.create(
#   engine="text-davinci-002",
#   prompt=prompt_text,
#   temperature=0.7,
#   top_p=1,
#   max_tokens=256,
#   frequency_penalty=0.5,
#   presence_penalty=0
# )

# print(response)
