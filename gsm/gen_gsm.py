import openai
import json
import numpy as np
import random
import os
from dotenv import load_dotenv

# random.seed(1234)
# random.seed(0)
random.seed(9999)

load_dotenv()
openai.api_type = "azure"
openai.api_base = os.environ['OPEN_AI_API_BASE']
openai.api_version = os.environ['OPEN_AI_API_VERSION']
openai.api_key = os.environ['OPEN_AI_API_KEY']


def construct_message(agents, question, idx):
    if len(agents) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct. Please reiterate your answer, with your final answer a single numerical number, in the form \\boxed{{answer}}."}

    prefix_string = "These are the solutions to the problem from other agents: "

    for agent in agents:
        agent_response = agent[idx]["content"]
        response = "\n\n One agent solution: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + """\n\n Using the solutions from other agents as additional information, can you provide your answer to the math problem? \n The original math problem is {}. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response.""".format(question)
    return {"role": "user", "content": prefix_string}


def construct_assistant_message(completion):
    content = completion["choices"][0]["message"]["content"]
    return {"role": "assistant", "content": content}


def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]

if __name__ == "__main__":
    agents = 3
    rounds = 2

    generated_description = {}

    questions = read_jsonl("test.jsonl")
    random.shuffle(questions)

    idx = 0
    for data in questions[:30]:
        idx += 1
        print(f"QUESTION: {idx}")
        
        question = data['question']
        answer = data['answer']

        agent_contexts = [[{"role": "user", "content": """Can you solve the following math problem? {} Explain your reasoning. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response. """.format(question)}] for agent in range(agents)]

        for round in range(rounds):
            print(f"ROUND: {round}")
            for i, agent_context in enumerate(agent_contexts):

                if round != 0:
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                    message = construct_message(agent_contexts_other, question, 2*round - 1)
                    agent_context.append(message)

                completion = openai.ChatCompletion.create(
                    # engine="gpt-35-turbo",
                    # engine="gpt-4",
                    engine="gpt-4o",
                    messages=agent_context,
                    n=1
                )
                #completion = openai.ChatCompletion.create(
                #          model="gpt-3.5-turbo-0301",
                #          messages=agent_context,
                #          n=1)

                assistant_message = construct_assistant_message(completion)
                agent_context.append(assistant_message)

                print(assistant_message)
                print("--------------")
            

        generated_description[question] = (agent_contexts, answer)

    json.dump(generated_description, open("gsm_{}_{}_round_3.json".format(agents, rounds), "w"))

    import pdb
    pdb.set_trace()
    print(answer)
    print(agent_context)
