import argparse
import re
import os
import sys
import openai
import time
import random
from tqdm import tqdm
import numpy as np
from dotenv import load_dotenv

from data_utils import ANLI, ECQA, Aqua, DateUnderstanding, GSM8k, StrategyQA


load_dotenv()
openai.api_type = "azure"
openai.api_base = os.environ['OPEN_AI_API_BASE']
openai.api_version = os.environ['OPEN_AI_API_VERSION']
openai.api_key = os.environ['OPEN_AI_API_KEY']

def construct_message(agents, question, idx):
    if len(agents) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct. Put your final answer in the form (...) at the end of your response. For example: (yes), (123.45), (A)"}

    prefix_string = "These are the solutions to the problem from other agents: "

    for agent in agents:
        agent_response = agent[idx]["content"]
        response = "\n\n One agent solution: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + """\n\n Using the reasoning from other agents as additional advice, can you give an updated answer? Examine your solution and that other agents step by step. Put your answer in the form (...) at the end of your response.  For example: (yes), (123.45), (A)""".format(question)
    return {"role": "user", "content": prefix_string}


def construct_assistant_message(completion):
    content = completion["choices"][0]["message"]["content"]
    return {"role": "assistant", "content": content}


def generate_answer(answer_context):
    try:
        completion = openai.ChatCompletion.create(
              # engine="gpt-35-turbo",
              engine="gpt-4",
              messages=answer_context)
    
    except Exception as e:
        print(e)
        print("retrying due to an error......")
        time.sleep(20)
        return generate_answer(answer_context)

    return completion


def most_frequent(List):
    counter = 0
    num = List[0]

    for i in List:
        current_frequency = List.count(i)
        if current_frequency > counter:
            counter = current_frequency
            num = i

    return num


def parse_answers(dataset, text_answers):
    invalid_answer = 0
    parsed_answers = []

    if dataset == "SQA":
        for ans in text_answers:
            if re.search(r'\b(no|No|NO)\.?\b', ans):
                parsed_answers.append("no")
            elif re.search(r'\b(yes|Yes|YES)\.?\b', ans):
                parsed_answers.append("yes")
        
        if not parsed_answers:
            invalid_answer += 1
    elif dataset == "ECQA":
        for ans in text_answers:
            match = re.search(r'\(([A-E])\)', ans)

            if match:
                letter = match.group(1)
                parsed_answers.append(letter)
        
        if not parsed_answers:
            invalid_answer += 1
    elif dataset == "Aqua":
        for ans in text_answers:
            match = re.search(r'\(([A-E])\)', ans)

            if match:
                letter = match.group(1)
                parsed_answers.append(letter)
        
        if not parsed_answers:
            invalid_answer += 1
    elif dataset == "ANLI":
        for ans in text_answers:
            match = re.search(r'\((e|c|n|contradiction|neutral|entailment)\)', ans, re.IGNORECASE)

            if match:
                letter = match.group(1)
                parsed_answers.append(letter)
        if not parsed_answers:
            invalid_answer += 1
    elif dataset == "DateUnderstanding":
        for ans in text_answers:
            match = re.search(r'\(([A-E])\)', ans)

            if match:
                letter = match.group(1)
                parsed_answers.append(letter)
        if not parsed_answers:
            invalid_answer += 1

    return parsed_answers, invalid_answer


def main(args, dataset, test_samples):
    agents = 3
    rounds = 2

    random.seed(0)

    num_error = 0
    invalid_answer = 0

    accuracies = []
    for _ in tqdm(range(args.rounds)):
        num_correct = 0

        for test_sample in tqdm(test_samples):
            try:
                question = test_sample["question"]
                answer = test_sample["answer"]

                agent_contexts = [[{"role": "user", "content": question}] for _ in range(agents)]

                for round in range(rounds):
                    for i, agent_context in enumerate(agent_contexts):

                        if round != 0:
                            agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                            message = construct_message(agent_contexts_other, question, 2 * round - 1)
                            agent_context.append(message)

                        completion = generate_answer(agent_context)

                        assistant_message = construct_assistant_message(completion)
                        agent_context.append(assistant_message)
                        print(completion)

                text_answers = []

                for agent_context in agent_contexts:
                    text_answer =  agent_context[-1]['content']

                    if text_answer is None:
                        continue

                    text_answers.append(text_answer)


                try:
                    print(text_answers)
                    parsed_text_answers, invalid_answer_cnt = parse_answers(dataset, text_answers)

                    invalid_answer += invalid_answer_cnt

                    text_answer = most_frequent(parsed_text_answers)
                    
                    if text_answer == answer:
                        num_correct += 1
                except:
                    num_error += 1
                    continue
            except Exception as e:
                print(f"Exception during simulation: {e}.", file=sys.stderr)
                num_error += 1

            # TODO nekde si asi logovat ty vysledky, ta je to pak dohledatelny

        accuracy = num_correct / len(test_samples)
        print(f"Accuracy: {accuracy}")

        accuracies.append(accuracy)

    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)

    # TODO ukladat accuracies a mean accuracy s std_accuracy

    print(f"Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Num error: {num_error}")
    print(f"Num invalid answer: {invalid_answer}")


if __name__ == '__main__':
    # Path from root folder (backend/)
    dataset_dir = "./dataset/ReConcile/"
    parser = argparse.ArgumentParser(
        description="Simulation of Human Interaction using AI Evaluation Tool for Group Debate"
    )

    parser.add_argument('--dataset', default='SQA', type=str)
    parser.add_argument('--num_samples', default=0, type=int)
    parser.add_argument('--rounds', default=3, type=int)
    # parser.add_argument(
    #     "--clever_mode",
    #     default=False,
    #     type=bool,
    #     help="The clever mode to let the model add reasoning step and provide the best possible answer without knowledge levels.",
    # )

    # TODO
    # results_dir

    args = parser.parse_args()
    # os.environ["FINAL_ANSWER_TEMPLATE"] = "generate_final_answer_base.j2"

    if args.dataset == "SQA":
        data = StrategyQA(data_dir=f'{dataset_dir}{args.dataset}')
    elif args.dataset == "ECQA":
        data = ECQA(data_dir=f'{dataset_dir}{args.dataset}')
    elif args.dataset == "Aqua":
        data = Aqua(data_dir=f'{dataset_dir}{args.dataset}')
    elif args.dataset == "ANLI":
        data = ANLI(data_dir=f'{dataset_dir}{args.dataset}')
    elif args.dataset == "DateUnderstanding":
        data = DateUnderstanding(data_dir=f'{dataset_dir}{args.dataset}')
    else:
        print(f"Invalid dataset provided.", file=sys.stderr)
        sys.exit(1)

    # TODO random vyber samplu dle poctu samples
    # If num_samples == 0, take all samples available.
    test_samples = (
        data.get_test_samples()[: args.num_samples]
        if args.num_samples != 0
        else data.get_test_samples()
    )

    # TODO pokracovani
    main(args, args.dataset, test_samples)
