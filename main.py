from gaia_agent.api import fetch_questions, submit_answers, get_file
from gaia_agent.agent import GAIAAgent
from tqdm import tqdm
import yaml


def download_questions_and_files():
    questions = fetch_questions(timeout=10)

    files = dict()
    for question in questions:
        task_id = question["task_id"]
        file = get_file(task_id)
        files.update({task_id: file})

    file_names = {
        "cca530fc-4052-43b2-b130-b30968d8aa44": "chess_position.jpg",
        "99c9cc74-fdc8-46c6-8f8d-3ce2d3bfeea3": "Strawberry pie.mp3",
        "f918266a-b3e0-4914-865d-4faa564f1aef": "python_code.py",
        "1f975693-876d-457b-a649-393859e79bf3": "Homework.mp3",
        "7bd855d8-463d-4ed5-93ca-5fe35145f733": "menu_items.xlsx",
    }

    for task_id, file_name in file_names.items():
        # Get the bytes sequence for the task_id
        bytes_sequence = files[task_id]
        # Store the bytes sequence in a temporary file
        with open(f"files/{file_name}", "wb") as f:
            f.write(bytes_sequence)

    return questions


async def answer_questions(agent, questions):
    answers = list()
    for question_dict in tqdm(questions):
        query = question_dict["question"]
        response, reasoning_process = await agent.run(
            query, return_reasoning_process=True
        )
        task_id = question_dict["task_id"]
        answer = {
            "task_id": task_id,
            "submitted_answer": response,
            "reasoning_trace": reasoning_process,
        }
        answers.append(answer)
    return answers


if __name__ == "__main__":
    questions = download_questions_and_files()
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    agent = GAIAAgent(config)

    answers = answer_questions(agent, questions)

    status = submit_answers(
        username=config["username"],
        agent_code=config["agent_code"],
        answers=answers,
    )
    print(status)
    print("Done!")
