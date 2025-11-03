# PlanMe.py — fixed Phi-2 output to only show final schedule
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer
import torch

def load_models():
    print("Loading google/flan-t5-base (scheduler)...")
    scheduler_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    scheduler_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base", dtype=torch.float32)
    print("Flan-T5 loaded.\n")

    print("Loading microsoft/phi-2 (parser)...")
    parser_tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    parser_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", dtype=torch.float32)
    print("Phi-2 loaded.\n")

    return scheduler_model, scheduler_tokenizer, parser_model, parser_tokenizer


def generate_schedule(prompt, model, tokenizer):
    input_text = f"Create a clean daily schedule from this description: {prompt}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=150)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def parse_schedule(raw_text, model, tokenizer):
    instruction = (
        "Reformat this text into a numbered daily schedule.\n"
        "Follow this style of formatting (but do NOT copy the times or tasks):\n"
        "1. [Activity] from [start time]–[end time]\n"
        "2. [Activity] from [start time]–[end time]\n"
        "If a duration is given (like 'sleep 7 hours from 2100'), convert it to an end time.\n"
        "Use clean, human-readable time ranges.\n\n"
        f"Text to clean and format:\n{raw_text}\n\n"
        "Return only the final numbered schedule — no examples, no explanations."
    )

    inputs = tokenizer(instruction, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=200)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only numbered lines
    lines = [l.strip() for l in text.splitlines() if l.strip().startswith(tuple(str(i) + "." for i in range(1, 10)))]

    # Remove duplicates
    seen = set()
    unique_lines = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            unique_lines.append(line)

    return "\n".join(unique_lines) if unique_lines else text.strip()


    inputs = tokenizer(instruction, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=200)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only numbered lines
    lines = [l.strip() for l in text.splitlines() if l.strip().startswith(tuple(str(i) + "." for i in range(1, 10)))]

    # Remove duplicates (e.g., repeated schedules)
    seen = set()
    unique_lines = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            unique_lines.append(line)

    return "\n".join(unique_lines) if unique_lines else text.strip()


    inputs = tokenizer(instruction, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=200)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Post-cleaning: keep only lines starting with a number and dot
    cleaned_lines = []
    for line in text.splitlines():
        if line.strip().startswith(tuple(str(i) + "." for i in range(1, 10))):
            cleaned_lines.append(line.strip())

    if cleaned_lines:
        return "\n".join(cleaned_lines)
    else:
        return text.strip()


def main():
    scheduler_model, scheduler_tokenizer, parser_model, parser_tokenizer = load_models()

    print("\nWelcome to Cadet Scheduler (Plan Me)")
    print("Enter schedule in this format:")
    print("Day 1 schedule: class 0800-1300, gym 1400-1530, homework 1800-2100, sleep 7 hours")
    print("Type 'exit' to quit.\n")

    while True:
        user_prompt = input("Enter prompt: ")
        if user_prompt.lower() == "exit":
            break

        print("\nGenerating schedule with Flan-T5...")
        raw_schedule = generate_schedule(user_prompt, scheduler_model, scheduler_tokenizer)

        print("\nRefining format with Phi-2...")
        cleaned_schedule = parse_schedule(raw_schedule, parser_model, parser_tokenizer)

        print("\nFinal Output:\n")
        print(cleaned_schedule)
        print("\n--------------------------------------\n")


if __name__ == "__main__":
    main()
