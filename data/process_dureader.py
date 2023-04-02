import json
import string


def read_data(file_path):
    with open(file_path, "r") as f:
        return f.readlines()


def replace_punc(stri):
    for i in string.punctuation:
        stri = stri.replace(i, '')
    return stri.strip()


def extract_questions_answers_fake_answers(data):
    questions, answers, fake_answers = [], [], []
    for i in data:
        o = json.loads(i)
        questions.append(o["question"])
        answers.append(o["answers"])
        fake_answers.append(o["fake_answers"])
    return questions, answers, fake_answers


def make_pairs(question, answers, fake_answers):
    if len(answers) == 0 or len(fake_answers) == 0:
        return False, None
    truth, fake, pairs = answers[0], fake_answers[0], []
    for a in answers:
        if len(a) > len(truth):
            truth = a
    for b in fake_answers:
        if 10 < len(b) < len(fake):
            fake = b
    for b in fake_answers:
        if replace_punc(b) == replace_punc(truth):
            continue
        pairs.append({"prompt": question, "chosen": truth, "rejected": b})
    for a in answers:
        if replace_punc(a) == replace_punc(truth):
            continue
        if replace_punc(a) == replace_punc(fake):
            continue
        pairs.append({"prompt": question, "chosen": a, "rejected": fake})
    return True, pairs


def process_data(questions, answers, fake_answers):
    only_prompt, pairs_data = [], []
    for q, a, f in zip(questions, answers, fake_answers):
        ok, pairs = make_pairs(q, a, f)
        if ok:
            pairs_data.extend(pairs)
        else:
            only_prompt.append({"instruction": q})
    return only_prompt, pairs_data


def save_jsonl(data, output_file):
    with open(output_file, 'w', encoding="utf-8") as jsonl_output:
        jsonl_output.write(json.dumps(data, ensure_ascii=False, indent=4))


def main():
    file_path = "/path-to-dureader/preprocessed/trainset/zhidao.train.json"
    data = read_data(file_path)
    questions, answers, fake_answers = extract_questions_answers_fake_answers(data)
    only_prompt, pairs_data = process_data(questions, answers, fake_answers)
    print(len(pairs_data))
    save_jsonl(pairs_data, "train_pairs.json")
    save_jsonl(only_prompt, "only_prompts.json")
