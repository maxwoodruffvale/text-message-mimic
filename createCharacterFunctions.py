import sys
import sqlite3
import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, TrainerCallback
from datasets import Dataset


def gather_data(phone_number, limit=5000):
    db_path = os.path.expanduser('~/Library/Messages/chat.db')
    conn = sqlite3.connect(db_path)

    cursor=conn.cursor()

    query = f"""
    SELECT message.is_from_me, message.text, message.attributedBody
    FROM message
    LEFT JOIN handle ON message.handle_id = handle.ROWID
    WHERE handle.id = '+1{phone_number}'
    ORDER BY message.date ASC
    LIMIT {limit};
    """
    #remove limit for actual but i think this would cook my computer like crazy

    cursor.execute(query)
    results = cursor.fetchall()

    cursor.close()
    conn.close()

    updated_results = []
    
    for result in results:
        if(result[1] is not None): # the text is plain to see
            updated_results.append(result[0:1])
            continue

        if(result[2] is not None): # attribute body exists
            attributed_body = result[2].decode('utf-8', errors='replace')
            if "NSNumber" in str(attributed_body):
                attributed_body = str(attributed_body).split("NSNumber")[0]
                if "NSString" in attributed_body:
                    attributed_body = str(attributed_body).split("NSString")[1]
                    if "NSDictionary" in attributed_body:
                        attributed_body = str(attributed_body).split("NSDictionary")[0]
                        attributed_body = attributed_body[6:-12]
                        updated_results.append([result[0], attributed_body])

    with open('message_conversation.txt', 'w') as file:
        speaker=updated_results[0][0]
        prev_speaker=updated_results[0][0]
        message=str(updated_results[0][0]) + ": "
        for row in updated_results:
            if(len(row) < 2 or row[1] is None or row[1] == "￼" or row[1].startswith('Loved ')):
                continue
            speaker = row[0]
            if(speaker == prev_speaker):
                message += row[1] + ". "
            else:
                file.write(message + "\n")
                message = str(speaker) + ": " + row[1] + ". "
            prev_speaker = speaker
    print("Conversation Data gathered succesfully")


def train_model(phone_number):
    model_name = "distilgpt2"
    global tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("running on: " + str(device))
    model.to(device)

    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))

    dialogues = load_data('message_conversation.txt')
    print(dialogues[:5])  # Check the first few dialogues for correctness

    tokenized_data = tokenize_data(dialogues)
    dataset = Dataset.from_dict(tokenized_data)

    print("Data tokenized succesfuly")

    training_args = TrainingArguments(
        output_dir = './results',
        num_train_epochs = 3,
        per_device_train_batch_size = 2,
        logging_dir = './logs',
        logging_steps = 50,
        save_steps = 500,
        save_total_limit = 2,
        learning_rate=5e-5,
        evaluation_strategy = "no",
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    print("Began training loop(can take a little bit of time ~ 5-10 minutes)")
    trainer.train()

    print("Training loop complete")

    model.save_pretrained("./model" + str(phone_number))
    tokenizer.save_pretrained("./model" + str(phone_number))

#helpers
def load_data(file_path):
    dialogues = []
    with open(file_path, 'r') as f:
        dialogue = ""
        for line in f.readlines():
            if(': ' not in line):
                continue
            if(len(line.strip().split(': ')) < 2):
                continue
            if line.startswith("1:"):
                dialogue += line.strip().split(": ")[1] + " <|endoftext|> "  # Add end token after user input
            elif line.startswith("0:"):
                dialogue += line.strip().split(": ")[1] + " <|endoftext|> "
                dialogues.append(dialogue.strip())
                dialogue = ""
    return dialogues

def tokenize_data(dialogues):
    tokenized_dialogues = tokenizer(dialogues, return_tensors='pt', truncation=True, padding=True)
    input_ids = tokenized_dialogues['input_ids']
    labels = input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    return {'input_ids': input_ids, 'attention_mask': tokenized_dialogues['attention_mask'], 'labels': labels}