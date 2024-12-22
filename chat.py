import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained('./character')
tokenizer = GPT2Tokenizer.from_pretrained('./character')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def chat_with_bot():
    print("Start chatting with the bot (type 'exit' to stop)...")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        inputs = tokenizer(user_input + tokenizer.eos_token, return_tensors='pt', padding=True, truncation=True).to(device)
        
        response_ids = model.generate(
            inputs['input_ids'], 
            attention_mask=inputs['attention_mask'],
            max_length=50,
            do_sample=True,
            temperature=0.7,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )
        
        response = tokenizer.decode(response_ids[:, inputs['input_ids'].shape[-1]:][0], skip_special_tokens=True)
        print(f"Your contact: {response}")

chat_with_bot()
