import tkinter as tk
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import tkinter.messagebox 
import os
from createCharacterFunctions import gather_data, train_model
from tkinter.scrolledtext import ScrolledText
import threading
import sys
import logging
import time

class TextRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        self.text_widget.insert(tk.END, message)
        self.text_widget.update()

    def flush(self):
        pass

def fine_tune_model(contact_name, data_amount):
    gather_data(contact_name, data_amount)
    train_model(contact_name)
    global model
    global tokenizer
    global device
    model = GPT2LMHeadModel.from_pretrained('./model' + contact_name)
    tokenizer = GPT2Tokenizer.from_pretrained('./model' + contact_name)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    
def loading_screen(log_widget, thread):
    contact_name = contact_input.get().strip()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    stream_handler = logging.StreamHandler(TextRedirector(log_widget))
    logger.addHandler(stream_handler)

    elapsed = 0
    while thread.is_alive():
        chat_window.config(state=tk.NORMAL)
        chat_window.delete(1.0, tk.END) 

        time_str = time.strftime("%M:%S", time.gmtime(elapsed))

        chat_window.insert(tk.END, f"loading... {time_str}", "center")
        chat_window.config(state=tk.DISABLED)

        logger.info("")
        elapsed+=1
        time.sleep(1)
    
    chat_window.config(state=tk.NORMAL)
    chat_window.delete(1.0, tk.END) 
    chat_window.insert(tk.END, f"Chat with {contact_name}\n\n", "center")
    chat_window.config(state=tk.DISABLED)
    contact_input.delete(0, tk.END)
    
def set_contact():
    contact_name = contact_input.get().strip()
    if contact_name:
        if os.path.isdir("model" + contact_name):
            global model
            global tokenizer
            global device
            model = GPT2LMHeadModel.from_pretrained('./model' + contact_name)
            tokenizer = GPT2Tokenizer.from_pretrained('./model' + contact_name)
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            model.to(device)
        else:
            open_custom_message_box()
    
        chat_window.config(state=tk.NORMAL)
        chat_window.delete(1.0, tk.END) 
        chat_window.insert(tk.END, f"Chat with {contact_name}\n\n", "center")
        chat_window.config(state=tk.DISABLED)
        contact_input.delete(0, tk.END)

def handle_result(result):
    contact_name = contact_input.get().strip()
    
    data_amount = 3000
    if result < 0:
        data_amount = 3000
    elif result > 100:
        data_amount = 20000
    else:
        data_amount = int(result * 170) + 3000
    print(data_amount)
    thread = threading.Thread(target=fine_tune_model, args=(contact_name, data_amount), daemon=True)
    thread.start()
    loading_screen(log_widget, thread)


def generate_response(user_input):
    try:
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
        return response
    except NameError as e:
        return "Model not initialized; please enter the phone number above, and/or click regenerate."

def send_message(event=None):  
    user_message = user_input.get()
    if user_message.strip():
        chat_window.config(state=tk.NORMAL)
        chat_window.insert(tk.END, f"You: {user_message}\n", "right")
        user_input.delete(0, tk.END)

        response = generate_response(user_message)

        chat_window.insert(tk.END, f"Computer: {response}\n", "left")
        chat_window.config(state=tk.DISABLED)
        chat_window.see(tk.END)


def open_custom_message_box():
    def on_submit():
        value = slider.get()
        custom_box.destroy()
        handle_result(value)
    
    def on_cancel():
        custom_box.destroy()

    custom_box = tk.Toplevel(root)
    custom_box.title("Confirmation")
    custom_box.geometry("600x200")
    custom_box.grab_set()  # Make it modal

    label = tk.Label(custom_box, text="Select a value:")
    label.pack(pady=10)

    slider_frame = tk.Frame(custom_box)
    slider_frame.pack(pady=10, padx=20, fill="x")

    min_label = tk.Label(slider_frame, text="text soon\n(~3 minutes)\npoorer accuracy")
    min_label.pack(side="left", padx=5)

    slider = tk.Scale(
        slider_frame,
        from_=0, to=100,
        orient="horizontal",
        length=100,          # Slider length
        sliderlength=20,     # Size of the slider knob
        width=20,            # Thickness of the scale line
        highlightthickness=0, # Removes border highlight
        bg="#f0f0f0",        # Background color
        troughcolor="#cccccc", # Color of the trough
        fg="#333333"         # Text color
    )
    slider.pack(side="left", expand=True, fill="x")

    max_label = tk.Label(slider_frame, text="text later\n(a few hours)\nbest accuracy")
    max_label.pack(side="left", padx=5)


    buttons_frame = tk.Frame(custom_box)
    buttons_frame.pack(pady=10, padx=20, fill="x")

    cancel_button = tk.Button(buttons_frame, text="Cancel", command=on_cancel)
    cancel_button.pack(side="left", padx=5)

    submit_button = tk.Button(buttons_frame, text="Submit", command=on_submit)
    submit_button.pack(side="right", padx=5)

    custom_box.mainloop()

root = tk.Tk()
root.title("Texting Interface")
root.geometry("800x600")  

contact_frame = tk.Frame(root)
contact_frame.pack(padx=10, pady=5, fill=tk.X)

contact_input = tk.Entry(contact_frame, width=40)
contact_input.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

contact_button = tk.Button(contact_frame, text="Enter", command=set_contact)
contact_button.pack(side=tk.RIGHT, padx=5)

log_widget = ScrolledText(root, wrap=tk.WORD, height=10, width=80)
log_widget.pack(padx=10, pady=10)
log_widget.pack_forget()

chat_window = tk.Text(root, state=tk.DISABLED, wrap=tk.WORD)
chat_window.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

chat_window.tag_configure("left", justify="left")
chat_window.tag_configure("right", justify="right")
chat_window.tag_configure("center", justify="center")

input_frame = tk.Frame(root)
input_frame.pack(padx=10, pady=5, fill=tk.X)

user_input = tk.Entry(input_frame)
user_input.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
user_input.bind("<Return>", send_message)

send_button = tk.Button(input_frame, text="Send", command=send_message)
send_button.pack(side=tk.RIGHT, padx=5)

root.mainloop()