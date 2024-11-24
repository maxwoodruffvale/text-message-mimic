import tkinter as tk

def generate_response(user_input):
    return "This is a response."

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

def set_contact():
    contact_name = contact_input.get().strip()
    if contact_name:
        #check if that model exists in yo directory
        # prompt to generate a new one if not
        chat_window.config(state=tk.NORMAL)
        chat_window.delete(1.0, tk.END) 
        chat_window.insert(tk.END, f"Chat with {contact_name}\n\n", "center")
        chat_window.config(state=tk.DISABLED)
        contact_input.delete(0, tk.END)

root = tk.Tk()
root.title("Texting Interface")
root.geometry("800x600")  

contact_frame = tk.Frame(root)
contact_frame.pack(padx=10, pady=5, fill=tk.X)

contact_input = tk.Entry(contact_frame, width=40)
contact_input.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

contact_button = tk.Button(contact_frame, text="Enter", command=set_contact)
contact_button.pack(side=tk.RIGHT, padx=5)

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
