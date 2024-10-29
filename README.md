# text-message-mimic

Creates a chatbot to mimic the way someone has texted you through imessages.

More accurate with longer text history.


Usage:
```bash
$ python createCharacter.py xxxyyyzzzz (a phone number)
$ python chat.py
```

Note
- only functional on M! or M2 architecture macs
- running environment needs access to chat.db file on mac
  - system settings -> privacy & security -> full disk access -> select your running environment
 
Specifics
- fine tunes gpt2
