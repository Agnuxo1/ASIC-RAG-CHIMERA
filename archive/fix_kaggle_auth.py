import os
import json

kaggle_dir = os.path.expanduser('~/.kaggle')
if not os.path.exists(kaggle_dir):
    os.makedirs(kaggle_dir)

config = {
    "username": "franciscoangulo",
    "key": "efab210d9fb6ffb3e21979ecd38e103c"
}

with open(os.path.join(kaggle_dir, 'kaggle.json'), 'w') as f:
    json.dump(config, f)

print("Kaggle config written successfully")
