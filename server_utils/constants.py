CONTROLLER_HEART_BEAT_EXPIRATION = 90
WORKER_HEART_BEAT_INTERVAL = 30
WORKER_API_TIMEOUT = 20

import os
user = os.environ["USER"]
print(user)
LOGDIR = f"/home/{user}/VLP_web/server_utils/runtime_log"
CONVERSATION_SAVE_DIR = f'/home/{user}/data1/VLP_web_data/conversation_data'


rules_markdown = """ ### Rules
- Vote for VLM on visual question answering.
- Load an image and ask a question. Only one question is supported per round.
- Two models are anonymous before your vote.
"""


notice_markdown = """ ### Terms of use
Placeholder for terms of use
"""


license_markdown = """ ### License
Placeholder for license
"""
