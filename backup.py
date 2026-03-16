import shutil
from datetime import datetime

backup_name = f"backups/plantguard_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.db"

import os
os.makedirs("backups", exist_ok=True)

shutil.copy2("plantguard.db", backup_name)
print(f"Backed up to: {backup_name}")