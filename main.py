import os
from dotenv import load_dotenv

# Carga el archivo .env automáticamente
load_dotenv()

# Lee la variable del entorno
token = os.environ.get("SUPER_TOKEN_SECRETO")

print(f"La variable secreta dice: {token}")
