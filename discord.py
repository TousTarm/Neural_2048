import os
import requests
from dotenv import load_dotenv
from datetime import datetime  # <-- Nový import pro čas

# Načte proměnné z .env souboru
load_dotenv()

WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

def send_new_record_notification(tile, episode, epsilon, score):
    """
    Odešle kompaktní notifikaci o novém rekordu na Discord s časem.
    """
    if not WEBHOOK_URL:
        print("Discord Webhook URL nenalezeno v .env! Notifikace nebude odeslána.")
        return

    # Získáme aktuální čas v evropském formátu
    # %d = den, %m = měsíc, %Y = rok, %H = hodina (24h), %M = minuta, %S = sekunda
    current_time = datetime.now().strftime("%d.%m.%Y %H:%M:%S")

    message = {
        "content": "**NEW HIGH SCORE!**",
        "embeds": [
            {
                "title": f"Tile: {int(tile)}",
                "color": 65535,  # Azurová (Cyan)
                "fields": [
                    {"name": "Score", "value": str(int(score)), "inline": True},
                    {"name": "Episode", "value": str(episode), "inline": True},
                    {"name": "Epsilon", "value": f"{epsilon:.4f}", "inline": True},
                    # Přidáno pole pro čas
                    {"name": "TIMESTAMP", "value": current_time, "inline": True}
                ],
                "footer": {"text": "Neural 2048 Training Bot"}
            }
        ]
    }

    try:
        response = requests.post(WEBHOOK_URL, json=message, timeout=2)
        if response.status_code == 204:
            pass
        else:
            print(f"Chyba Discordu: {response.status_code}")
    except Exception as e:
        print(f"Chyba při odesílání webhooku: {e}")