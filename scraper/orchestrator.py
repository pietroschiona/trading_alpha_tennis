"""
Orchestrator per gestire il ciclo di vita delle proxy e dello scraper.
Scarica proxy da GitHub, le testa, avvia lo scraper e ripete se necessario.
"""

import subprocess
import sys
import os
import time
import requests
import psutil
from datetime import datetime

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

# URLs delle proxy (raw GitHub)
PROXY_URLS = [
    ("https://raw.githubusercontent.com/proxifly/free-proxy-list/main/proxies/protocols/https/data.txt", "http"),
    ("https://raw.githubusercontent.com/proxifly/free-proxy-list/main/proxies/protocols/socks4/data.txt", "socks4"),
    ("https://raw.githubusercontent.com/proxifly/free-proxy-list/main/proxies/protocols/socks5/data.txt", "socks5"),
]

# File paths
ALL_PROXIES_FILE = "all_proxies.txt"
VERIFIED_PROXIES_FILE = "verified_proxies.txt"
SCRAPER_SCRIPT = "scraper.py"
TESTER_SCRIPT = "test_proxies.py"

# RAM Management (per 16GB)
MAX_RAM_PERCENT = 80  # Ferma se RAM supera 80%

# ============================================================================
# FUNZIONI
# ============================================================================


def log(message):
    """Logging con timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")


def check_ram():
    """Controlla uso RAM e avvisa se troppo alto."""
    ram = psutil.virtual_memory()
    percent = ram.percent
    if percent > MAX_RAM_PERCENT:
        log(f"⚠️  ATTENZIONE: RAM al {percent}% - Aspetto che scenda...")
        while psutil.virtual_memory().percent > MAX_RAM_PERCENT:
            time.sleep(5)
        log("✅ RAM tornata a livelli normali")
    return percent


def kill_all_chrome():
    """Termina tutti i processi Chrome/ChromeDriver zombie."""
    killed = 0
    for proc in psutil.process_iter(['name']):
        try:
            name = proc.info['name']
            if name and name.lower() in ['chrome.exe', 'chromedriver.exe']:
                proc.kill()
                killed += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    if killed > 0:
        log(f"🧹 Terminati {killed} processi Chrome zombie")
    return killed


def download_proxies():
    """Scarica proxy da tutti gli URL e le unisce in un file."""
    log("📥 Scaricando proxy da GitHub...")

    all_proxies = set()  # Usa set per evitare duplicati

    for url, protocol in PROXY_URLS:
        try:
            log(f"   → Scaricando {protocol}...")
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            lines = response.text.strip().split('\n')
            count = 0

            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Aggiungi protocollo se non presente
                    if not line.startswith(('http://', 'https://', 'socks4://', 'socks5://')):
                        # Il formato è ip:port, aggiungi protocollo
                        proxy = f"{protocol}://{line}"
                    else:
                        proxy = line
                    all_proxies.add(proxy)
                    count += 1

            log(f"   ✓ {protocol}: {count} proxy")

        except Exception as e:
            log(f"   ✗ Errore {protocol}: {e}")

    # Salva tutte le proxy
    with open(ALL_PROXIES_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(sorted(all_proxies)))

    log(f"✅ Totale: {len(all_proxies)} proxy scaricate → {ALL_PROXIES_FILE}")
    return len(all_proxies)


def run_proxy_tester():
    """Esegue test_proxies.py e aspetta che finisca."""
    log("🔍 Avviando test delle proxy...")
    check_ram()

    cmd = [sys.executable, TESTER_SCRIPT,
           ALL_PROXIES_FILE, "-o", VERIFIED_PROXIES_FILE]

    try:
        result = subprocess.run(cmd, check=True)

        # Conta proxy verificate
        if os.path.exists(VERIFIED_PROXIES_FILE):
            with open(VERIFIED_PROXIES_FILE, 'r') as f:
                verified = len([l for l in f if l.strip()
                               and not l.startswith('#')])
            log(f"✅ Test completato: {verified} proxy verificate")
            return verified
        return 0

    except subprocess.CalledProcessError as e:
        log(f"❌ Errore nel test proxy: {e}")
        return 0
    except KeyboardInterrupt:
        log("🛑 Test interrotto dall'utente")
        return 0


def run_scraper():
    """Esegue lo scraper e ritorna l'exit code."""
    log("🚀 Avviando scraper...")
    check_ram()

    cmd = [sys.executable, SCRAPER_SCRIPT]

    try:
        result = subprocess.run(cmd)
        return result.returncode

    except KeyboardInterrupt:
        log("🛑 Scraper interrotto dall'utente")
        return -1


def main():
    """Loop principale dell'orchestratore."""
    print("\n" + "="*60)
    print("🎯 ORCHESTRATOR - Gestione automatica proxy e scraper")
    print("   Exit codes: 0=completato, 2=proxy esaurite, 3=restart schedulato")
    print("="*60 + "\n")

    cycle = 0

    while True:
        cycle += 1
        log(f"\n{'='*40}")
        log(f"📍 CICLO #{cycle}")
        log(f"{'='*40}\n")

        # 0. Cleanup Chrome zombie processes
        kill_all_chrome()
        time.sleep(2)  # Wait for processes to terminate

        # 1. Controlla RAM
        ram_percent = check_ram()
        log(f"💾 RAM: {ram_percent}%")

        # 2. Scarica proxy
        proxy_count = download_proxies()
        if proxy_count == 0:
            log("❌ Nessuna proxy scaricata. Riprovo tra 60 secondi...")
            time.sleep(60)
            continue

        # 3. Testa proxy
        verified_count = run_proxy_tester()
        if verified_count == 0:
            log("❌ Nessuna proxy verificata. Riprovo tra 60 secondi...")
            time.sleep(60)
            continue

        # 4. Avvia scraper
        exit_code = run_scraper()

        # 5. Analizza risultato
        if exit_code == 0:
            log("\n" + "="*60)
            log("🎉 SCRAPER COMPLETATO CON SUCCESSO!")
            log("="*60 + "\n")
            break
        elif exit_code == -1:
            log("\n🛑 Interrotto dall'utente. Uscita.")
            break
        elif exit_code == 2:
            log(f"\n⚠️  Proxy esaurite (exit code 2)")
            log("🔄 Ricomincio il ciclo con nuove proxy...")
            time.sleep(3)
        elif exit_code == 3:
            log(
                f"\n🔄 Restart schedulato (exit code 3) - {MAX_LINKS_BEFORE_RESTART} links processati")
            log("🔄 Refresh proxy e riavvio scraper...")
            time.sleep(3)
        else:
            log(f"\n⚠️  Scraper terminato con codice sconosciuto {exit_code}")
            log("🔄 Ricomincio il ciclo...")
            time.sleep(5)


# Config for showing in logs
MAX_LINKS_BEFORE_RESTART = 600


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Orchestrator terminato dall'utente")
        kill_all_chrome()  # Cleanup on exit
        sys.exit(0)
