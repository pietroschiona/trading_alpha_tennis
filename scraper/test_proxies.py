import urllib3
import requests
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional
import time
from pathlib import Path
import re
import threading
import sys
try:
    import socks
except ImportError:
    socks = None


# ============================================================================
# CONFIGURAZIONE - MODIFICA QUESTI PARAMETRI
# ============================================================================

# File di input/output
DEFAULT_PROXY_FILE = "all_proxies.txt"  # File txt con le proxy (uno per riga)
# File dove salvare le proxy verificate
DEFAULT_OUTPUT_FILE = "verified_proxies.txt"

# Impostazioni velocità e performance
# Timeout in secondi per ogni richiesta (più basso = più veloce ma meno affidabile)
DEFAULT_TIMEOUT = 5
# Numero di thread paralleli (50 è sicuro per 16GB RAM)
DEFAULT_MAX_WORKERS = 50
PROGRESS_UPDATE_INTERVAL = 50  # Ogni quante proxy mostrare aggiornamento progresso

# Servizi per verificare l'IP reale (solo 2 per velocità)
IP_CHECK_SERVICES = [
    ('https://api.ipify.org?format=json', 'ip'),
    ('http://ip-api.com/json/', 'query'),
]

# Livello di logging (DEBUG per dettagli completi, INFO per normale, WARNING per solo errori)
LOG_LEVEL = logging.INFO

# ============================================================================
# FINE CONFIGURAZIONE
# ============================================================================

# Configurazione logging
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Disabilita warning SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class ProxyTester:
    """Testa proxy da un file txt e salva quelli verificati in tempo reale."""

    def __init__(self, input_file: str, output_file: str = DEFAULT_OUTPUT_FILE,
                 timeout: int = DEFAULT_TIMEOUT, max_workers: int = DEFAULT_MAX_WORKERS):
        """
        Inizializza il tester.

        Args:
            input_file (str): File di input con i proxy (uno per riga)
            output_file (str): File dove salvare i proxy verificati
            timeout (int): Timeout in secondi per ogni richiesta
            max_workers (int): Numero massimo di thread paralleli
        """
        if socks is None:
            logger.warning(
                "⚠️  ATTENZIONE: Libreria 'pysocks' non trovata! I proxy SOCKS (socks5://...) falliranno tutti.")
            logger.warning(
                "   Installa con: pip install pysocks requests[socks]")

        self.input_file = Path(input_file)
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.timeout = timeout
        self.max_workers = max_workers
        self.file_lock = threading.Lock()  # Lock per scrittura file thread-safe
        self.verified_count = 0
        self.total_tested = 0

        # Inizializza il file di output
        self._init_output_file()

    def _init_output_file(self):
        """Inizializza il file di output con header."""
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(f"# Proxy verificati\n")
            f.write(f"# Generato il: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Formato: proxy | IP verificato | Tempo risposta\n\n")

        logger.info(
            f"📝 File di output inizializzato: {self.output_file.absolute()}")

    def extract_ip_from_proxy(self, proxy: str) -> Optional[str]:
        """
        Estrae l'IP dalla stringa del proxy.

        Args:
            proxy (str): Stringa proxy (es: http://1.2.3.4:8080 o socks5://user:pass@5.6.7.8:1080)

        Returns:
            Optional[str]: IP estratto o None
        """
        # Pattern per estrarre IP (supporta anche autenticazione)
        # Esempi: http://1.2.3.4:80, socks5://user:pass@5.6.7.8:1080
        patterns = [
            # Con o senza auth
            r'://(?:[^:@]+:[^@]+@)?(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})',
            r'^(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # Solo IP
        ]

        for pattern in patterns:
            match = re.search(pattern, proxy)
            if match:
                return match.group(1)

        return None

    def get_real_ip(self, proxy_dict: dict) -> Optional[str]:
        """
        Ottiene l'IP reale usando la proxy.

        Args:
            proxy_dict (dict): Dizionario proxy per requests

        Returns:
            Optional[str]: IP rilevato o None
        """
        for url, key in IP_CHECK_SERVICES:
            try:
                response = requests.get(
                    url,
                    proxies=proxy_dict,
                    timeout=self.timeout,
                    verify=False
                )

                if response.status_code == 200:
                    data = response.json()
                    if key in data:
                        ip = data[key]
                        # Pulisci l'IP (a volte httpbin restituisce "ip, ip")
                        if ',' in ip:
                            ip = ip.split(',')[0].strip()
                        return ip

            except Exception as e:
                logger.debug(f"Errore get_real_ip su {url}: {str(e)[:50]}")
                continue

        return None

    def load_proxies(self) -> List[str]:
        """Carica i proxy dal file txt."""
        if not self.input_file.exists():
            logger.error(f"File {self.input_file} non trovato!")
            return []

        proxies = []
        with open(self.input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Salta righe vuote e commenti
                if line and not line.startswith('#'):
                    # Rimuovi eventuali commenti inline
                    if '#' in line:
                        line = line.split('#')[0].strip()
                    if line:
                        proxies.append(line)

        logger.info(f"Caricati {len(proxies)} proxy da {self.input_file}")
        return proxies

    def test_proxy(self, proxy: str) -> Tuple[str, bool, float, Optional[str]]:
        """
        Testa un singolo proxy verificando anche l'IP reale.

        Args:
            proxy (str): Stringa del proxy (es: http://ip:port o socks5://ip:port)

        Returns:
            Tuple[str, bool, float, Optional[str]]: (proxy, success, response_time, real_ip)
        """
        start_time = time.time()

        # Estrai IP dichiarato dal proxy
        declared_ip = self.extract_ip_from_proxy(proxy)

        # Prepara il dizionario proxy per requests
        if proxy.startswith('socks'):
            proxy_dict = {
                'http': proxy,
                'https': proxy
            }
        else:
            proxy_dict = {
                'http': proxy,
                'https': proxy
            }

        try:
            # Ottieni l'IP reale tramite la proxy
            real_ip = self.get_real_ip(proxy_dict)

            if not real_ip:
                response_time = time.time() - start_time
                logger.warning(
                    f"❌ PROXY FALLITO: {proxy} - Nessun IP rilevato")
                return (proxy, False, response_time, None)

            # Verifica che l'IP corrisponda
            if declared_ip and real_ip != declared_ip:
                response_time = time.time() - start_time
                logger.warning(
                    f"⚠️  PROXY SCARTATO: {proxy} - IP non corrisponde! Dichiarato: {declared_ip}, Reale: {real_ip}")
                return (proxy, False, response_time, real_ip)

            response_time = time.time() - start_time
            logger.info(
                f"✅ PROXY VERIFICATO: {proxy} - IP: {real_ip} - Tempo: {response_time:.2f}s")
            return (proxy, True, response_time, real_ip)

        except Exception as e:
            response_time = time.time() - start_time
            # Mostra l'errore completo per debugging
            error_msg = str(e)
            if "SOCKSHTTPSConnectionPool" in error_msg and socks is None:
                error_msg += " (Probabilmente manca 'pysocks')"

            logger.warning(f"❌ PROXY FALLITO: {proxy} - Errore: {error_msg}")
            return (proxy, False, response_time, None)

    def save_verified_proxy(self, proxy: str, real_ip: str, response_time: float):
        """
        Salva immediatamente una proxy verificata nel file.

        Args:
            proxy (str): Stringa proxy
            real_ip (str): IP reale verificato
            response_time (float): Tempo di risposta
        """
        with self.file_lock:
            with open(self.output_file, 'a', encoding='utf-8') as f:
                f.write(f"{proxy} | {real_ip} | {response_time:.2f}s\n")

            self.verified_count += 1
            logger.info(
                f"💾 Salvato in file: {proxy} ({self.verified_count} verificati)")

    def test_all_proxies(self):
        """Testa tutti i proxy in parallelo."""
        proxies = self.load_proxies()
        if not proxies:
            return

        total = len(proxies)
        self.total_tested = 0

        logger.info(f"\n{'='*60}")
        logger.info(
            f"🚀 Inizio test di {total} proxy con {self.max_workers} thread...")
        logger.info(f"{'='*60}\n")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Avvia tutti i test
            future_to_proxy = {
                executor.submit(self.test_proxy, proxy): proxy
                for proxy in proxies
            }

            # Raccogli i risultati man mano che arrivano
            for future in as_completed(future_to_proxy):
                proxy, success, response_time, real_ip = future.result()
                self.total_tested += 1

                if success and real_ip:
                    # Salva immediatamente la proxy verificata
                    self.save_verified_proxy(proxy, real_ip, response_time)

                # Progress update
                if self.total_tested % PROGRESS_UPDATE_INTERVAL == 0:
                    logger.info(
                        f"\n📊 Progresso: {self.total_tested}/{total} ({self.total_tested/total*100:.1f}%) - Verificati: {self.verified_count}\n")

        logger.info(f"\n{'='*60}")
        logger.info(f"✅ Test completato!")
        logger.info(f"Proxy testati: {total}")
        logger.info(f"Proxy verificati: {self.verified_count}")
        logger.info(f"Tasso di successo: {self.verified_count/total*100:.1f}%")
        logger.info(f"File risultati: {self.output_file.absolute()}")
        logger.info(f"{'='*60}\n")

    def run(self):
        """Esegue il test completo."""
        self.test_all_proxies()


def main():
    """Funzione principale."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Testa proxy da un file txt con verifica IP')
    parser.add_argument(
        'input_file',
        type=str,
        nargs='?',
        default=DEFAULT_PROXY_FILE,
        help=f'File txt contenente i proxy (default: {DEFAULT_PROXY_FILE})'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=DEFAULT_OUTPUT_FILE,
        help=f'File di output per i proxy verificati (default: {DEFAULT_OUTPUT_FILE})'
    )
    parser.add_argument(
        '-t', '--timeout',
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f'Timeout in secondi per ogni proxy (default: {DEFAULT_TIMEOUT})'
    )
    parser.add_argument(
        '-w', '--workers',
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f'Numero di thread paralleli (default: {DEFAULT_MAX_WORKERS})'
    )

    args = parser.parse_args()

    # Crea il tester ed esegui
    tester = ProxyTester(
        input_file=args.input_file,
        output_file=args.output,
        timeout=args.timeout,
        max_workers=args.workers
    )

    tester.run()


if __name__ == '__main__':
    main()
