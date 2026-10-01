import time
import os
import csv
import random
import logging
import concurrent.futures
import threading
import signal
import sys
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

# --- IMMEDIATE SHUTDOWN FLAG ---
SHUTDOWN_FLAG = threading.Event()
CAPTCHA_DETECTED = threading.Event()  # Flag for CAPTCHA detection


def signal_handler(sig, frame):
    """Handle Ctrl+C immediately."""
    print("\n🛑 CTRL+C RECEIVED - SHUTTING DOWN IMMEDIATELY...")
    SHUTDOWN_FLAG.set()
    os._exit(0)  # Force immediate exit, no cleanup


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# --- CONFIGURATION (STRICT & PARANOID) ---
# INPUT_LINKS_FILE = "test_links.txt"
INPUT_LINKS_FILE = "test_links.txt"
INPUT_PROXIES_FILE = "verified_proxies.txt"
OUTPUT_FILE = "tennis_odds_data.csv"
SKIPPED_LINKS_FILE = "skipped_links.txt"  # Links visited but without odds
# Links with odds extracted successfully
PROCESSED_LINKS_FILE = "processed_links.txt"
FAILED_LINKS_FILE = "failed_links.txt"  # Links that failed after all retries

# Global Locks for file operations
PROXY_FILE_LOCK = threading.Lock()
SKIPPED_FILE_LOCK = threading.Lock()
PROCESSED_FILE_LOCK = threading.Lock()
FAILED_FILE_LOCK = threading.Lock()

# PERFORMANCE & CONCURRENCY
# Sweet spot: 15 workers allows reuse of 199 proxies every ~13 cycles, safe for IP health
# and manageable for CPU/RAM on a desktop.

# ((((58 chiamate l'ora per worker)*n° di workers)*numero di ore)*100)/ n° di visite giornaliere = percentuale di traffico giornaliero occupato)
# (58*21*24*100)/418703,672 = 6,98%

MAX_WORKERS = 51
MIN_PROXIES_THRESHOLD = 1  # Stop when only 1 proxy left
TIMEOUT_SEVEN_SECONDS = 17  # Max wait for page load

# PARANOID SECURITY SETTINGS
SAFETY_DELAY_MIN = 1.5
SAFETY_DELAY_MAX = 3.0
MAX_RETRIES = 3
VERIFY_IP_URL = "https://api.ipify.org?format=json"

# --- LOGGING CONFIG ---
# Set to True to save logs to file (creates scraper.log)
ENABLE_FILE_LOGGING = False
LOG_FILE = "scraper.log"


class ColorFormatter(logging.Formatter):
    """Formatter con colori per console Windows."""

    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'
    }

    ICONS = {
        'DEBUG': '🔍',
        'INFO': '✅',
        'WARNING': '⚠️',
        'ERROR': '❌',
        'CRITICAL': '🚨'
    }

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        icon = self.ICONS.get(record.levelname, '')
        reset = self.COLORS['RESET']

        # Formato: [HH:MM:SS] ICON MESSAGE
        record.msg = f"{color}[{datetime.now().strftime('%H:%M:%S')}] {icon} {record.msg}{reset}"
        return super().format(record)


class FileFormatter(logging.Formatter):
    """Formatter pulito per file di log."""

    def format(self, record):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return f"[{timestamp}] [{record.levelname:8}] {record.getMessage()}"


def setup_logging():
    """Configura logging elegante per console e file."""
    # Abilita colori ANSI su Windows
    os.system('')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Rimuovi handler esistenti
    logger.handlers = []

    # Console handler (colorato)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(ColorFormatter('%(message)s'))
    logger.addHandler(console)

    # File handler (dettagliato) - solo se abilitato
    if ENABLE_FILE_LOGGING:
        file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(FileFormatter())
        logger.addHandler(file_handler)

    return logger


# Inizializza logging
setup_logging()


def load_file_lines(filepath):
    """Safe read of lines from a file."""
    if not os.path.exists(filepath):
        logging.error(f"File not found: {filepath}")
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        # Parse proxy lines that look like: "socks5://ip:port | ip | time"
        lines = []
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            if "|" in line:
                lines.append(line.split("|")[0].strip())
            else:
                lines.append(line.strip())
        return lines


def remove_proxy_permanently(proxy_to_remove):
    """
    Removes a bad proxy from the text file in a thread-safe manner.
    """
    with PROXY_FILE_LOCK:
        try:
            # 1. Read all lines
            if not os.path.exists(INPUT_PROXIES_FILE):
                return

            with open(INPUT_PROXIES_FILE, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # 2. Filter out the bad proxy
            new_lines = []
            removed = False
            for line in lines:
                # Basic check: if the proxy string is in the line
                if proxy_to_remove in line:
                    removed = True
                    continue
                new_lines.append(line)

            # 3. Write back ONLY if we actually removed something (to save I/O)
            if removed:
                with open(INPUT_PROXIES_FILE, 'w', encoding='utf-8') as f:
                    f.writelines(new_lines)
                logging.warning(
                    f"🗑️ PERMANENTLY DELETED proxy {proxy_to_remove} from {INPUT_PROXIES_FILE}")

        except Exception as e:
            logging.error(f"Failed to remove proxy from file: {e}")


def get_already_scraped_urls(filepath):
    """Reads the output CSV and returns a set of URLs that are already scraped."""
    scraped = set()
    if not os.path.exists(filepath):
        return scraped

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "url" in row and row["url"].strip():
                    scraped.add(row["url"].strip())
    except Exception as e:
        logging.warning(f"Could not read existing progress: {e}")

    return scraped


def get_skipped_urls(filepath):
    """Reads the skipped links file and returns a set of URLs without odds."""
    skipped = set()
    if not os.path.exists(filepath):
        return skipped

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                url = line.strip()
                if url:
                    skipped.add(url)
    except Exception as e:
        logging.warning(f"Could not read skipped links: {e}")

    return skipped


def save_skipped_url(url):
    """Appends a URL to the skipped links file in a thread-safe manner."""
    with SKIPPED_FILE_LOCK:
        try:
            with open(SKIPPED_LINKS_FILE, 'a', encoding='utf-8') as f:
                f.write(url + '\n')
        except Exception as e:
            logging.error(f"Failed to save skipped URL: {e}")


def save_processed_url(url):
    """Appends a successfully processed URL to the tracking file."""
    with PROCESSED_FILE_LOCK:
        try:
            with open(PROCESSED_LINKS_FILE, 'a', encoding='utf-8') as f:
                f.write(url + '\n')
        except Exception as e:
            logging.error(f"Failed to save processed URL: {e}")


def save_failed_url(url):
    """Appends a URL that failed all retries to the tracking file."""
    with FAILED_FILE_LOCK:
        try:
            with open(FAILED_LINKS_FILE, 'a', encoding='utf-8') as f:
                f.write(url + '\n')
        except Exception as e:
            logging.error(f"Failed to save failed URL: {e}")


def load_tracking_file(filepath):
    """Loads URLs from a simple text tracking file (fast, no CSV parsing)."""
    urls = set()
    if not os.path.exists(filepath):
        return urls
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                url = line.strip()
                if url:
                    urls.add(url)
    except Exception as e:
        logging.warning(f"Could not read tracking file {filepath}: {e}")
    return urls


def get_paranoid_driver(proxy_url):
    """Creates a Selenium driver with strict proxy settings and stealth headers."""
    chrome_options = Options()

    # 1. IP MASKING
    chrome_options.add_argument(f'--proxy-server={proxy_url}')

    # 2. HEADLESS MODE (Stability with SOCKS5 proxies)
    chrome_options.add_argument("--headless=new")

    # 3. STEALTH MODE (HEADERS & FEATURES)
    chrome_options.add_argument(
        "--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--disable-infobars")
    chrome_options.add_argument("--start-maximized")

    # 3. IGNORE SSL/PRIVACY ERRORS (CRITICAL FIX)
    chrome_options.add_argument('--ignore-certificate-errors')
    chrome_options.add_argument('--ignore-ssl-errors')
    chrome_options.add_argument('--allow-running-insecure-content')

    # Random User-Agent (Basic rotation, can be expanded)
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0"
    ]
    chrome_options.add_argument(f"user-agent={random.choice(user_agents)}")

    # 4. PERFORMANCE OPTIMIZATIONS
    # Disable images
    prefs = {"profile.managed_default_content_settings.images": 2}
    chrome_options.add_experimental_option("prefs", prefs)

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver


def verify_proxy_ip(driver, expected_proxy):
    """
    PARANOID CHECK: Verifies the browser's public IP matches the proxy.
    Returns tuple: (success: bool, should_remove_proxy: bool)
    - (True, False) = IP verified, don't remove
    - (False, True) = Real IP mismatch, REMOVE proxy
    - (False, False) = Crash/timeout, DON'T remove (not proxy's fault)
    """
    try:
        driver.get(VERIFY_IP_URL)
        page_content = driver.find_element(By.TAG_NAME, "body").text
        # Very basic check: just ensure we got JSON back and it's not empty
        if "ip" in page_content:
            logging.info(
                f"IP Verified via Proxy: {expected_proxy} -> content: {page_content}")
            return (True, False)  # Success, don't remove
        logging.warning(f"Proxy check failed content: {page_content}")
        return (False, True)  # Real failure, remove proxy
    except Exception as e:
        logging.error(f"Failed to verify IP for proxy {expected_proxy}: {e}")
        # Crash/timeout - NOT the proxy's fault, don't remove it
        return (False, False)


def detect_captcha(page_source):
    """Detects if CAPTCHA is present in the page."""
    captcha_indicators = [
        'captcha',
        'recaptcha',
        'g-recaptcha',
        'cf-challenge',  # CloudFlare challenge
        'challenge-form',
        'hcaptcha',
        'are you a robot',
        'verify you are human',
        'security check'
    ]

    page_lower = page_source.lower()
    for indicator in captcha_indicators:
        if indicator in page_lower:
            return True
    return False


def extract_numbers_from_string(text):
    """Extracts odds (float) from a messy string."""
    try:
        return text.strip().split()[0]
    except:
        return ""


def parse_match_details(html):
    """Parses the page source with BeautifulSoup to extract ALL data."""
    soup = BeautifulSoup(html, 'html.parser')
    data = {}

    # --- HEADER INFO ---
    try:
        header_box = soup.select_one("div#center .box.boxBasic.lGray")
        if header_box:
            # Example: 25.11.2025, 20:05, Bogotá challenger, 1. round, clay
            info_text = header_box.get_text(" ", strip=True)
            parts = info_text.split(",")
            data['date_str'] = parts[0].strip() if len(parts) > 0 else ""
            data['time_str'] = parts[1].strip() if len(parts) > 1 else ""

            # Tournament often in an 'a' tag
            tourn_link = header_box.select_one("a")
            data['tournament'] = tourn_link.text.strip() if tourn_link else ""

            # Surface is often the last part
            data['surface'] = parts[-1].strip() if len(parts) > 0 else ""

            # Round
            data['round'] = parts[-2].strip() if len(parts) > 2 else ""

        # Player Names
        players = soup.select("table.result.gDetail th.plName a")
        if len(players) >= 2:
            data['player_1'] = players[0].text.strip()
            data['player_2'] = players[1].text.strip()
        else:
            # Fallback title
            h1 = soup.select_one("h1")
            data['match_title'] = h1.text.strip() if h1 else "Unknown"

    except Exception as e:
        logging.error(f"Error parsing header: {e}")

    # --- ODDS EXTRACTION ---
    extracted_odds = []

    # We look for all odds tables.
    # Usually: Match Winner (tabs-1), O/U (tabs-2), AH (tabs-3), CS (tabs-4)
    # The ID structure is oddsMenu-X-data

    odds_containers = soup.select("div[id^='oddsMenu-'][id$='-data']")

    for container in odds_containers:
        cont_id = container.get('id', '')
        # Determine Market Type roughly
        market_type = "Unknown"
        if "oddsMenu-1-data" in cont_id:
            market_type = "Match Winner"
        elif "oddsMenu-2-data" in cont_id:
            market_type = "Over/Under"
        elif "oddsMenu-3-data" in cont_id:
            market_type = "Asian Handicap"
        elif "oddsMenu-4-data" in cont_id:
            market_type = "Correct Score"

        rows = container.select("table.result tr.one, table.result tr.two")
        for row in rows:
            try:
                # Bookmaker Name
                bookie_node = row.select_one("td.first a.icons .t")
                bookie_name = bookie_node.text.strip() if bookie_node else "Unknown"

                # If O/U or AH, there might be a "Total/Handicap" value called 'value'
                val_node = row.select_one("td.value")
                handicap_val = val_node.text.strip() if val_node else ""

                # Odds Cells (k1, k2, etc.)
                # We iterate all 'odds-in' divs to find opening/closing
                odds_cells = row.select("div.odds-in")

                # Selection Names: k1 usually Home/Over, k2 Away/Under.
                # This is heuristic. Ideally we map columns to headers, but keeping it generic for "ALL" data is safer.

                for idx, cell in enumerate(odds_cells):
                    selection_name = f"Selection_{idx+1}"
                    closing_odd = ""
                    opening_odd = ""

                    # Closing Odd is the direct text of the cell (excluding children)
                    # Actually parsed better by taking the first text node or removing child div
                    closing_odd = cell.find(text=True, recursive=False)
                    if not closing_odd:
                        closing_odd = cell.get_text().split("Opening")[
                            0]  # Fallback

                    closing_odd = closing_odd.strip()

                    # Opening Odd is hidden in .odds-change-div -> tr -> td.bold (second one usually)
                    # Structure: <tr><td>Date</td><td class="bold">ODD</td></tr>
                    change_table = cell.select_one(".odds-change-div table")
                    if change_table:
                        # Find the row with "Opening odds"
                        # Actually the structure is often: row 1 = current change, row 2 = title "Opening odds", row 3 = opening value
                        # Let's just find the last '.bold' in the table, or look specifically
                        all_bolds = change_table.select("td.bold")
                        if all_bolds:
                            # Usually the last bold is the opening
                            opening_odd = all_bolds[-1].text.strip()

                    # Store Row
                    item = {
                        "market_type": market_type,
                        "bookmaker": bookie_name,
                        "selection": selection_name + (f" ({handicap_val})" if handicap_val else ""),
                        "closing_odd": closing_odd,
                        "opening_odd": opening_odd
                    }
                    extracted_odds.append(item)

            except Exception as e:
                pass  # Skip bad rows

    return data, extracted_odds


def scrape_link(url, all_proxies):
    """Worker function to scrape a single URL with retries and rotation."""
    driver = None
    results = []

    for attempt in range(MAX_RETRIES):
        # 0. Check for shutdown or CAPTCHA
        if SHUTDOWN_FLAG.is_set() or CAPTCHA_DETECTED.is_set():
            if CAPTCHA_DETECTED.is_set():
                logging.critical("CAPTCHA detected - aborting this worker")
            return results

        # 0.5 Safety Check: Minimum Proxies
        if len(all_proxies) < MIN_PROXIES_THRESHOLD or not all_proxies:
            print("\n" + "="*80)
            print("🛑 SHUTDOWN: TUTTI I PROXY SONO ESAURITI!")
            print("="*80)
            logging.critical(
                f"CRITICAL: Active proxies ({len(all_proxies)}) dropped below threshold ({MIN_PROXIES_THRESHOLD}). STOPPING ALL PROCESSES.")
            SHUTDOWN_FLAG.set()
            # Exit code 2 = proxy exhausted, orchestrator will restart
            os._exit(2)

        # Pick a random proxy for this attempt
        current_proxy = random.choice(all_proxies)

        try:
            logging.info(
                f"Attempt {attempt + 1}/{MAX_RETRIES} for {url} using proxy {current_proxy}")

            # 1. Setup Driver
            driver = get_paranoid_driver(current_proxy)

            # 2. Verify IP
            ip_verified, should_remove = verify_proxy_ip(driver, current_proxy)
            if not ip_verified:
                if should_remove:
                    # Real IP mismatch - REMOVE permanently
                    logging.warning(
                        f"IP Verification FAILED (real mismatch) for {current_proxy}. PERMANENTLY Removing...")

                    # Remove from Memory
                    try:
                        all_proxies.remove(current_proxy)
                    except ValueError:
                        pass

                    # Remove from File
                    remove_proxy_permanently(current_proxy)
                else:
                    # Crash/timeout - NOT proxy's fault, just skip this attempt
                    logging.warning(
                        f"IP Verification crashed for {current_proxy}. Skipping attempt (proxy NOT removed).")

                driver.quit()
                driver = None
                continue

            # 3. Access Target
            logging.info(f"Accessing {url} ...")
            driver.get(url)

            # 3.5. Check for CAPTCHA IMMEDIATELY
            page_source = driver.page_source
            if detect_captcha(page_source):
                logging.critical(
                    f"\n\n{'='*80}\n"
                    f"🚨 CAPTCHA DETECTED on {url}\n"
                    f"🛑 STOPPING ALL WORKERS IMMEDIATELY\n"
                    f"{'='*80}\n"
                )
                CAPTCHA_DETECTED.set()
                SHUTDOWN_FLAG.set()
                driver.quit()
                os._exit(1)  # Force immediate exit

            # 4. Wait for Odds
            try:
                WebDriverWait(driver, TIMEOUT_SEVEN_SECONDS).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "result"))
                )
            except:
                logging.warning(
                    f"Page {url} loaded but generic wait timed out or no results.")

            # 5. Extract Data
            page_html = driver.page_source
            match_info, odds_list = parse_match_details(page_html)

            # Save to skipped_links.txt if no odds found
            if not odds_list:
                logging.warning(
                    f"No odds found for {url}. Saving to skipped_links.txt.")
                save_skipped_url(url)

            # 6. Flatten for CSV
            for odd in odds_list:
                row = {
                    "url": url,
                    **match_info,
                    **odd
                }
                results.append(row)

            logging.info(f"Success {url}: Found {len(results)} odds lines.")
            save_processed_url(url)  # Track successful scrape

            # 7. Sleep (Paranoid)
            time.sleep(random.uniform(SAFETY_DELAY_MIN, SAFETY_DELAY_MAX))

            # If we reached here, success! Break the retry loop.
            break

        except Exception as e:
            error_str = str(e).lower()
            # Check if this is a proxy connection failure
            proxy_error_indicators = [
                'err_proxy_connection_failed',
                'err_tunnel_connection_failed',
                'err_socks_connection_failed',
                'proxy_connection_failed',
                'connection refused',
                'connection timed out',
                'no route to host',
                'network is unreachable'
            ]

            is_proxy_error = any(
                indicator in error_str for indicator in proxy_error_indicators)

            if is_proxy_error:
                logging.warning(
                    f"🗑️ PROXY ERROR on {url} attempt {attempt+1}: {e}")
                logging.warning(
                    f"🗑️ REMOVING DEAD PROXY: {current_proxy}")

                # Remove from Memory
                try:
                    all_proxies.remove(current_proxy)
                except ValueError:
                    pass

                # Remove from File
                remove_proxy_permanently(current_proxy)

                # Immediate check after removal
                if len(all_proxies) < MIN_PROXIES_THRESHOLD or not all_proxies:
                    print("\n" + "="*80)
                    print("🛑 SHUTDOWN: TUTTI I PROXY SONO ESAURITI!")
                    print("="*80)
                    logging.critical(
                        "ALL PROXIES EXHAUSTED AFTER REMOVAL! STOPPING ALL PROCESSES.")
                    SHUTDOWN_FLAG.set()
                    os._exit(2)  # Exit code 2 = proxy exhausted
            else:
                # Other errors - just log, don't remove proxy
                logging.warning(
                    f"Error scraping {url} on attempt {attempt+1}: {e}")
        finally:
            if driver:
                try:
                    driver.quit()
                except:
                    pass
            driver = None

    # After all retries, if no results and not skipped, mark as failed
    if not results:
        save_failed_url(url)
        logging.warning(f"FAILED after {MAX_RETRIES} attempts: {url}")

    return results


def main():
    # 1. Setup
    proxies = load_file_lines(INPUT_PROXIES_FILE)
    links = load_file_lines(INPUT_LINKS_FILE)

    if not proxies:
        logging.critical("NO PROXIES FOUND. Aborting for security.")
        # Exit code 2 = proxy exhausted, triggers orchestrator restart
        sys.exit(2)
    if not links:
        logging.critical("No links found.")
        return

    logging.info(f"Loaded {len(proxies)} proxies and {len(links)} links.")

    # 1.5 RESUME LOGIC - FAST (using tracking files instead of parsing huge CSV)
    processed_urls = load_tracking_file(PROCESSED_LINKS_FILE)
    skipped_urls = load_tracking_file(SKIPPED_LINKS_FILE)
    failed_urls = load_tracking_file(FAILED_LINKS_FILE)

    exclude_urls = processed_urls | skipped_urls | failed_urls

    if exclude_urls:
        logging.info(
            f"Resume: {len(processed_urls)} processed, {len(skipped_urls)} skipped, {len(failed_urls)} failed. Filtering...")
        initial_count = len(links)
        links = [l for l in links if l not in exclude_urls]
        logging.info(
            f"Filtered {initial_count} -> {len(links)} links remaining to scrape.")

    if not links:
        logging.info("ALL LINKS ALREADY SCRAPED. Nothing to do.")
        return

    # 2. CSV Init
    # We verify the headers based on our data structure
    fieldnames = ["url", "date_str", "time_str", "tournament", "surface", "round",
                  "player_1", "player_2", "match_title",
                  "market_type", "bookmaker", "selection", "closing_odd", "opening_odd"]

    # Check if we need to write headers (new file) or just append
    write_header = not os.path.exists(OUTPUT_FILE)

    # Open in APPEND mode if it exists, otherwise WRITE (create)
    open_mode = 'a' if os.path.exists(OUTPUT_FILE) else 'w'

    if write_header:
        with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    # 3. Execution Loop
    # We use a ThreadPool but we must be careful.
    # Since we need to 'assign' proxies, we can just map simple indices.

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for i, link in enumerate(links):
                if SHUTDOWN_FLAG.is_set():
                    break
                # Pass ALL proxies so the worker can rotate if needed
                futures.append(executor.submit(scrape_link, link, proxies))

            for future in concurrent.futures.as_completed(futures):
                if SHUTDOWN_FLAG.is_set():
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
                try:
                    data_rows = future.result(timeout=1)
                except concurrent.futures.TimeoutError:
                    continue
                except Exception:
                    continue

                if data_rows:
                    # Write immediately to file (append mode)
                    with open(OUTPUT_FILE, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        for row in data_rows:
                            # Ensure only valid fields are written
                            clean_row = {k: row.get(k, "") for k in fieldnames}
                            writer.writerow(clean_row)
    except KeyboardInterrupt:
        print("\n🛑 INTERRUPTED - Exiting...")
        SHUTDOWN_FLAG.set()
        os._exit(0)

    logging.info("ALL TASKS COMPLETED.")


if __name__ == "__main__":
    main()
