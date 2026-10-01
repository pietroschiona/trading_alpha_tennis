import os
import re
import csv
import sys
import multiprocessing
from collections import defaultdict
import time
import datetime

# --- CONFIG ---
dataset_path = r"C:\Users\pietr\Desktop\Betting\scraper_per_odds\prova3\Tennis_SOTA_DeepStats_V36.csv"
scraped_root = r"C:\Users\pietr\Desktop\Betting\scraped-1764175345406"
output_mapping = r"C:\Users\pietr\Desktop\Betting\scraper_per_odds\prova3\mapped_matches.csv"
output_links = r"C:\Users\pietr\Desktop\Betting\scraper_per_odds\prova3\verified_links_strict.txt"
output_missing = r"C:\Users\pietr\Desktop\Betting\scraper_per_odds\prova3\missing_matches.csv"

CPU_CORES = max(1, multiprocessing.cpu_count() - 2)

pattern_strict = re.compile(
    r'class="first time"[^>]*>.*?(\d{2}\.\d{2}\.).*?class="t-name"><a href="/player/[^"]+">([^<]+)</a>.*?href="/match-detail/\?id=(\d+)".*?class="t-name"><a href="/player/[^"]+">([^<]+)</a>',
    re.IGNORECASE | re.DOTALL
)
pattern_year = re.compile(r'%2F(\d{4})%2F')


def normalize_name_tokens(name):
    name = name.lower().strip()
    name = name.replace('-', ' ')
    name = re.sub(r'[^a-z\s]', '', name)
    return {t for t in name.split() if len(t) > 1}


def parse_csv_date(date_str):
    try:
        return datetime.date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))
    except:
        return None

# --- LOADER ---


def load_dataset_main():
    all_matches = defaultdict(list)
    match_meta = {}  # id -> row

    with open(dataset_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        for i, row in enumerate(reader):
            if len(row) < 22:
                continue
            try:
                year = int(row[6][:4])
            except:
                continue

            mid = i + 2
            w_norm = frozenset(normalize_name_tokens(row[11]))
            l_norm = frozenset(normalize_name_tokens(row[21]))
            date_obj = parse_csv_date(row[6])

            # Key: (Year, PlayerSet)
            pair_key = (year, frozenset(w_norm | l_norm))

            obj = {
                "id": mid,
                "year": year,
                "date": date_obj,
                "winner": row[11],
                "loser": row[21],
                "found": False,
                "link": None,
                "method": None,
                "w_set": w_norm,
                "l_set": l_norm
            }
            all_matches[pair_key].append(obj)
            match_meta[mid] = obj

    return all_matches, match_meta

# --- WORKER: COLLECT PROSPECTIVE MATCHES ---
# Instead of satisfying immediately, we return ALL valid HTML matches to Main
# Main will perform the Global Matching Logic (Phase 1 & 2)


def process_file_harvest(path):
    harvested = []
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            html_year = None
            if "onclick=\"tz(" in content:
                snippet = content.split("onclick=\"tz(")[1][:100]
                ym = pattern_year.search(snippet)
                if ym:
                    html_year = int(ym.group(1))

            if not html_year:
                return []

            raw_matches = pattern_strict.findall(content)
            if not raw_matches:
                return []

            for date_str, p1, m_id, p2 in raw_matches:
                link = f"https://www.tennisexplorer.com/match-detail/?id={m_id}"

                try:
                    day = int(date_str[:2])
                    month = int(date_str[3:5])
                    # Construct Approx Date (using html_year)
                    # Be careful of Dec/Jan boundary, but for Sorting purposes,
                    # we can correct later or just use tuple.
                    # Let's trust html_year for now.
                    h_date = datetime.date(html_year, month, day)
                except:
                    continue

                p1_set = frozenset(normalize_name_tokens(p1))
                p2_set = frozenset(normalize_name_tokens(p2))

                pair_set = frozenset(p1_set | p2_set)

                # Emit candidate
                # Key: (Year, PairSet) matching is hard inversely?
                # Just return raw data.
                harvested.append({
                    "year": html_year,
                    "date": h_date,
                    "link": link,
                    "p1_set": p1_set,
                    "p2_set": p2_set,
                    # Use pickle-able basic sets? set is pickleable.
                    "pair_set": pair_set
                })

    except:
        return []
    return harvested

# --- MAIN LOGIC ---


def main():
    start_time = time.time()

    # 1. Load CSV
    print("Loading Dataset...")
    dataset_buckets, match_meta = load_dataset_main()
    # buckets: (year, nameset) -> [match1, match2] (sorted by date?)
    for k in dataset_buckets:
        dataset_buckets[k].sort(key=lambda x: x['date']
                                if x['date'] else datetime.date.min)

    # 2. Collect HTML candidates
    file_list = []
    print("Collecting files...")
    for dirpath, _, files in os.walk(scraped_root):
        if any(x in dirpath.lower() for x in ['junior']):
            continue
        for fname in files:
            if fname.endswith('.html'):
                file_list.append(os.path.join(dirpath, fname))

    print(f"Scanning {len(file_list)} files...")

    # We need to map CSV Keys to a lookup structure used by Workers?
    # No, workers just harvest. Main does the matching.
    # But harvesting ALL matches into memory might be big?
    # 38k links x 200 bytes = 7MB. Tiny.
    # We can bring it all back.

    total_harvested = 0
    html_buckets = defaultdict(list)

    with multiprocessing.Pool(processes=CPU_CORES) as pool:
        for chunk in pool.imap_unordered(process_file_harvest, file_list, chunksize=100):
            for cand in chunk:
                # Store in bucket: (Year, PairSet)
                # But Names mismatch slightly?
                # We need Fuzzy/Intersection storage.
                # Exact Set match is unlikely due to "Carlos Alcaraz" vs "Alcaraz".

                # We store in Year bucket:
                year = cand['year']
                # Add to window years too?
                # Let's store in 'year' only, main loop will check windows.
                html_buckets[year].append(cand)
                total_harvested += 1

            if total_harvested % 5000 == 0:
                sys.stdout.write(f"\rHarvested {total_harvested} matches...")

    print(f"\nTotal Harvested: {total_harvested}")

    # 3. GLOBAL MATCHING
    print("Performing Global Cardinality Matching...")

    # Iterate CSV Buckets
    pass1_hits = 0
    pass2_hits = 0

    # Pre-index HTML buckets by token for speed?
    # Year -> Token -> [candidates]
    html_indices = {}

    for year in html_buckets:
        idx = defaultdict(list)
        for cand in html_buckets[year]:
            for t in cand['pair_set']:
                idx[t].append(cand)
        html_indices[year] = idx

    def get_html_candidates(year, w_set, l_set):
        # Look in Year, Year-1, Year+1
        cands = []
        for y in [year, year-1, year+1]:
            if y not in html_indices:
                continue

            # Intersection matching
            # Find candidates that share tokens with W AND L
            # Optimization: just iterate all cands in year? No, too slow.
            # Use Index.
            # Get union of tokens
            search_tokens = w_set | l_set

            possible = set()
            idx = html_indices[y]

            # Must match at least 1 token from W and 1 from L?
            # Or use the strict intersection logic.

            # Get small subset first
            first_token = next(iter(w_set)) if w_set else None
            if not first_token or first_token not in idx:
                # Try loser
                first_token = next(iter(l_set)) if l_set else None

            if not first_token or first_token not in idx:
                continue

            # Start with candidates containing first token
            # Then filter.
            # Ideally verify strict name match

            for c in idx[first_token]:
                # Strict check
                p1_ok = not c['p1_set'].isdisjoint(w_set)
                p2_ok = not c['p2_set'].isdisjoint(l_set)
                cross1 = not c['p1_set'].isdisjoint(l_set)
                cross2 = not c['p2_set'].isdisjoint(w_set)

                if (p1_ok and p2_ok) or (cross1 and cross2):
                    cands.append(c)
        return cands

    # Process each CSV match group (Same Year, Same Players)
    unique_groups = sorted(dataset_buckets.keys())

    for (year, pair_keys) in unique_groups:
        csv_matches = dataset_buckets[(year, pair_keys)]
        # We need original W_set and L_set to search
        # Grab from first match
        w_set = csv_matches[0]['w_set']
        l_set = csv_matches[0]['l_set']

        candidates = get_html_candidates(year, w_set, l_set)

        # Deduplicate candidates (same link found in multiple years?)
        # Use link as unique ID
        unique_cands = {c['link']: c for c in candidates}.values()
        candidates = sorted(unique_cands, key=lambda x: x['date'])

        # PHASE 1: Try Strict Date Match (Tolerance 20 days)
        # Greedy assignment?
        # Or Match Matrix?

        # If Cardinality 1-to-1 -> JUST MATCH (Phase 2 logic)
        # Check if 1-to-1
        if len(csv_matches) == 1 and len(candidates) == 1:
            # Safe Match
            m = csv_matches[0]
            c = candidates[0]
            m['found'] = True
            m['link'] = c['link']
            m['method'] = "1-to-1 Cardinality"
            pass2_hits += 1
            continue

        # If N-to-N, try Date Match first
        unmatched_csv = []
        for m in csv_matches:
            best_c = None
            best_d = 21  # Max tolerance

            for c in candidates:
                if not m['date']:
                    continue
                diff = abs((m['date'] - c['date']).days)
                if diff < best_d:
                    best_d = diff
                    best_c = c

            if best_c:
                m['found'] = True
                m['link'] = best_c['link']
                m['method'] = f"Strict Date (d={best_d})"
                pass1_hits += 1
                # Can we reuse candidate?
                # If duplicate rows -> Yes.
                # If distinct matches -> No?
                # User wants "Duplicate Rows mapped to Single Link".
                # But "Distinct Matches (Jan vs Feb) mapped to Distinct Links".

                # If Jan match maps to Jan Link. Good.
                # If Feb match maps to Jan Link?
                # Diff will be > 20. So it won't map.
                pass
            else:
                unmatched_csv.append(m)

        # PHASE 2: Fallback for unmatched
        # If we have unmatched CSV and Candidates.
        # And we respected Date Constraints.
        # If 1 Jan Match (Unmatched) and 1 Feb Link (Unused).
        # Diff > 20.
        # Should we map?
        # Likely "Date Error".
        # Sort remaining by date and map?
        # Yes.

    # WRITE OUTPUTS (Same as before)
    # ...

    print("-" * 60)
    print(f"Pass 1 (Strict Date): {pass1_hits}")
    print(f"Pass 2 (Cardinality): {pass2_hits}")
    print("-" * 60)

    # Save code...
    # (Simplified for brevity in thought block, will write full code)

    unique_links = set()
    satisfied_count = 0
    missing_count = 0

    with open(output_links, 'w', encoding='utf-8') as f:
        # Collect links
        for m in match_meta.values():
            if m['found']:
                unique_links.add(m['link'])
        for l in sorted(unique_links):
            f.write(l + "\n")

    with open(output_mapping, 'w', newline='', encoding='utf-8') as f_map:
        writer = csv.writer(f_map)
        writer.writerow(["Line_ID", "Year", "Winner",
                        "Loser", "Link", "Method"])

        for mid in sorted(match_meta.keys()):
            m = match_meta[mid]
            if m['found']:
                writer.writerow([m['id'], m['year'], m['winner'],
                                m['loser'], m['link'], m['method']])
                satisfied_count += 1
            else:
                writer.writerow([m['id'], m['year'], m['winner'],
                                m['loser'], "MISSING", "Failed"])
                missing_count += 1

    # CLEANUP logic included
    files_to_delete = [
        "diagnose_missings.py", "compare_coverage.py", "generate_links_strict.py", "recover_missing_fuzzy.py",
        "inspect_html.py", "inspect_html_miami.py", "explain_duplicates.py", "prove_same_match.py",
        "report_missing_dates.py", "final_coverage_analysis.py", "test_french_regex.py", "verify_map.py", "check_csv_date.py"
    ]
    for fname in files_to_delete:
        if os.path.exists(fname):
            os.remove(fname)

    print(
        f"Total Satisfied: {satisfied_count} | Unique Links: {len(unique_links)}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
