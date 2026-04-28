"""
step_00_sharepoint.py — Optional SharePoint download step.

Prompts the user whether to download the latest Excel source files from SharePoint.
If yes, authenticates via MSAL device-code flow (browser sign-in) and re-uses a
cached token if one is still valid. Downloaded files are saved to:
  1. pipeline/outputs/raw_excel/   ← always (new pipeline copy)
  2. The original SpkData paths     ← so the rest of the pipeline picks them up

Token cache is stored in pipeline/token_cache.bin and reused across runs.
"""

import os
import sys
import shutil
import time
from pathlib import Path

import requests
import msal

# ── Constants (from sharepoint_extractor.py) ──────────────────────────────────
CLIENT_ID  = 'b8b30bd8-6584-4322-8877-dd67ba1c782f'
TENANT_ID  = 'b22ab0f9-a15b-4346-b250-85f86e98ca7b'
AUTHORITY  = f"https://login.microsoftonline.com/{TENANT_ID}"
SCOPES     = ["https://jettiresources.sharepoint.com/Sites.Read.All"]
SITE_URL_ROSETTA = 'https://jettiresources.sharepoint.com/sites/SparklingData-Rosetta'
SITE_URL_BKMR    = 'https://jettiresources.sharepoint.com/sites/BK-MR'

# ── SpkData base path ──────────────────────────────────────────────────────────
SPKDATA_BASE = "/Users/administration/Library/CloudStorage/OneDrive-JettiResources/PythonProjects/SpkData/Jetti01"

PIPELINE_DIR = Path(__file__).parent
TOKEN_CACHE_FILE = str(PIPELINE_DIR / "token_cache.bin")


# ==============================================================================
# Auth
# ==============================================================================

def _get_session() -> requests.Session | None:
    """
    Acquire an authenticated requests.Session using MSAL device-code flow.
    Re-uses a cached token if still valid; prompts browser sign-in otherwise.
    Token is persisted to pipeline/token_cache.bin for future runs.
    """
    cache = msal.SerializableTokenCache()
    if os.path.exists(TOKEN_CACHE_FILE):
        try:
            with open(TOKEN_CACHE_FILE, 'r') as f:
                cache.deserialize(f.read())
            print("[step_00] Loaded cached token.")
        except Exception as e:
            print(f"[step_00] Could not read token cache: {e}")

    app = msal.PublicClientApplication(CLIENT_ID, authority=AUTHORITY, token_cache=cache)

    token = None
    accounts = app.get_accounts()
    if accounts:
        result = app.acquire_token_silent(SCOPES, account=accounts[0])
        if result and "access_token" in result:
            token = result["access_token"]
            print("[step_00] Reusing cached token — no sign-in needed.")

    if not token:
        flow = app.initiate_device_flow(scopes=SCOPES)
        if "user_code" not in flow:
            print(f"[step_00] ✗ Failed to start device flow: {flow.get('error')}")
            return None
        print(f"\n[step_00] {flow['message']}\n")
        deadline = time.time() + 300
        while time.time() < deadline:
            result = app.acquire_token_by_device_flow(flow)
            if "access_token" in result:
                token = result["access_token"]
                print("[step_00] ✓ Authentication successful.")
                break
            elif "error" in result:
                print(f"[step_00] ✗ Auth error: {result.get('error_description', 'unknown')}")
                return None
            time.sleep(5)

    if not token:
        print("[step_00] ✗ Could not acquire token within timeout.")
        return None

    # Persist cache
    try:
        with open(TOKEN_CACHE_FILE, 'w') as f:
            f.write(cache.serialize())
    except Exception as e:
        print(f"[step_00] ⚠ Could not save token cache: {e}")

    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {token}",
        "Accept": "application/json;odata=verbose",
        "Content-Type": "application/json;odata=verbose",
    })
    return session


# ==============================================================================
# Download helpers
# ==============================================================================

def _list_files(session: requests.Session, site_url: str, folder_path: str) -> list[str]:
    url = f"{site_url}/_api/web/GetFolderByServerRelativeUrl('{folder_path}')/Files"
    try:
        r = session.get(url)
        r.raise_for_status()
        return [f['Name'] for f in r.json()['d']['results']]
    except Exception as e:
        print(f"[step_00]   ⚠ Could not list {folder_path}: {e}")
        return []


def _download_file(
    session: requests.Session,
    site_url: str,
    folder_path: str,
    file_name: str,
    destinations: list[str],
) -> bool:
    """Download one file to all destination paths."""
    url = f"{site_url}/_api/web/GetFolderByServerRelativeUrl('{folder_path}')/Files('{file_name}')/$value"
    try:
        r = session.get(url, stream=True)
        r.raise_for_status()
        # Write to first destination
        first = destinations[0]
        os.makedirs(first, exist_ok=True)
        target = os.path.join(first, file_name)
        with open(target, 'wb') as fh:
            for chunk in r.iter_content(chunk_size=8192):
                fh.write(chunk)
        # Copy to remaining destinations
        for dest in destinations[1:]:
            os.makedirs(dest, exist_ok=True)
            shutil.copy2(target, os.path.join(dest, file_name))
        print(f"[step_00]   ✓ {file_name}")
        return True
    except Exception as e:
        print(f"[step_00]   ✗ {file_name}: {e}")
        return False


# ==============================================================================
# Main download logic
# ==============================================================================

def _do_download(session: requests.Session, raw_excel_dir: str) -> list[str]:
    """
    Download all project Excel files from SharePoint.
    Returns list of file names that could not be downloaded.
    """
    import pandas as pd

    supported = {'xlsx', 'xlsm'}
    failed = []

    spk = SPKDATA_BASE
    datasources_path   = os.path.join(spk, 'Ongoing_Projects')
    input_sheets_path  = os.path.join(spk, 'input_sheets')
    map_columns_path   = os.path.join(spk, 'map_columns')
    terminated_path    = os.path.join(spk, 'terminated_projects')

    # ── 1. Base config files (input_dataframe_maker, formatted_headers) ────────
    print("[step_00] Downloading base config files...")
    base_folder = 'Shared Documents/General/Monthly project reports/standardizations_spkdata'
    base_files  = {
        'input_dataframe_maker': input_sheets_path,
        'formatted_headers':     map_columns_path,
    }
    available = _list_files(session, SITE_URL_BKMR, base_folder)
    for base_name, spk_dest in base_files.items():
        matched = [f for f in available
                   if base_name in f and f.rsplit('.', 1)[-1].lower() in supported]
        if matched:
            _download_file(session, SITE_URL_BKMR, base_folder, matched[0],
                           [raw_excel_dir, spk_dest])
        else:
            print(f"[step_00]   ⚠ {base_name} not found in SharePoint")
            failed.append(base_name)

    # ── 2. Terminated project CSVs ─────────────────────────────────────────────
    print("[step_00] Downloading terminated project files...")
    term_folder = 'Shared Documents/Rosetta/spk_terminated_projects'
    for fname in _list_files(session, SITE_URL_ROSETTA, term_folder):
        if 'terminated_project' in fname and fname.lower().endswith('.csv'):
            _download_file(session, SITE_URL_ROSETTA, term_folder, fname,
                           [raw_excel_dir, terminated_path])

    # ── 3. Project Excel files from input_dataframe_maker ─────────────────────
    maker_path = os.path.join(input_sheets_path, 'input_dataframe_maker.xlsx')
    if not os.path.exists(maker_path):
        print(f"[step_00] ⚠ {maker_path} not found — skipping project file download")
        return failed

    print("[step_00] Downloading project Excel files...")
    df_index = pd.read_excel(maker_path, sheet_name='project_index')
    df_index = df_index[['file_name', 'folder_path']].drop_duplicates(subset=['file_name'])

    for _, row in df_index.iterrows():
        f_name = str(row['file_name']).strip()
        url    = str(row['folder_path']).strip()

        try:
            parts      = url.split('/sites/')
            site_key   = parts[1].split('/')[0]
            site_url   = parts[0] + '/sites/' + site_key
            rel_path   = parts[1].replace(site_key + '/', '')
            folder_path = '/'.join(rel_path.split('/')[:-1])
        except Exception:
            print(f"[step_00]   ⚠ Could not parse URL for {f_name}")
            failed.append(f_name)
            continue

        available = _list_files(session, site_url, folder_path)
        matched = [fn for fn in available
                   if f_name == fn.rsplit('.', 1)[0].strip()
                   and fn.rsplit('.', 1)[-1].lower() in supported]

        if matched:
            ok = _download_file(session, site_url, folder_path, matched[0],
                                [raw_excel_dir, datasources_path])
            if not ok:
                failed.append(f_name)
        else:
            print(f"[step_00]   ⚠ {f_name} not found")
            failed.append(f_name)

    return failed


# ==============================================================================
# Entry point
# ==============================================================================

def run(intermediate_dir: Path = PIPELINE_DIR / "intermediate") -> dict:
    """
    Prompt the user, then optionally download source Excel files from SharePoint.
    Token is cached in pipeline/token_cache.bin and reused if still valid.
    """
    print("\n" + "─" * 60)
    answer = input("  Download latest source files from SharePoint? [y/N] ").strip().lower()
    print("─" * 60)

    if answer not in ('y', 'yes'):
        print("[step_00] Skipped — using existing local files.")
        return {"skipped": True, "failed": []}

    outputs_dir   = PIPELINE_DIR / "outputs"
    raw_excel_dir = str(outputs_dir / "raw_excel")
    os.makedirs(raw_excel_dir, exist_ok=True)

    print("[step_00] Connecting to SharePoint...")
    session = _get_session()
    if not session:
        print("[step_00] ✗ Could not authenticate. Skipping download.")
        return {"skipped": True, "failed": []}

    failed = _do_download(session, raw_excel_dir)

    if failed:
        print(f"\n[step_00] ⚠ {len(failed)} file(s) not downloaded:")
        for f in failed:
            print(f"    • {f}")
    else:
        print("[step_00] ✓ All files downloaded successfully.")

    print(f"[step_00] Raw Excel copies → {raw_excel_dir}")
    return {"skipped": False, "failed": failed, "raw_excel_dir": raw_excel_dir}
