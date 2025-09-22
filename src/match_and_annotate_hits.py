#!/usr/bin/env python3
"""
match_and_annotate_hits.py

Script to:
- Match entries between TopPIC prsm reports (or published hits) and TDAuditor outputs by SourceFile and scan number.
- Annotate matched pairs with a 'hit' column (1 if matched, 0 if not).
- Add instrument info to final output TSV.

Author: [Your Name]
Github: [Your handle]
"""

import pandas as pd

### ---- Config ---- ###
# For each mode, update paths as needed
MODES = [
    {
        "name": "TopFD",
        "prsm_file":   "20241108-1551-TopPIC-FullDB-all_prsm_single.tsv",
        "prsm_col_file": "Data file name",
        "prsm_col_scan": "Scan(s)",
        "td_file":   "/Users/user/PycharmProjects/TDAuditor/20250430-TDAuditor-TSVs-TopFD-FLASH-PSPDXtract/TDAuditor-TopFD-byMSn.tsv",
        "instrument_file": "/Users/user/PycharmProjects/TDAuditor/20250430-TDAuditor-TSVs-TopFD-FLASH-PSPDXtract/TDAuditor-TopFD-byRun.tsv",
        "instrument_save": "tdauditor_topfd_with_instrument.tsv",
        "output_file": "topfd_td_auditor_hit_column.tsv",
        "skiprows": 43,
        "prsm_file_type": "csv"
    },
    {
        "name": "FlashDeconv",
        "prsm_file": "20250213-1551-FLASH-TopPIC-FullDB-all_prsm_single.tsv",
        "prsm_col_file": "Data file name",
        "prsm_col_scan": "Scan(s)",
        "td_file":   "/Users/user/PycharmProjects/TDAuditor/20250430-TDAuditor-TSVs-TopFD-FLASH-PSPDXtract/TDAuditor-FLASHDeconv-byMSn.tsv",
        "instrument_file": "/Users/user/PycharmProjects/TDAuditor/20250430-TDAuditor-TSVs-TopFD-FLASH-PSPDXtract/TDAuditor-FLASHDeconv-byRun.tsv",
        "instrument_save": "tdauditor_flashdeconv_with_instrument.tsv",
        "output_file": "flash_td_auditor_hit_column.tsv",
        "skiprows": 43,
        "prsm_file_type": "csv"
    },
    {
        "name": "PRO-PSPDXtract",
        "prsm_file": "20200811-Published-Identifications_Hit-Report-Sorted.xlsx",
        "prsm_col_file": "File Name",
        "prsm_col_scan": "Fragment Scans",
        "td_file":   "/Users/user/PycharmProjects/TDAuditor/20250430-TDAuditor-TSVs-TopFD-FLASH-PSPDXtract/TDAuditor-PSPDXtract-byMSn.tsv",
        "instrument_file": "/Users/user/PycharmProjects/TDAuditor/20250430-TDAuditor-TSVs-TopFD-FLASH-PSPDXtract/TDAuditor-PSPDXtract-byRun.tsv",
        "instrument_save": "tdauditor_pro_with_instrument.tsv",
        "output_file": "pro_td_auditor_hit_column.tsv",
        "skiprows": 0,
        "prsm_file_type": "excel"
    },
]

def clean_filename_mode(filename, mode, tdauditor_filenames):
    """Cleans filenames according to source type."""
    # TopFD & FlashDeconv: remove '_ms2.msalign'
    if mode in ["TopFD", "FlashDeconv"]:
        filename = filename.replace("_ms2.msalign", "")
    if mode == "PRO-PSPDXtract":
        filename = filename.replace(".raw", "").replace("+", "_")
    # Try to match to known TDAuditor source file names
    for tdf in tdauditor_filenames:
        if tdf in filename:
            return tdf
    return filename

def match_and_annotate(mode_opts):
    # Load prsm file
    print(f"[{mode_opts['name']}] Loading PRSM file ...")
    if mode_opts["prsm_file_type"] == "csv":
        prsm_df = pd.read_csv(mode_opts["prsm_file"], sep="\t",
                              skiprows=mode_opts["skiprows"], engine="python", dtype=str)
    else:
        prsm_df = pd.read_excel(mode_opts["prsm_file"])

    # Load TDAuditor file
    td_df = pd.read_csv(mode_opts["td_file"], delimiter="\t", dtype=str)

    # Collect set of valid TDAuditor filenames
    tdauditor_filenames = set(td_df["SourceFile"])

    # Clean filename column in prsm (new column)
    print(f"[{mode_opts['name']}] Cleaning filenames ...")
    prsm_df["Cleaned SourceFile"] = prsm_df[mode_opts["prsm_col_file"]].apply(
        lambda x: clean_filename_mode(str(x), mode_opts["name"], tdauditor_filenames)
    )

    # Print unmatched files (optional debug)
    unmatched = set(prsm_df["Cleaned SourceFile"]) - tdauditor_filenames
    if unmatched:
        print(f"[{mode_opts['name']}] Unmatched cleaned files ({len(unmatched)}): {list(unmatched)[:5]} ...")

    # Ensure scan numbers are str
    prsm_df[mode_opts["prsm_col_scan"]] = prsm_df[mode_opts["prsm_col_scan"]].astype(str)
    td_df["ScanNumber"] = td_df["ScanNumber"].astype(str)

    # Set of matches
    matching_pairs = set(zip(prsm_df["Cleaned SourceFile"], prsm_df[mode_opts["prsm_col_scan"]]))

    # Add hit column
    print(f"[{mode_opts['name']}] Annotating hits ...")
    td_df["hit"] = td_df.apply(
        lambda row: 1 if (row["SourceFile"], row["ScanNumber"]) in matching_pairs else 0, axis=1
    )

    # Save hit-annotated output
    td_df.to_csv(mode_opts["output_file"], sep="\t", index=False)
    print(f"[{mode_opts['name']}] Saved with hit column: {mode_opts['output_file']} ({td_df['hit'].sum()} hits)")

    # Add instrument info
    print(f"[{mode_opts['name']}] Adding instrument info ...")
    instr_df = pd.read_csv(mode_opts["instrument_file"], delimiter="\t", dtype=str)
    instr_df = instr_df[["SourceFile", "Instrument"]]
    merged = td_df.merge(instr_df, on="SourceFile", how="left")
    merged.to_csv(mode_opts["instrument_save"], sep="\t", index=False)
    print(f"[{mode_opts['name']}] Instrument-annotated: {mode_opts['instrument_save']}")
    return merged  # Optionally return for later processing

if __name__ == "__main__":
    for opts in MODES:
        match_and_annotate(opts)
    print("All done!")