"""
exporter.py
-----------
Per-process CSV + XLSX export of every generation.
Supports multi-medicine extractions per query with dedicated additional_instruction column.
"""
import os
import re
import datetime
import pandas as pd

import config


def parse_output_fields(output: str, query: str = None) -> list[dict]:
    """Parse structured fields (Drug_name, strength, frequency, duration, route, instruction, additional_instruction)
    from the raw model generation output. Supports multiple medicine blocks per query."""
    blocks_raw = re.split(r"(?i)(?=Drug_name\s*:)", output.strip())
    parsed_items = []

    pattern = r"(?i)^(Drug_name|strength|frequency|duration|route|additional_instruction|instruction)\s*:\s*(.*)$"

    for block in blocks_raw:
        if not block.strip():
            continue

        fields = {
            "Drug_name": "NONE",
            "strength": "NONE",
            "frequency": "NONE",
            "duration": "NONE",
            "route": "NONE",
            "instruction": "NONE",
            "additional_instruction": "NONE",
        }

        has_data = False
        for line in block.strip().splitlines():
            match = re.match(pattern, line.strip())
            if match:
                key_name = match.group(1).strip()
                val = match.group(2).strip().strip('"').strip("'").strip()
                for canonical_key in fields.keys():
                    if canonical_key.lower() == key_name.lower():
                        fields[canonical_key] = val if val else "NONE"
                        has_data = True
                        break

        if not has_data or fields["Drug_name"] == "NONE":
            continue

        # Strip quotes and placeholders from all parsed field values
        for k in fields:
            if isinstance(fields[k], str):
                cleaned_v = fields[k].strip('"').strip("'").strip()
                if cleaned_v.startswith("<") and cleaned_v.endswith(">"):
                    cleaned_v = "NONE"
                fields[k] = cleaned_v

        # Strip speech fillers from Drug_name if present
        if fields["Drug_name"] != "NONE":
            fields["Drug_name"] = re.sub(r"(?i)\b(tabs|tab|tablets|tablet|pills|pill|capsules|capsule)\b", "", fields["Drug_name"]).strip()
            fields["Drug_name"] = re.sub(r"\s+", " ", fields["Drug_name"]).strip()

        parsed_items.append(fields)

    # Deduplicate: small models sometimes loop and repeat medicine blocks.
    seen_names = set()
    deduped = []
    for item in parsed_items:
        key = item["Drug_name"].lower().strip()
        if key != "none" and key in seen_names:
            continue
        seen_names.add(key)
        deduped.append(item)
    parsed_items = deduped if deduped else parsed_items

    # Fallback if no valid blocks found
    if not parsed_items:
        parsed_items.append({
            "Drug_name": "NONE",
            "strength": "NONE",
            "frequency": "NONE",
            "duration": "NONE",
            "route": "NONE",
            "instruction": "NONE",
            "additional_instruction": "NONE",
        })

    # Post-processor dosage attachment for single med queries
    if query and len(parsed_items) == 1:
        fields = parsed_items[0]
        dosages = re.findall(r"\b\d+(?:\.\d+)?\s*(?:mg|g|mcg|ml|l|drops)\b", query, re.IGNORECASE)
        if len(dosages) == 1:
            single_dose = dosages[0].strip()
            num_part = re.search(r"\d+(?:\.\d+)?", single_dose).group(0)
            unit_part = re.search(r"[a-zA-Z]+", single_dose).group(0)
            formatted_dose = f"{num_part} {unit_part}"

            if fields["Drug_name"] != "NONE" and num_part not in fields["Drug_name"]:
                fields["Drug_name"] = f"{fields['Drug_name']} {formatted_dose}"
            elif fields["Drug_name"] == "NONE":
                fields["Drug_name"] = formatted_dose

            fields["strength"] = "NONE"
        elif len(dosages) >= 2 and fields["Drug_name"] != "NONE" and fields["strength"] == "NONE":
            dose1 = dosages[0].strip()
            dose2 = dosages[1].strip()
            num_part1 = re.search(r"\d+(?:\.\d+)?", dose1).group(0)
            num_part2 = re.search(r"\d+(?:\.\d+)?", dose2).group(0)
            unit_part2 = re.search(r"[a-zA-Z]+", dose2).group(0)
            formatted_dose2 = f"{num_part2} {unit_part2}"

            if num_part1 in fields["Drug_name"]:
                fields["strength"] = formatted_dose2

    return parsed_items


def new_output_paths(process_name: str):
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in process_name) or "process"
    base = f"{safe_name}_{stamp}"
    csv_path = os.path.join(config.OUTPUT_DIR, base + ".csv")
    xlsx_path = os.path.join(config.OUTPUT_DIR, base + ".xlsx")
    return csv_path, xlsx_path


def append_generation(
    csv_path: str,
    xlsx_path: str,
    query: str,
    output: str,
    generation_time: float = None,
    llm_model_used: str = None,
):
    parsed_items = parse_output_fields(output, query=query)
    now = datetime.datetime.now()

    rows = []
    for parsed in parsed_items:
        rows.append({
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "query": query,
            "Drug_name": parsed["Drug_name"],
            "strength": parsed["strength"],
            "frequency": parsed["frequency"],
            "duration": parsed["duration"],
            "route": parsed["route"],
            "instruction": parsed["instruction"],
            "additional_instruction": parsed["additional_instruction"],
            "llm_model_used": llm_model_used if llm_model_used else "N/A",
            "generation_time_sec": generation_time if generation_time is not None else "N/A",
        })

    df_rows = pd.DataFrame(rows)

    # CSV: append all extracted rows
    header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    df_rows.to_csv(csv_path, mode="a", header=header, index=False)

    # XLSX: read-modify-write
    try:
        if os.path.exists(xlsx_path) and os.path.getsize(xlsx_path) > 0:
            existing = pd.read_excel(xlsx_path)
            full = pd.concat([existing, df_rows], ignore_index=True)
        else:
            full = df_rows
        full.to_excel(xlsx_path, index=False)
    except Exception:
        pass
