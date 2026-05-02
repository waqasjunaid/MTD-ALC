#from pathlib import Path
#import json
#import re
#import random

#DIAG_DIR = Path("/mnt/data/junaid/linevul/linevul/outputs/diagnosis")
#SRC_DIR  = Path("/mnt/data/junaid/linevul/linevul/outputs/sources")
#OUT_PATH = Path("/mnt/data/junaid/linevul/linevul/outputs/counterfactual_repairs.json")

# Safe replacements for common dangerous APIs
#SAFE_REPLACEMENTS = {
#    "strcpy": "strncpy(dest, src, sizeof(dest)-1); dest[sizeof(dest)-1] = '\\0';",
#    "gets":   "fgets(input, sizeof(input), stdin);",
#    "sprintf": "snprintf(buf, sizeof(buf), fmt, args);"
#}

#all_repairs = []

#for diag_file in DIAG_DIR.glob("*.json"):
#    sample_id = diag_file.stem
#    diagnosis = json.load(open(diag_file))

#    src_file = SRC_DIR / f"{sample_id}.c"
#    if not src_file.exists():
#        continue

#    source = src_file.read_text().splitlines()

#    for d in diagnosis:
#        ln = d.get("line", 0) - 1
#        if ln < 0 or ln >= len(source):
#            continue

#        original = source[ln].strip()
#        feature = d.get("feature", "")
#        diag_type = d.get("type", "")

#        repaired = original

        # Different strategies based on diagnosis type
#        if diag_type == "api_mask" and feature in SAFE_REPLACEMENTS:
#            repaired = SAFE_REPLACEMENTS[feature]
#        elif diag_type in ["var_rename", "token_remove"]:
#            repaired = re.sub(rf"\b{re.escape(feature)}\b", "safe_var", original)
#        elif diag_type == "dead_danger":
            # Add safe equivalent instead of dangerous
#            repaired = original + "\n// safe check: if (len > 0) strcpy_safe(dest, src);"
#        else:
            # Fallback: comment out
#            repaired = "// " + original

        # Very basic semantic check: at least keep ; or }
#        if not repaired.strip().endswith(('{', ';', '}')):
#            repaired += ";"

#        all_repairs.append({
#            "sample": sample_id,
#            "line": ln + 1,
#            "original": original,
#            "repaired": repaired,
#            "feature": feature,
#            "type": diag_type
#        })

#OUT_PATH.write_text(json.dumps(all_repairs, indent=2))
#print(f"Generated {len(all_repairs)} counterfactual repairs")



# repair/generate_counterfactuals.py
from pathlib import Path
import json
import re
import random
import time
from ollama import Client

DIAG_DIR = Path("/mnt/data/junaid/linevul/linevul/outputs/diagnosis")
SRC_DIR  = Path("/mnt/data/junaid/linevul/linevul/outputs/sources")
OUT_PATH = Path("/mnt/data/junaid/linevul/linevul/outputs/counterfactual_repairs.json")

# Local Ollama client (runs on localhost)
client = Client(host='http://localhost:11434')

# Use a strong model (pull it first with ollama pull llama3.1:70b)
MODEL_NAME = "llama3.1:70b"  # or "qwen2.5:32b" if 70B is too slow

# Safe replacements (fallback)
SAFE_REPLACEMENTS = {
    "strcpy": "strncpy(dest, src, sizeof(dest)-1); dest[sizeof(dest)-1] = '\\0';",
    "gets":   "fgets(input, sizeof(input), stdin);",
    "sprintf": "snprintf(buf, sizeof(buf), fmt, args);"
}

all_repairs = []

for diag_file in DIAG_DIR.glob("*.json"):
    sample_id = diag_file.stem
    diagnosis = json.load(open(diag_file))

    src_file = SRC_DIR / f"{sample_id}.c"
    if not src_file.exists():
        print(f"Source file not found: {src_file}")
        continue

    source = src_file.read_text().splitlines()

    for d in diagnosis:
        ln = d.get("line", 0) - 1
        if ln < 0 or ln >= len(source):
            continue

        original = source[ln].strip()
        feature = d.get("feature", "")
        diag_type = d.get("type", "")

        repaired = original

        # Try local Ollama LLM for smart repair
        try:
            prompt = f"""You are a C/C++ vulnerability repair expert.
            Fix this suspicious line safely while preserving semantics as closely as possible.

            Line: {original}
            Diagnosis type: {diag_type}
            Feature: {feature}

            IMPORTANT: Return ONLY valid JSON, nothing else. No explanations, no extra text.
            Format exactly:
            {{"repairs": ["safe repair 1", "safe repair 2"]}}

            Generate 2 different safe alternatives."""

            response = client.generate(
                model=MODEL_NAME,
                prompt=prompt,
                options={"temperature": 0.7, "num_predict": 300}
            )

            # Ollama response is text — parse JSON
            try:
                start = response['response'].find('{')
                end = response['response'].rfind('}') + 1
                json_str = response['response'][start:end]
                llm_json = json.loads(json_str)
                llm_repairs = llm_json.get("repairs", [])
            except:
                llm_repairs = []

            if llm_repairs:
                repaired = random.choice(llm_repairs)
                print(f"Ollama repair for line {ln+1}: {repaired}")
            else:
                print("Ollama returned no valid JSON → using fallback")

        except Exception as e:
            print(f"Ollama error: {e} → using fallback rule")

        # Fallback to original rule-based repair
        if repaired == original:
            if diag_type == "api_mask" and feature in SAFE_REPLACEMENTS:
                repaired = SAFE_REPLACEMENTS[feature]
            elif diag_type in ["var_rename", "token_remove"]:
                repaired = re.sub(rf"\b{re.escape(feature)}\b", "safe_var", original)
            elif diag_type == "dead_danger":
                repaired = original + "\n// safe check: if (len > 0) strcpy_safe(dest, src);"
            else:
                repaired = "// " + original

        # Basic semantic fix
        if not repaired.strip().endswith(('{', ';', '}')):
            repaired += ";"

        all_repairs.append({
            "sample": sample_id,
            "line": ln + 1,
            "original": original,
            "repaired": repaired,
            "feature": feature,
            "type": diag_type
        })

        time.sleep(0.5)  # small delay

OUT_PATH.write_text(json.dumps(all_repairs, indent=2))
print(f"Generated {len(all_repairs)} counterfactual repairs")
print(f"Saved to: {OUT_PATH}")