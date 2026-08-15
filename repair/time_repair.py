import json, time, statistics
from ollama import Client
client = Client(host="http://localhost:11434")
MODEL = "llama3.1:70b"
samples = [json.loads(l) for l in open("results.jsonl") if l.strip()][:15]
lat = []
for i, s in enumerate(samples, 1):
    code = s.get("vulnerable_func", "")[:4000]
    prompt = f"Provide a security fix for the following C function:\n{code}"
    t = time.time()
    client.chat(model=MODEL, messages=[{"role":"user","content":prompt}],
                options={"temperature":0.2, "num_predict":2048})
    dt = time.time() - t
    lat.append(dt)
    print(f"{i}/15: {dt:.1f}s", flush=True)
print(f"\nmean={statistics.mean(lat):.1f}s median={statistics.median(lat):.1f}s "
      f"min={min(lat):.1f}s max={max(lat):.1f}s n={len(lat)}", flush=True)
