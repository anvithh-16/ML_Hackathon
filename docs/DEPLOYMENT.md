# Two-laptop deployment runbook  (owner: Member 4)

## Laptop 1 - API
```bash
cd pathsnap && source .venv/bin/activate
# evaluator trips: copy the folder you are given so that data/eval/<trip_id>/<trip_id>.track ... exists
ls data/eval | head
python -m src.export.export_artifacts --model <model> --variant <variant> --reason "<why>"   # once
ipconfig getifaddr en0                      # LAN IP, e.g. 192.168.1.23
uvicorn api.main:app --host 0.0.0.0 --port 8000 2>&1 | tee logs/api_console.log
```
Check: `curl http://localhost:8000/health` -> `"mode": "model"`.

## Laptop 2 - UI
```bash
cd pathsnap && python3 -m venv .venv-ui && source .venv-ui/bin/activate
pip install -r ui/requirements-ui.txt
curl http://<LAPTOP1_IP>:8000/health
PATHSNAP_API_URL=http://<LAPTOP1_IP>:8000 streamlit run ui/app.py
```

## If Laptop 2 cannot reach Laptop 1
1. Same Wi-Fi network on both? Campus Wi-Fi often isolates clients -> use a phone hotspot for both laptops.
2. Laptop 1: System Settings -> Network -> Firewall -> Options -> allow incoming connections for Python (or turn the firewall off for the demo).
3. API started with `--host 0.0.0.0` (not 127.0.0.1)?
4. IP changed? Re-run `ipconfig getifaddr en0` and edit the URL in the UI sidebar.
5. Port busy: `lsof -i :8000` then `kill <pid>`.

## After the demo
`cp logs/api_console.log results/logs/api_demo.log` and commit it (report §7 screenshot source).
