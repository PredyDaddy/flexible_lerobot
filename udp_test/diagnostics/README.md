# Record timing probe

These scripts are readonly side-channel diagnostics for the current JZ UDP recording flow.
They do not replace `udp_test/all/start_record.sh` or `record.sh`, and they do not publish robot commands.

Run the x86 receiver first:

```bash
PYTHONPATH=src conda run --no-capture-output -n lerobot_flex python \
  udp_test/diagnostics/x86_record_timing_probe.py \
  --bind-ip 0.0.0.0 \
  --state-port 39110 \
  --action-port 39130 \
  --duration-s 180 \
  --allowed-sender-ip 192.168.1.81 \
  --out tests/outputs/record_timing_probe.jsonl
```

Run the Orin sender in another terminal:

```bash
PYTHONPATH=src conda run --no-capture-output -n lerobot python \
  udp_test/diagnostics/orin_record_timing_probe.py \
  --target-ip 192.168.1.106 \
  --bind-ip 192.168.1.81 \
  --state-port 39110 \
  --action-port 39130
```

Then run the normal recording flow exactly as before:

```bash
bash udp_test/all/start_record.sh
bash record.sh
bash udp_test/all/stop_record.sh
```

The receiver prints:

- receive frequency
- sequence loss and duplicates/reordering
- Orin-side source age and source skew
- x86 receive age relative to Orin probe sample time
- state/action sample skew

Because this uses separate diagnostic ports, it estimates the timing behavior without touching the actual recording
ports `39010` and `39030`.
