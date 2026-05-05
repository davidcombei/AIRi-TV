from codecarbon import EmissionsTracker
import subprocess
import sys
import json
import os

component_name = sys.argv[1]
command = sys.argv[2:]

os.makedirs("Measurements/GPU_CONSUMPTION/RESULTS", exist_ok=True)

tracker = EmissionsTracker(
    project_name=component_name,
    gpu_ids=[4],
    output_dir="Measurements/GPU_CONSUMPTION/RESULTS/",
    log_level="error",
    save_to_file=True
)

tracker.start()
subprocess.run(command)
emissions = tracker.stop()

result = {
    "component": component_name,
    "energy_kwh": round(tracker._total_energy.kWh, 6),
    "duration_s": round(tracker._elapsed_time, 1),
}

out_path = f"Measurements/GPU_CONSUMPTION/RESULTS/{component_name}.json"
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)

print(f"\n[{component_name}]")
print(f"  Energie: {result['energy_kwh']:.6f} kWh")
print(f"  Durata:  {result['duration_s']:.1f}s")
