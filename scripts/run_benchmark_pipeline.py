import subprocess

steps = [
    ["python", "scripts/benchmark_mytorch_vs_pytorch.py"],
    ["python", "scripts/generate_benchmark_report.py"],
    ["python", "scripts/md_to_pdf.py"],
]

for cmd in steps:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

print("Benchmark pipeline completed.")
