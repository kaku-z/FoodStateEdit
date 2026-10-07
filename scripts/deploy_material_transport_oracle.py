"""Package or execute the CPU oracle through a working SSH alias.

No dependency installation, GPU reservation, existing-run overwrite or process
termination. Authentication and runtime preflight happen before remote writes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import uuid
import zipfile


def run(args, timeout=60):
    return subprocess.run(args, check=True, text=True, timeout=timeout,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="gp40")
    parser.add_argument("--local-tunnel-port", type=int,
                        help="Use an existing localhost SSH tunnel; preserves the host's key alias")
    parser.add_argument("--remote-root", default="/host/space0/guo-z/tf-ufi")
    parser.add_argument("--python", default="/host/space0/guo-z/envs/geoedit/bin/python")
    parser.add_argument("--bundle-dir", required=True, type=Path)
    parser.add_argument("--package-only", action="store_true")
    parser.add_argument("--cases", type=int, default=16)
    parser.add_argument("--experiment", choices=("oracle", "sensitivity", "identifiability"), default="oracle")
    parser.add_argument("--resolution", type=int, default=160)
    args = parser.parse_args()
    if args.host.startswith("-") or any(c.isspace() for c in args.host):
        parser.error("host must be a single SSH alias")
    if args.cases < 1:
        parser.error("cases must be positive")
    if args.local_tunnel_port is not None and not 1 <= args.local_tunnel_port <= 65535:
        parser.error("tunnel port must be in 1..65535")
    root = Path(__file__).resolve().parents[1]
    files = ["foodstateedit/__init__.py", "foodstateedit/acceptance.py", "foodstateedit/engine.py",
             "foodstateedit/model.py", "foodstateedit/material_transfer/__init__.py",
             "foodstateedit/material_transfer/core.py", "scripts/run_material_transport_oracle.py",
             "tests/test_material_transfer_oracle.py", "docs/material_transport_oracle.md"]
    if args.experiment in ("sensitivity", "identifiability"):
        files += ["foodstateedit/material_transfer/sensitivity.py", "scripts/run_material_sensitivity.py",
                  "tests/test_material_sensitivity.py"]
    if args.experiment == "identifiability":
        files += ["foodstateedit/material_transfer/identifiability.py", "scripts/run_material_identifiability.py",
                  "tests/test_material_identifiability.py"]
    args.bundle_dir.mkdir(parents=True, exist_ok=False)
    archive = args.bundle_dir / "material_transport_oracle.zip"
    manifest = {}
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for filename in files:
            data = (root / filename).read_bytes()
            bundle.writestr(filename, data)
            manifest[filename] = hashlib.sha256(data).hexdigest()
        bundle.writestr("manifest.json", json.dumps(manifest, indent=2))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (args.bundle_dir / "bundle_sha256.txt").write_text(digest + "\n", encoding="ascii")
    print(f"Package: {archive.resolve()} sha256={digest}", flush=True)
    if args.package_only:
        return
    connection = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
    if args.local_tunnel_port:
        connection += ["-o", "ProxyJump=none", "-o", "HostName=127.0.0.1",
                       "-o", f"Port={args.local_tunnel_port}", "-o", f"HostKeyAlias={args.host}"]
    ssh = ["ssh", "-n", *connection, args.host]
    preflight = "import socket,numpy,PIL;print(socket.gethostname());print(numpy.__version__);print(PIL.__version__)"
    print(run(ssh + [f"{shlex.quote(args.python)} -c {shlex.quote(preflight)}"]), flush=True)
    remote = args.remote_root.rstrip("/") + "/material_" + args.experiment + "_" + uuid.uuid4().hex[:12]
    run(ssh + ["mkdir " + shlex.quote(remote)])
    run(["scp", *connection, str(archive),
         args.host + ":" + remote + "/bundle.zip"])
    verify = ("import hashlib,json,pathlib,zipfile;"
              f"p=pathlib.Path({remote!r});a=p/'bundle.zip';"
              f"assert hashlib.sha256(a.read_bytes()).hexdigest()=={digest!r};"
              "zipfile.ZipFile(a).extractall(p);m=json.loads((p/'manifest.json').read_text());"
              "assert all(hashlib.sha256((p/k).read_bytes()).hexdigest()==v for k,v in m.items())")
    run(ssh + [f"{shlex.quote(args.python)} -c {shlex.quote(verify)}"])
    script = {"oracle": "scripts/run_material_transport_oracle.py", "sensitivity": "scripts/run_material_sensitivity.py",
              "identifiability": "scripts/run_material_identifiability.py"}[args.experiment]
    commands = (f"cd {shlex.quote(remote)} && export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 && "
                f"{shlex.quote(args.python)} -m unittest discover -s tests -p 'test_material_*.py' > tests.log 2>&1 && "
                f"{shlex.quote(args.python)} {script} --output results --cases {args.cases} "
                f"--resolution {args.resolution} > experiment.log 2>&1 && cat experiment.log")
    # Synchronous pilot: SSH disconnect/timeout is an incomplete run, never success.
    (args.bundle_dir / "remote_run.json").write_text(json.dumps(
        {"host": args.host, "remote": remote, "command": commands, "status": "prepared"}, indent=2))
    print(f"Remote run: {remote}", flush=True)
    print(run(ssh + [commands], timeout=1800), flush=True)
    compress = ("import shutil,pathlib;" + f"p=pathlib.Path({remote!r});" +
                "shutil.copy(p/'tests.log',p/'results'/'tests.log');"
                "shutil.copy(p/'experiment.log',p/'results'/'experiment.log');"
                "shutil.make_archive(str(p/'results'),'zip',p/'results')")
    run(ssh + [f"{shlex.quote(args.python)} -c {shlex.quote(compress)}"])
    run(["scp", *connection, args.host + ":" + remote + "/results.zip",
         str(args.bundle_dir / "server_results.zip")])
    with zipfile.ZipFile(args.bundle_dir / "server_results.zip") as result_zip:
        result_zip.extractall(args.bundle_dir / "server_results")
    (args.bundle_dir / "remote_run.json").write_text(json.dumps(
        {"host": args.host, "remote": remote, "command": commands, "status": "completed"}, indent=2))
    print("Server results retrieved", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(str(exc), flush=True)
        if getattr(exc, "stderr", None):
            print(exc.stderr, flush=True)
        raise SystemExit(1)
