"""Tiny paramiko SSH helper for the AgoraPlanner cloud runbook.

Credentials must be supplied at invocation time:
    export AGORA_SSH_PASSWORD=...
    python scripts/cloud/remote.py "command"

Never store the password in this file or in the repository.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import paramiko

HOST = os.environ.get("AGORA_SSH_HOST", "connect.westd.seetacloud.com")
PORT = int(os.environ.get("AGORA_SSH_PORT", "30573"))
USER = os.environ.get("AGORA_SSH_USER", "root")


def connect() -> paramiko.SSHClient:
    password = os.environ.get("AGORA_SSH_PASSWORD")
    if not password:
        raise SystemExit("AGORA_SSH_PASSWORD is not set")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        HOST,
        port=PORT,
        username=USER,
        password=password,
        timeout=30,
        banner_timeout=30,
        auth_timeout=30,
        look_for_keys=False,
        allow_agent=False,
    )
    return client


def run(client: paramiko.SSHClient, command: str, timeout: float = 3600) -> int:
    stdin, stdout, stderr = client.exec_command(command, timeout=timeout, get_pty=False)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    if out:
        sys.stdout.buffer.write(out.encode("utf-8", errors="replace"))
        sys.stdout.buffer.flush()
    if err:
        sys.stderr.buffer.write(err.encode("utf-8", errors="replace"))
        sys.stderr.buffer.flush()
    return code


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?", default="")
    parser.add_argument("--put", nargs=2, metavar=("LOCAL", "REMOTE"))
    parser.add_argument("--write", nargs=2, metavar=("LOCAL", "REMOTE"))
    parser.add_argument("--get", nargs=2, metavar=("REMOTE", "LOCAL"))
    parser.add_argument("--timeout", type=float, default=3600)
    args = parser.parse_args()
    client = connect()
    try:
        if args.put:
            sftp = client.open_sftp()
            sftp.put(args.put[0], args.put[1])
            sftp.close()
            print(f"uploaded {args.put[0]} -> {args.put[1]}")
            return 0
        if args.write:
            import shlex
            data = Path(args.write[0]).read_bytes()
            command = f"cat > {shlex.quote(args.write[1])}"
            stdin, stdout, stderr = client.exec_command(command, timeout=120)
            stdin.write(data)
            stdin.channel.shutdown_write()
            code = stdout.channel.recv_exit_status()
            err = stderr.read().decode("utf-8", errors="replace")
            if code != 0:
                print(err, file=sys.stderr)
                return code
            print(f"wrote {args.write[0]} -> {args.write[1]} ({len(data)} bytes)")
            return 0
        if args.get:
            sftp = client.open_sftp()
            sftp.get(args.get[0], args.get[1])
            sftp.close()
            print(f"downloaded {args.get[0]} -> {args.get[1]}")
            return 0
        return run(client, args.command, timeout=args.timeout)
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
