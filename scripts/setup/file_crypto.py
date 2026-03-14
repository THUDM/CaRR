#!/usr/bin/env python3
import argparse
import base64
import hashlib
import json
from pathlib import Path

FORMAT = "carr-xor-sha256-v1"


def derive_key(password: str, length: int) -> bytes:
    digest = hashlib.sha256(password.encode("utf-8")).digest()
    return digest * (length // len(digest)) + digest[: length % len(digest)]


def xor_bytes(data: bytes, key: bytes) -> bytes:
    return bytes(a ^ b for a, b in zip(data, key))


def encrypt_bytes(data: bytes, password: str) -> str:
    key = derive_key(password, len(data))
    encrypted = xor_bytes(data, key)
    return base64.b64encode(encrypted).decode("utf-8")


def decrypt_bytes(ciphertext_b64: str, password: str) -> bytes:
    encrypted = base64.b64decode(ciphertext_b64)
    key = derive_key(password, len(encrypted))
    return xor_bytes(encrypted, key)


def encrypt_file(input_path: Path, output_path: Path, password: str) -> None:
    raw = input_path.read_bytes()
    payload = {
        "format": FORMAT,
        "filename": input_path.name,
        "size": len(raw),
        "ciphertext_b64": encrypt_bytes(raw, password),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def decrypt_file(input_path: Path, output_path: Path, password: str) -> None:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if payload.get("format") != FORMAT:
        raise ValueError(f"Unsupported format in {input_path}: {payload.get('format')}")
    raw = decrypt_bytes(payload["ciphertext_b64"], password)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(raw)


def cmd_encrypt_dir(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    files = sorted(input_dir.glob(args.pattern))
    if not files:
        raise ValueError(f"No files matched pattern {args.pattern} in {input_dir}")
    for file_path in files:
        output_name = file_path.name + args.suffix
        output_path = output_dir / output_name
        encrypt_file(file_path, output_path, args.password)
        if args.remove_plain:
            file_path.unlink()
        print(f"encrypted: {file_path} -> {output_path}")


def cmd_decrypt_dir(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    files = sorted(input_dir.glob(args.pattern))
    if not files:
        raise ValueError(f"No files matched pattern {args.pattern} in {input_dir}")
    for file_path in files:
        output_name = file_path.name
        if args.strip_suffix and output_name.endswith(args.strip_suffix):
            output_name = output_name[: -len(args.strip_suffix)]
        output_path = output_dir / output_name
        decrypt_file(file_path, output_path, args.password)
        print(f"decrypted: {file_path} -> {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Encrypt/decrypt dataset files.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    encrypt_dir = subparsers.add_parser("encrypt-dir", help="Encrypt files in a directory")
    encrypt_dir.add_argument("--input-dir", required=True)
    encrypt_dir.add_argument("--output-dir", required=True)
    encrypt_dir.add_argument("--pattern", default="*.jsonl")
    encrypt_dir.add_argument("--password", required=True)
    encrypt_dir.add_argument("--suffix", default=".enc")
    encrypt_dir.add_argument("--remove-plain", action="store_true")
    encrypt_dir.set_defaults(func=cmd_encrypt_dir)

    decrypt_dir = subparsers.add_parser("decrypt-dir", help="Decrypt files in a directory")
    decrypt_dir.add_argument("--input-dir", required=True)
    decrypt_dir.add_argument("--output-dir", required=True)
    decrypt_dir.add_argument("--pattern", default="*.enc")
    decrypt_dir.add_argument("--password", required=True)
    decrypt_dir.add_argument("--strip-suffix", default=".enc")
    decrypt_dir.set_defaults(func=cmd_decrypt_dir)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
