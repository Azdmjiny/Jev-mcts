#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
source_dir="${1:-$project_root/runtime/llm-mcts-source}"
mkdir -p "$project_root/runtime"

if [ ! -d "$source_dir/vh/data_gene" ]; then
  git clone --recurse-submodules https://github.com/Azdmjiny/llm-mcts.git "$source_dir"
  git -C "$source_dir" checkout 1d5c6a617e577fcb4c494fb38c1bdd517c68e7b6
  git -C "$source_dir" submodule update --init --recursive
fi

if [ ! -e "$project_root/runtime/vh" ]; then
  ln -s "$(cd "$source_dir" && pwd)/vh" "$project_root/runtime/vh"
fi

if [ "$(uname -s)" = Darwin ]; then
  app="$project_root/runtime/macos/macos_exec.v2.3.0.app"
  if [ ! -d "$app" ]; then
    archive="$project_root/runtime/macos_exec.zip"
    curl --fail --location --retry 3 \
      http://virtual-home.org/release/simulator/v2.0/v2.3.0/macos_exec.zip \
      --output "$archive"
    expected=d27e7c43d7471669dc8a253ec92ab6bbc458dd937ed7e2aea658dc79e33d69dd
    actual="$(shasum -a 256 "$archive" | awk '{print $1}')"
    if [ "$actual" != "$expected" ]; then
      echo "Unity archive SHA-256 mismatch" >&2
      exit 1
    fi
    mkdir -p "$project_root/runtime/macos"
    unzip -q "$archive" -d "$project_root/runtime/macos"
    rm "$archive"
  fi
  echo "Unity app: $app"
else
  echo "Use a native Linux x86-64 host for the original Unity Linux binary."
fi
echo "VirtualHome source: $project_root/runtime/vh"
