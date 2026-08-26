import os


# Repository-facing documents are intentionally kept out of the source aggregate.
# `holosphere.txt` is the generated output itself and must never be read back in.
TOP_LEVEL_EXCLUDED_FILES = {
    "CHANGELOG.md",
    "CITATION.cff",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "GOVERNANCE.md",
    "LICENSE-APACHE",
    "LICENSE-MIT",
    "MAINTAINERS.md",
    "README.md",
    "RELEASING.md",
    "SECURITY.md",
    "SUPPORT.md",
    "holosphere.txt",
}


def generate_aggregate():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_path = os.path.join(repo_root, "holosphere.txt")
    
    ignore_dirs = {".git", ".github", "target", ".gemini", ".idea", ".vscode", "__pycache__", "node_modules", "datasets", "benchmark_databases", "scripts", "tests", "benches", "performance-baseline-v1" }
    ignore_files = {"holosphere.txt", "Cargo.lock", "SignalTraceMACROS.md"}
    
    collected_files = []
    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        for f in files:
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, repo_root).replace("\\", "/")
            if f in ignore_files or rel_path in TOP_LEVEL_EXCLUDED_FILES:
                continue
            collected_files.append((rel_path, full_path))
            
    collected_files.sort(key=lambda x: x[0])
    
    with open(output_path, "w", encoding="utf-8") as out:
        for rel_path, full_path in collected_files:
            try:
                with open(full_path, "r", encoding="utf-8") as inf:
                    content = inf.read()
            except Exception:
                continue
            header = f"File: {rel_path}\n" + "=" * (len(rel_path) + 6) + "\n"
            out.write(header)
            out.write(content)
            out.write("\n\n")
            
    print(f"Successfully aggregated {len(collected_files)} files into {output_path}")

if __name__ == "__main__":
    generate_aggregate()
