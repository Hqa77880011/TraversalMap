"""Extract downloaded Ubuntu development/runtime files into a private prefix.

No package installation or global filesystem changes. Windows extraction avoids
NTFS chmod/chown overhead; zstandard is needed for Ubuntu's data.tar.zst payloads.
"""

import argparse
import io
import tarfile
from pathlib import Path


def extract(directory, prefix):
    prefix = Path(prefix).resolve()
    links, count = [], 0
    for deb in sorted(Path(directory).glob("*.deb")):
        with deb.open("rb") as stream:
            if stream.read(8) != b"!<arch>\n":
                raise ValueError("Invalid deb archive: "+str(deb))
            while header := stream.read(60):
                size = int(header[48:58])
                name = header[:16].decode().strip().rstrip("/")
                payload = stream.read(size)
                if size % 2:
                    stream.read(1)
                if not name.startswith("data.tar"):
                    continue
                if name.endswith(".zst"):
                    import zstandard
                    with zstandard.ZstdDecompressor().stream_reader(io.BytesIO(payload)) as reader:
                        payload = reader.read()
                with tarfile.open(fileobj=io.BytesIO(payload), mode="r:*") as archive:
                    for member in archive:
                        relative = member.name.removeprefix("./")
                        if not relative.startswith(("usr/include/", "usr/lib/", "usr/bin/", "usr/share/cmake", "usr/share/eigen3")):
                            continue
                        target = (prefix / relative).resolve()
                        if not target.is_relative_to(prefix):
                            raise ValueError("Archive member escapes private prefix")
                        if member.isfile():
                            target.parent.mkdir(parents=True, exist_ok=True)
                            target.write_bytes(archive.extractfile(member).read())
                            count += 1
                        elif member.issym():
                            links.append((target, member.linkname))
    for _ in range(4):
        remaining = []
        for target, source in links:
            source = (prefix / source.lstrip("/") if source.startswith("/") else target.parent / source).resolve()
            if source.is_relative_to(prefix) and source.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
            else:
                remaining.append((target, str(source)))
        links = remaining
    print(f"Extracted {count} regular files; unresolved optional links: {len(links)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debs", required=True)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()
    extract(args.debs, args.prefix)
