#!/usr/bin/env bash
# Fetch the Chinook sample database (a real relational business DB: artists, albums, tracks,
# customers, invoices) and prepare a held-out form-fill split from it. For each selected track we
# hold out its genre and unit price; the recipe fills them back from the track's album siblings,
# and gold.jsonl scores the fill. Nothing is committed; everything lands under data/.
#
# Chinook is distributed by lerocha/chinook-database under the MIT license.
#
# Usage: recipes/form_autofill/fetch.sh [--url URL] [--limit N]
#   --url URL   location of Chinook_Sqlite.sqlite (default: the project's GitHub raw copy)
#   --limit N   how many held-out tracks to prepare (default: 800)

source "$(dirname "${BASH_SOURCE[0]}")/../../tools/lib/common.sh"

url="https://github.com/lerocha/chinook-database/raw/master/ChinookDatabase/DataSources/Chinook_Sqlite.sqlite"
limit=800
while [[ $# -gt 0 ]]; do
    case "$1" in
        --url) url="${2:?--url needs a value}"; shift 2 ;;
        --limit) limit="${2:?--limit needs a value}"; shift 2 ;;
        -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown argument '$1' (see --help)" ;;
    esac
done

[[ "$limit" =~ ^[1-9][0-9]*$ ]] || die "--limit must be a positive integer, got '${limit}'"

data_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/data"
mkdir -p "$data_dir"
python="$(project_python)"
assert_python_new_enough "$python"

db="${data_dir}/chinook.sqlite"
if [[ ! -f "$db" ]]; then
    require_command curl "Install curl to download the dataset."
    note "downloading Chinook from ${url}"
    curl -fL --output "$db" "$url" || die "download failed; check --url or your connection."
fi

note "preparing held-out form-fill split (up to ${limit} tracks)"
"$python" - "$db" "$data_dir" "$limit" <<'PY' || die "preparing the dataset failed (see above)"
import json, sqlite3, sys
from pathlib import Path

db, data_dir, limit = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
conn = sqlite3.connect(db)
conn.row_factory = sqlite3.Row

# Only tracks on albums with a sibling that also has a genre -- otherwise the retrieval context is
# empty and the fill would be an unguided guess. Deterministic order for reproducibility.
rows = conn.execute("""
    SELECT t.TrackId, t.Name AS track, t.AlbumId, t.Composer, t.Milliseconds,
           g.Name AS genre, t.UnitPrice AS unit_price, al.Title AS album, ar.Name AS artist
    FROM Track t
    JOIN Genre g ON g.GenreId = t.GenreId
    JOIN Album al ON al.AlbumId = t.AlbumId
    JOIN Artist ar ON ar.ArtistId = al.ArtistId
    WHERE t.AlbumId IN (
        SELECT AlbumId FROM Track WHERE GenreId IS NOT NULL
        GROUP BY AlbumId HAVING COUNT(*) > 1)
    ORDER BY t.TrackId
    LIMIT ?
""", (limit,)).fetchall()
if not rows:
    raise SystemExit("no eligible tracks found; is this a Chinook database?")

with (data_dir / "heldout.jsonl").open("w", encoding="utf-8") as heldout, \
     (data_dir / "gold.jsonl").open("w", encoding="utf-8") as gold:
    for r in rows:
        rid = f"track-{r['TrackId']}"
        source = (f"Track {r['track']!r} from the album {r['album']!r} by {r['artist']}"
                  + (f", composed by {r['Composer']}" if r['Composer'] else "") + ".")
        meta = {"album_id": r["AlbumId"], "track_id": r["TrackId"], "track_name": r["track"],
                "album_title": r["album"], "artist_name": r["artist"],
                "composer": r["Composer"], "milliseconds": r["Milliseconds"]}
        heldout.write(json.dumps({"record_id": rid, "source": source, "meta": meta},
                                 ensure_ascii=False) + "\n")
        gold.write(json.dumps({"record_id": rid, "genre": r["genre"],
                               "unit_price": r["unit_price"]}, ensure_ascii=False) + "\n")
conn.close()
print(f">> {len(rows)} held-out tracks written with gold genre + unit price")
PY

note "done. Fill data/heldout.jsonl with the recipe; score against data/gold.jsonl via recipes/form_autofill/eval.py."
