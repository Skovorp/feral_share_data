# feral_share_data

Shared FERAL data dumps (CalMS21 raw outputs, preset/backbone comparisons, example videos).

## ⚠️ Large files are stored as split parts

GitHub rejects files over 100MB and warns above 50MB. A few files here are larger
than that, so each one is committed as a set of **45MB chunks** instead of the whole
file:

```
FINAL_DATA/_preset/max/calms/raw_test.json          ->  raw_test.json.part_aa, .part_ab, ... .part_ae
FINAL_DATA/_preset/default/calms/raw_test.json      ->  raw_test.json.part_aa, .part_ab
FINAL_DATA/_preset/lite/calms/raw_test.json         ->  raw_test.json.part_aa, .part_ab
FINAL_DATA/_backbone/vjepa2.1/calms/raw_test.json   ->  raw_test.json.part_aa, .part_ab
FINAL_DATA/_backbone/vjepa2/calms/raw_test.json     ->  raw_test.json.part_aa, .part_ab
FINAL_DATA/_backbone/videoprism/calms/raw_test.json ->  raw_test.json.part_aa, .part_ab
FINAL_DATA/raw_output_calms.json                    ->  raw_output_calms.json.part_aa, .part_ab
example_videos/SUP_1_calms_2min.mp4                 ->  SUP_1_calms_2min.mp4.part_aa, .part_ab
```

The original (un-split) files are listed in `.gitignore`, so once you rebuild them
`git status` stays clean and you can't accidentally re-commit the oversized versions.

## Merge the parts back after cloning

Run the reassemble script once:

```bash
./reassemble.sh
```

It concatenates every `<file>.part_*` set back into the original file and verifies
nothing is left dangling. Safe to run more than once.

### Doing it by hand

Splitting and joining is plain `split` / `cat` — no special tooling. To rebuild a
single file manually:

```bash
cat FINAL_DATA/_preset/max/calms/raw_test.json.part_* > FINAL_DATA/_preset/max/calms/raw_test.json
```

The `part_*` glob expands in alphabetical order (`part_aa`, `part_ab`, ...), which is
the correct concatenation order, so the rebuilt file is byte-for-byte identical to the
original.

## Updating a large file (maintainers)

If you change one of these files (or add a new file over 50MB), re-split before
committing:

```bash
./split_large.sh          # re-chunks any file > 50MB into <file>.part_*
git add -A                # stage the new parts (originals are git-ignored)
git commit -m "update data"
```

If you add a **new** large file, also add its path to `.gitignore` so the full file
isn't tracked.
