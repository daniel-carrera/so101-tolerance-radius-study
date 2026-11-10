# Training: from two recorded datasets to two policies at the rig

Two ACT policies, one per dataset, with identical settings (`protocol/protocol.md`, section 5). This page is the procedure:
each step gives the command and what a good result looks like.

**Status: prepared on 2026-10-05; steps 0 to 3 were run on 2026-10-07 as printed here, and both runs started
(journal of that day). Step 5 was run at the first checkpoints of the runs that held (17:17, both comparisons as printed, both `identical`) and again on `080000` (21:45). Step 7 was run as printed when both runs had ended (21:45: the download, and the load check under both LeRobots, 234 tensors and 51,597,190 parameters each time). Step 8 and the resume of step 6 have not been run.** Run that day exactly as printed:
0.1, 0.3, 0.4 and 0.5; step 2's `repo create`, `upload` and `repo tag create` (the tool says that `hf repo` gives way
to `hf repos`, and works); step 3.3's `colab new`, the three `colab upload` and `colab exec`; step 3.4's `grep`,
which gave 1. The account held two A100 machines at once. Each command carries a mark that says how far it was
checked before that day:

- **[ran]** run on this machine on 2026-10-05, against August's dataset or August's checkpoint;
- **[read]** read in the tool's source and not run, because it needs the network, an account or a GPU;
- **[doc]** stated by the earlier project's own records (`../lerobot-101`), and not checked further.

Nothing on this page touches the arm, a camera or a serial port.

| File | What it is |
|---|---|
| `README.md` | this procedure |
| `train_act_colab.ipynb` | the notebook that checks a Colab machine and starts one training run; one file for both runs |
| `settings.md` | every setting of the August run, the four that change this week, and where each value was read |
| `compare_train_configs.py` | the check behind "identical settings": compares two saved training configurations |
| `test_compare_train_configs.py` | its tests |
| `test_train_act_colab.py` | the notebook's logic, cell by cell, against stand-ins for Colab, the Hub, the dataset and the trainer |
| `local/` | made by step 3 and ignored by git (`.gitignore` here): parameter files and launch outputs, which carry the Hub namespace |

## Names used on this page

Type these once in the shell that runs the commands. Commands are run from this repository's folder.

```bash
LOCAL="${TOLRAD_NAMESPACE:?set it to the name the recorder was given}"   # the folder the recorder wrote under
export NAMESPACE="$LOCAL"                            # owner of the repositories on the Hub: see below if it differs
RIG_PY="$HOME/miniconda3/envs/lerobot/bin/python"   # the rig's interpreter (RIG_PY in the Makefile)
BIN="$(dirname "$RIG_PY")"                           # where its `hf`, `colab` and `wandb` commands are
TOOLS=../lerobot-101                                 # the earlier project: read, never changed
CACHE="$HOME/.cache/huggingface/lerobot"             # where the recorder puts datasets
```

`NAMESPACE` is the Hub account or organisation that owns the datasets and the policies. `LOCAL` is the name the
recorder was given (`--namespace`, or `TOLRAD_NAMESPACE`): the folder under `$CACHE` that holds the recorded sets. The
first line stops with a message if that variable is not set. **The two need not be the same name** (on 2026-10-07 the
recorder's name was no account on the Hub): when step 0.3 shows neither a user nor an organisation of that name, type
`export NAMESPACE=<the user it shows>` after these lines, and nothing else changes. A recorded set's files do not carry
the folder's name (*checked* 2026-10-07: no file under `meta/` or `data/` holds it), so a folder is uploaded to any
owner as it is. Prefer the user's own account to an organisation: a private repository of an organisation can be read
by its members. The names are typed into the shell and written into no file of this repository: the review is double
blind. The same goes for the W&B account.

| Thing | Name |
|---|---|
| Datasets, on disk and on the Hub | `$NAMESPACE/tolrad-clustered-v1`, `$NAMESPACE/tolrad-spread-v1` |
| Policy repositories, job names, W&B run names | `act-tolrad-clustered-v1`, `act-tolrad-spread-v1` |
| Colab sessions | `tolrad-clustered`, `tolrad-spread` |
| Checkpoint evaluated | step 80,000: `checkpoints/080000/` in each policy repository |

Terms are explained at the end of the page.

## The whole thing at a glance

| Step | What | Where | How long |
|---|---|---|---|
| 0 | Accounts, balance, tests, a rehearsal | this machine | 15 min; do it before the recording day ends |
| 1 | Check each recorded dataset | this machine, no network | a few minutes each |
| 2 | Upload each dataset as a private repository and tag it | this machine | about 0.42 GB each; depends on the uplink |
| 3 | Start one run per dataset | two Colab machines | 10 to 15 min each (*estimate*) |
| 4 | Watch | W&B and the Hub | about 5 h 10 min per run |
| 5 | Check that the two runs have the same settings | this machine | 2 min, after the first checkpoint (about 40 min in) |
| 6 | If a machine is lost: resume, or fall back | a new Colab machine | |
| 7 | Bring the two step-80,000 checkpoints to the rig | this machine | about 0.2 GB each |
| 8 | Release the machines, write the record | | |

A run starts as soon as its own dataset has passed steps 1 and 2. The two runs need not start together.

## 0. Before the recording day ends: accounts, balance, tests

August lost time to a login that had expired (`../lerobot-101/docs/run-journal.md`, line 48). Find out early.

**0.1 The Colab command still has its login.** **[read]**

```bash
"$BIN/colab" whoami
```

- **Good:** five kinds of line: `Auth provider:`, `Email:` with the Colab account, `Audience:`, `Expires in:` with a
  number of minutes, and a `Scopes:` list. The tool's own notes say the keep-alive needs the scope that ends in
  `colaboratory`.
- **`[colab] whoami: failed to refresh credentials`, `tokeninfo returned HTTP 400`, or a browser sign-in opens:** the
  login has expired. The person signs in again.
- **Do not use `colab sessions` as this test.** With no session of its own and a failed login it prints the same
  `No active sessions found on server.` as with a working one (`colab_cli/common.py`, lines 87 to 97).
- The command may also say that a newer version of itself exists. **Do not update it this week:** what this page says
  about it was read in the installed version, 0.6.0.

**0.2 Enough compute units, and two A100 machines at once.** Two runs need about 55 units (see "Time and cost"); have
at least 80. `"$BIN/colab" pay` opens the page with the balance **[read]**. Whether the account may hold two A100
machines at the same time is not known here: step 3 finds out, and the answer changes the night's plan (two runs in
parallel take 5 hours, one after the other 10).

**0.3 The Hub login is the account that owns `NAMESPACE`.** **[read]**

```bash
"$BIN/hf" auth whoami
```

- **Good:** it reports `Logged in`, and `user` or `orgs` contains the value of `$NAMESPACE`.
- **Another account** (August used two): `"$BIN/hf" auth switch`, then look again.

**0.4 The two secret files are there.** They are named by the earlier project's README and are never opened or printed:

```bash
ls -l "$TOOLS/secrets/huggingface.env" "$TOOLS/secrets/wandb.venv"
```

- **Good:** two files, neither empty. The first holds a line `HF_TOKEN=...`, the second `WANDB_API_KEY=...` **[doc]**.
  Step 3 uploads them as they are; the notebook reads either that form or a bare value.

**0.5 The tests of this folder pass.** **[ran]**

```bash
"$RIG_PY" -m unittest discover -s training
```

- **Good:** `Ran 95 tests` and `OK`, in about 8 seconds: 34 for the comparison script, 61 for the notebook. Run it
  again after any edit to the notebook: these tests are the only check of its gates short of a Colab machine.

**0.6 A rehearsal, if there is half an hour (recommended).** The practice block that the protocol records and discards
is a free test of steps 1 to 3, and of one pairing that has never run on Colab: a dataset written by the LeRobot the
recorder runs on (the checkout at `6adf515`, release 0.6.1 and 24 commits) and read by Colab's (0.6.1). On this
computer the 27 episodes recorded on 2026-10-06 were read with the source of release 0.6.1, and passed **[ran]**
(the dry run of that night). Upload the practice dataset as in step 2 under its own name, then run
step 3 on one A100 machine with a parameter file that names it and asks for a short run. `EXPECTED_EPISODES` is the
number of episodes the practice dataset holds:

```json
{"NAMESPACE": "...", "DATASET_NAME": "<practice name>", "RUN_NAME": "act-<practice name>",
 "STEPS": 200, "SAVE_FREQ": 200, "EXPECTED_EPISODES": 10}
```

This trains 200 steps (under a minute of training), pushes one checkpoint and the final policy to the Hub, and so
proves the token, the tag, the dataset, the trainer, W&B and the checkpoint upload before the real datasets exist.
Cost: about one unit. Stop the machine afterwards (step 8); the two practice repositories can then be removed on the
Hub's site. With `"MODE": "check"` in place of the two step values, every check runs and nothing is trained.

The rehearsal can also cover the one path of this page that has never been run, the resume of step 6.1: ask for
`"STEPS": 3000, "SAVE_FREQ": 500` instead (about 11 minutes if left alone), stop the machine as soon as the Hub lists
`checkpoints/000500/`, and follow step 6.1 with a parameter file that holds the practice names, the same
`EXPECTED_EPISODES` and `"MODE": "resume_hub"`. Good is the `resumed:` line and a later checkpoint arriving. That costs
two or three units more.

## 1. Check each recorded dataset

On this machine, with no network. Do it for `clustered` and for `spread`.

```bash
LAYOUT=clustered                      # then: spread
DS="tolrad-$LAYOUT-v1"
ROOT="$CACHE/$LOCAL/$DS"
ls "$ROOT"
```

- **Good:** `data  meta  videos` and nothing else. An `images` folder means the recording was not closed properly: stop.

**If step 7 of the recording runbook (`docs/runbook/02-arm-and-recording.md`) was run on the finished set, 1.1 to
1.2 below are done:** they are the same four commands. Take the frame count from that run.

**1.1 Counts, both camera streams, joint names, episode lengths.** **[ran]** on August's 55-episode set: 7 seconds.

```bash
HF_HUB_OFFLINE=1 "$RIG_PY" "$TOOLS/scripts/data/verify_dataset.py" \
    --repo-id "$NAMESPACE/$DS" --root "$ROOT" --expect-episodes 55
echo "exit status: $?"
```

- **Good:** four numbered sections with only `ok` lines, then `DATASET OK` and the folder (the tool's own advice
  follows); exit status **0**. The
  first line reads `55 episodes, N frames, 30 fps`: write `N` down, step 3 can compare it with the Hub's copy.
- **Exit status 1:** `DATASET NOT OK`, with a `FAIL` line for each fault; also when the folder is incomplete or will
  not load. A line starting `warn` (a few late frames, under 1 %) does not fail.
- **Always give `--repo-id` and `--root`.** Without them the script looks for a dataset whose name starts with `so101-`
  under one fixed account, finds August's, and reports on that (`verify_dataset.py:48`) **[read]**.
- **No `--expect-seconds` this week.** Until 2026-10-06 this command carried `--expect-seconds 25`: every episode
  lasted 25 s, and an episode more than 2 frames off 750 failed. Run on August's set, that check fails 16 episodes
  of 697 to 703 frames **[ran]**: the episodes recorded at about 28 Hz and stamped 30 (journal, line 21), in which
  the script's own pacing section saw nothing wrong ("no late frames"), because the frames carry nominal times.
  Since 2026-10-06 a take lasts up to 35 s and ends on the operator's key, so the episodes are not of one length and
  the frame count cannot show a slow loop. **The rate is now read from the sessions' logs:**

  ```bash
  scripts/rig/recording_rate.py "$LAYOUT"
  ```

  **Good:** one line per episode, the cycles equal to the frames of the dataset, and `RATE HELD` (no saved take
  more than 2 % under 30 Hz). An episode that fell short is re-recorded (protocol, section 4: "a check fails").
  **[ran]** on sitting 1's first three episodes (29.99 Hz each) and on August's log, where it names the 16 slow
  episodes.
- `HF_HUB_OFFLINE=1` keeps the check on the local copy. Without it, and once the dataset is on the Hub, a damaged
  local folder could make LeRobot fetch the Hub's copy and check that instead.

**1.2 Torn frames.** A camera transfer that lost part of a picture: valid video, damaged pixels in a band of rows.

```bash
scripts/rig/torn_frames.py "$LAYOUT" --namespace "$LOCAL" --save-frames "data/logs/torn-$LAYOUT"
echo "exit status: $?"
```

- **Good:** `NO TORN FRAME`, exit status **0**. It looks at every frame of both cameras (about four minutes for 27
  episodes, eight for a set of 55, *estimated* from that).
- **Exit status 4:** it names each torn frame by episode, camera, frame and time, and writes it with the frame
  before and after as one picture into `data/logs/torn-<layout>/` (ignored by git): look at them before deciding.
- **Not the earlier project's `check_video_artifacts.py`.** On this rig's white floor that tool called 93 whole
  frames of the first 27 episodes corrupt (*looked at*, all 93: an edge in motion); its test divides by a row
  gradient that a white floor makes small. This project's scan was held against every frame of three sets
  (*measured* 2026-10-07): in August's it finds 278 torn frames, all looked at and all torn, where the earlier tool
  names 171, of which 3 are whole; it calls none of 151 whole frames; the first 27 episodes of the clustered set
  hold none, and the four episodes put aside hold four. **What it does not see:** a band only a little too bright
  or too dark (8 levels or less: nine such frames in August), a band thinner than 8 rows, damage under 1000
  pixels or where the picture is moving, rows that repeat the frame just before. No real torn frame of the wrist
  camera exists in any set: that side rests on tears put in by hand.
- **The pass line is not written anywhere.** August found torn frames and kept the dataset. This week's rule
  (re-record the episode, or accept below some rate, the same for both sets) has to be decided before the last
  sitting, while re-recording is still possible.

**1.3 Not covered here:** the start-state audit, the protocol's fifth check (every episode in its planned cell and
within tolerance of its target). It has its own tool.

Write into `docs/journal.md`, for each set: episodes, frames, both exit statuses, the torn frames found, re-records.

## 2. Upload each dataset: a private repository, and a version tag

Still with `LAYOUT`, `DS` and `ROOT` set as in step 1. These commands were **[read]** in the source of the installed
`hf` (huggingface_hub 1.26.0); none was run.

```bash
"$BIN/hf" repo create "$NAMESPACE/$DS" --repo-type dataset --private
"$BIN/hf" upload "$NAMESPACE/$DS" "$ROOT" . --repo-type dataset --private --exclude "images/*"
```

- **`--private` on both lines.** If the first line fails and the second is run all the same, the upload creates the
  repository itself, and without the option it would be public (*read* in `hf upload --help`, huggingface_hub
  1.26.0: "Whether to create a private repo if repo doesn't exist"). The review is double blind.
- **Good:** the first reports `Repo created` with the repository's address; the second, after the transfer, `Uploaded`
  with the address of one commit. The 27 episodes recorded on 2026-10-06 are 206 MB (*measured* with `du`), so
  expect about 0.42 GB for a set of 55 (August's 55-episode set was 686 MB: its episodes were longer in the
  file). *Measured* 2026-10-07 from the Hub's own times, repository created to upload committed: 64 seconds for
  the clustered set (15 files, 393 MB as the tool counted) and 39 for the spread set (11 files). The rate in the
  tool's last line of progress (501 kB/s) is not its average.
- **`repo create` says the repository exists:** stop. No repository of this name should exist yet; find out what it is
  before uploading into it.

**The tag.** LeRobot loads a dataset from the Hub only if the repository carries a tag with the dataset's format
version. LeRobot's own upload creates it; a plain upload does not, and loading then fails with "Your dataset must be
tagged with a codebase version" (`datasets/utils.py:432` at `6adf515`, `:371` in 0.6.0 **[read]**; August's incident,
journal line 49). The tag is the version written in the dataset itself:

```bash
"$RIG_PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["codebase_version"])' "$ROOT/meta/info.json"
"$BIN/hf" repo tag create "$NAMESPACE/$DS" v3.0 --repo-type dataset
```

- **Good:** the first prints `v3.0` (**[ran]** on August's set; if it prints anything else, use that as the tag and say
  so in the journal). The second reports `Tag created`. Run a second time it fails with "already exists", which is
  harmless.

**Look at what the Hub holds:**

```bash
"$BIN/hf" repo tag list "$NAMESPACE/$DS" --repo-type dataset
"$BIN/hf" datasets ls "$NAMESPACE/$DS" --tree
"$BIN/hf" datasets info "$NAMESPACE/$DS"
find "$ROOT" -type f | wc -l
```

- **Good:** the tag list shows `v3.0`; the tree shows `meta/info.json`, `meta/stats.json`, `meta/tasks.parquet`, the
  episode and data files and the videos of both cameras, as many files as `find` counts locally (August's set: 10)
  plus `.gitattributes`; `info` shows the repository as private.

**This does not yet prove that the dataset loads.** Loading it on this machine would read the local folder, not the
Hub. The proof is made on a clean machine: cell 8 of the notebook, on the Colab machine, loads the dataset from the
Hub, decodes its first and last frame, and refuses to go on unless it finds 55 episodes, the cameras `front` and
`wrist` at 640 x 480, 30 frames a second, six joints and one task. That cell runs in step 3 before anything is trained.
To run the checks and start nothing, put `"MODE": "check"` in the parameter file.

## 3. Start the two runs

One Colab machine per run. Everything a run needs to differ in comes from one small parameter file; everything else
comes from the notebook, which is the same file for both.

### 3.1 What must differ between the two runs, and what must not

| Must differ | Set by |
|---|---|
| the dataset repository (`--dataset.repo_id`) | `DATASET_NAME` |
| the policy repository (`--policy.repo_id`) | `RUN_NAME` |
| the output directory on the machine (`--output_dir`) | follows `RUN_NAME` |
| the job name (`--job_name`), which is also the W&B run's name | `RUN_NAME` |
| the W&B run | W&B gives each run its own id |
| the Colab session and its machine | `-s tolrad-<layout>` |

| Must not differ | How it is held, and how it is checked |
|---|---|
| every other flag of the command (15 of them) | built by the notebook from constants; checked in step 5 |
| the seed | not passed: LeRobot's default, 1000, for both; checked in step 5 |
| LeRobot's version | pinned to 0.6.1 in the notebook, which refuses any other |
| the GPU type | the notebook refuses a machine without an A100 |
| the other packages (PyTorch and the rest) | each run prints one `ENV` line; the two lines must be equal (3.4) |
| the notebook itself | one file; note its commit in the journal: `git log -1 --format=%h -- training/train_act_colab.ipynb` |
| the Hub token and the W&B key | the same two files are uploaded to both machines |

The command the notebook builds for the clustered run (cell 10 prints it; this copy is generated from the notebook):

```bash
lerobot-train \
    --dataset.repo_id=$NAMESPACE/tolrad-clustered-v1 \
    --dataset.eval_split=0.0 \
    --policy.type=act \
    --policy.device=cuda \
    --policy.repo_id=$NAMESPACE/act-tolrad-clustered-v1 \
    --policy.private=true \
    --output_dir=/content/train/act-tolrad-clustered-v1 \
    --job_name=act-tolrad-clustered-v1 \
    --steps=80000 \
    --batch_size=32 \
    --num_workers=8 \
    --save_freq=10000 \
    --save_checkpoint_to_hub=true \
    --eval_steps=0 \
    --max_eval_samples=512 \
    --log_freq=200 \
    --wandb.enable=true \
    --wandb.project=tolrad-act
```

It is August's command with four values changed (`settings.md`). August also passed `--wandb.entity`; here the
uploaded key's own account is used unless `WANDB_ENTITY` is set.

### 3.2 The parameter files

```bash
mkdir -p training/local
for LAYOUT in clustered spread; do
    printf '{"NAMESPACE": "%s", "DATASET_NAME": "tolrad-%s-v1", "RUN_NAME": "act-tolrad-%s-v1"}\n' \
        "$NAMESPACE" "$LAYOUT" "$LAYOUT" > "training/local/$LAYOUT.params.json"
done
diff training/local/clustered.params.json training/local/spread.params.json
```

- **Good:** `diff` shows one line on each side, and the two differ only in the word `clustered` or `spread`, twice
  **[ran]**. `training/local/` is ignored by git.
- To have the Hub's copy compared with the local one, add the frame count from step 1.1 to a file by hand:
  `"EXPECTED_FRAMES": N`.
- August's command named the W&B account (`--wandb.entity`). To do the same, add `"WANDB_ENTITY": "..."` to **both**
  files. Without it W&B logs to the key's default account; if that account may not create the project `tolrad-act`,
  the trainer stops within seconds and step 3.3 reports `NOT RUNNING` with the reason.

### 3.3 One machine, one run

> **2026-10-07: this launch, as printed, lost both machines.** Each was taken back by Colab 20 minutes after the
> notebook's last cell had finished, with the keep-alive running and every ping answered `200` (`docs/journal.md`,
> that day). The trainer is detached, so the kernel executes nothing after the launch, and that is what Colab now
> counts. In August the same launch held for 5 hours 50 minutes. **Until a launch from the `colab` command has been
> shown to hold past 25 minutes, start a run from a browser tab with `"FOLLOW": true` (3.5).** Both runs were
> started again that way at 16:25 and 16:37 and held: step 3.5 below is now the launch that has been run.

Start a run only when its dataset has passed steps 1 and 2. **Do the six commands without a pause**: the connection
that `colab exec` uses is good for about an hour after `colab new` **[doc]** (`watch_training.py`, lines 148 to 153),
and the machine costs units from the moment it exists.

```bash
LAYOUT=clustered                      # then, in a second terminal: spread
S="tolrad-$LAYOUT"
"$BIN/colab" new -s "$S" --gpu A100
"$BIN/colab" upload -s "$S" "$TOOLS/secrets/huggingface.env" /content/hf_token_file
"$BIN/colab" upload -s "$S" "$TOOLS/secrets/wandb.venv" /content/wandb_key_file
"$BIN/colab" upload -s "$S" "training/local/$LAYOUT.params.json" /content/run_params.json
cp training/train_act_colab.ipynb "training/local/launch-$LAYOUT.ipynb"
"$BIN/colab" exec -s "$S" -f "training/local/launch-$LAYOUT.ipynb" --timeout 1800 2>&1 \
    | tee "training/local/launch-$LAYOUT.txt"
```

All **[read]** in the source of the installed `colab` command (0.6.0); none was run.

- **`colab new`. Good:** `[colab] Session READY.` **A message that the backend rejected the accelerator:** the account
  cannot have (another) A100 now. Do not take a different GPU for one of the two runs; see the decisions at the end of
  this section.
- **`colab upload`. Good:** `[colab] Uploaded '...' to '/content/...'`, three times. The secrets travel as files; the
  tool keeps a history of every command and of everything typed into a cell, so nothing secret is ever typed.
- **`colab exec`** runs the notebook's 14 code cells in order and prints `[colab] Executing cell 7/14 - ...` before
  each. `--timeout` is per cell; the longest cell waits up to 15 minutes for the first training step. The copy under
  `training/local/` is run so that the output notebook the tool writes beside it stays out of git.
- **Good, at the end:** `STARTED at ...` from cell 11, then from cell 12 a report with `trainer running: True`,
  `first logged step:  step:200 smpl:6K ...`, `the trainer counted: 55 episodes, N frames`,
  `for the watchers:  WANDB_RUN=...`, and a block headed `RUN RECORD`.
- **Not good:** any of `NOT STARTED`, `NOT RUNNING`, `NO LOGGED STEP`, `STOP THIS RUN`, `THE TRAINER HAS STOPPED`, or a
  Python traceback. Read from the top for the first failed check: later cells only repeat that a gate is shut.
- **The tool's exit status says nothing.** It carries on after a failed cell and exits 0 either way
  (`colab_cli/commands/execution.py`, lines 192 to 232) **[read]**. That is why the notebook has gates: every check
  records that it passed, and the launch cell starts nothing unless all nine did. Read the text.
- **After fixing what a check found** (a missing tag, a wrong name in the parameter file, which is then uploaded
  again): run the same `colab exec` line again. A second pass starts nothing if a trainer is already running.
- **The environment cell asks for a kernel restart:** `"$BIN/colab" restart-kernel -s "$S"`, then `colab exec` again.

What the notebook's checks cover, in order: the parameters; an A100 and enough disk; LeRobot 0.6.1 with a CUDA build of
PyTorch; the Hub token can act for `NAMESPACE` and is not read-only; the dataset repository is private; the policy
repository can be created, is private and holds no checkpoint from an earlier attempt; the dataset loads from the Hub
with 55 episodes (step 2); the command has no held-out split. After the start it reads the trainer's log and stops
with an error if the trainer says it held episodes out, or counted a number of episodes other than 55.

### 3.4 Straight after both have started

```bash
grep -h '^ENV' training/local/launch-clustered.txt training/local/launch-spread.txt | sort -u | wc -l
ps ax -o pid,etime,command | grep "[c]olab_cli.cli.*keep-alive"
caffeinate -is -t 21600 &
```

- **`grep`. Good:** `1`: the two machines have the same Python, LeRobot, PyTorch, GPU, processor count and package
  list. **`2`:** look at the two `ENV` lines. A different `pip_freeze_sha256` alone means some package differs: write
  it down. A different PyTorch or GPU model: stop and decide before training goes on.
- **`ps`. Good:** two lines, each ending in a machine's address and `tolrad-clustered` or `tolrad-spread`. These are
  the keep-alives (step 4). Run this line by itself. **[ran]** against two stand-in processes of the same shape.
- **`caffeinate`** keeps this machine awake for six hours **[ran]** (flags only). Plug it in and leave the lid open.
  Step 4 says why this matters, and why it is not enough to rely on.

From here on, **do not run `colab exec` on these sessions again** (step 4).

### 3.5 The same thing from a browser (the launch that held on 2026-10-07)

**Run on 2026-10-07 for both runs.** The steps as they were done: upload `training/train_act_colab.ipynb` (File,
Upload notebook; once per run: each copy gets its own machine), choose an A100 runtime, upload three files with the
Files panel, which opens in `/content/` (the run's `run_params.json` with `"FOLLOW": true`, and the two secret files,
renamed there to `hf_token_file` and `wandb_key_file`), then Runtime, Run all. With the parameter file in place the
first cell is not edited. **If the follow switch was left off** (cell 13 prints `FOLLOW is off`): add a cell with
`FOLLOW = True`, run it, and run cell 13 again; the trainer is not touched, and "Run all" is not pressed a second
time. Cell 13 prints its first line after five minutes. What follows is the page's earlier text. Open `colab.research.google.com`, upload `training/train_act_colab.ipynb`, choose an A100 runtime, drop the
two secret files into `/content/` with the Files panel and rename them there to `hf_token_file` and `wandb_key_file`,
set `NAMESPACE`, `DATASET_NAME`, `RUN_NAME` and `FOLLOW = True` in the first code cell, and run all cells. With
`FOLLOW` on, cell 13 keeps running until the trainer ends, which keeps the session busy; the tab must stay open on a
machine that stays awake, unless the Colab plan runs notebooks in the background. The edited copy lives in Colab, not
in this repository.

### 3.6 Decisions that belong to a person, before the first `colab new`

- **Only one A100 is available.** Run the two one after the other on the same GPU type (10 hours), or wait. Do not put
  one run on another GPU: the paper says "identical settings", and the hardware is part of what a reader will assume.
- **Who is awake when the runs end.** A finished run leaves its machine idle and still costing units (step 8).

### 3.7 A further run of a layout with another seed (not part of this submission)

The protocol leaves a second run per layout out of the submission and names it the first extension (section 5). The
notebook takes it as one more parameter, `SEED` (since 2026-10-07; the two runs of the paper pass none and get
LeRobot's default, 1000). A run with a seed must have a `RUN_NAME` of its own, or the first cell refuses: it would
otherwise push into the paper's policy repository.

```json
{"NAMESPACE": "...", "DATASET_NAME": "tolrad-clustered-v1", "RUN_NAME": "act-tolrad-clustered-v1-s2000",
 "SEED": 2000, "EXPECTED_FRAMES": 34582, "FOLLOW": true}
```

Launch it as in 3.5, from a browser tab. Its saved settings then differ from the paper's run of the same layout in the
seed and in the four names: step 5's first command with `--allow seed` added. **Nothing of such a run enters this
submission**, and a trial with its policy is a pilot unless the protocol is changed to name it.

## 4. Watch

There is nothing to decide while the runs go: the stop is `--steps=80000`, inside the trainer. Watching is for knowing
early that a run has died.

### 4.1 What depends on this machine staying awake

| Thing | Where it lives | If this machine sleeps |
|---|---|---|
| the trainer | the Colab machine, detached from the notebook | keeps going |
| the stop at 80,000 steps | the trainer | still happens |
| checkpoints | pushed to the Hub by the trainer, every 10,000 steps | keep arriving |
| the curves | sent to W&B by the trainer, every 200 steps | keep arriving |
| **the keep-alive** | **this machine**: `colab new` leaves a background process here that tells Colab once a minute that the session is in use (`colab_cli/commands/session.py`, lines 386 to 495) **[read]** | **stops** |
| any loop that polls from here | this machine | stops: nothing on this page uses one |

August's rule was that this machine sleeping must not matter (`../lerobot-101/docs/procedures/overnight-runs.md`,
lines 7 to 9 and 28 to 34). For the first four rows it does not. **For the keep-alive it is not known.** August has
one observation each way: a machine reclaimed about 20 minutes after its keep-alive stopped (2026-08-14,
`watch_training.py`, lines 148 to 153), and a machine that trained through a night in which this one slept
(2026-08-19, put down to the plan's background execution). The tool's own source says a session without keep-alive is
removed as idle. So:

- **Keep this machine awake** (step 3.4). That is a precaution, not the safeguard.
- **The safeguard is the Hub.** A lost machine costs at most the steps since the last checkpoint, 38 minutes of
  training, and the run resumes from there (step 6).
- **Never poll a training machine with `colab exec`.** When its connection has expired, about an hour after
  `colab new`, the tool takes the error to mean the session is gone, deletes its record and stops that session's
  keep-alive (`execution.py`, lines 178 to 184; `common.py`, lines 71 to 79) **[read]**. That is how August lost a run
  at step 13,000 **[doc]**. `colab sessions` and `colab status` are safer but also delete the record of any session
  the server does not list. While both runs are healthy, leave the `colab` command alone.

### 4.2 Which of August's watchers still work

| Watcher | This week | Why |
|---|---|---|
| `scripts/ops/watch_training.py`, W&B part | **works**, once per run, with `WANDB_RUN` set | it reads state, heartbeat, step and losses; none of that needs an evaluation loss |
| the same script's last line, `hub checkpoints:` | **misleading: ignore it** | the repository is a constant in the script (line 41). It lists **August's** checkpoints and looks healthy whatever this week's runs do |
| the same script with `COLAB_EXEC=1` | **do not use** | it runs `colab exec` (see above), and its progress pattern expects 100,000 steps (line 72) |
| `scripts/ops/eval_plateau.py` | **does not work, and is not needed** | there is no evaluation loss: it stops with `LookupError` (lines 71 to 76), which is the loud failure it was built to give. Nothing is stopped early this week |
| `scripts/ops/colab_orphan.py` | **works unchanged** | it lists and ends machines the `colab` command has lost; nothing in it depends on the training settings |

All **[read]**; none can be run before a run exists.

### 4.3 What to look at instead

**The curves and the heartbeat.** On any device: the run's page, `https://wandb.ai/<account>/<project>/runs/<id>`,
from the three parts that cell 12 printed after `WANDB_RUN=`. Or, in a terminal:

```bash
WANDB_RUN="<the value cell 12 printed after WANDB_RUN=>" "$RIG_PY" "$TOOLS/scripts/ops/watch_training.py"
```

- **Good:** `state: running`, a `heartbeat` a minute or two old at most, `step:` a number that is larger at the next
  look, three `train/` loss lines, and `avg` near 4.3 steps a second. It reads the W&B key from the earlier project's
  `secrets/` folder. Ignore its last line (table above).

**The checkpoints on the Hub:**

```bash
"$BIN/hf" models ls "$NAMESPACE/act-tolrad-$LAYOUT-v1" --tree
```

- **Good:** a folder `checkpoints/010000/` about 40 minutes after the first step, holding `pretrained_model/` (seven
  files, among them `model.safetensors` and `train_config.json`) and `training_state/` (four files); then one more
  folder every 38 minutes. An upload is one commit, so a folder that is there is whole **[read]**
  (`common/train_utils.py:526`).
- **An error that the repository is not found** means a wrong name or a wrong login, not "nothing yet": the notebook
  creates the repository before training starts.

**Expected times**, from August's average of 4.33 steps a second (journal, line 31):

| Step | 10,000 | 20,000 | 30,000 | 40,000 | 50,000 | 60,000 | 70,000 | 80,000 |
|---|---|---|---|---|---|---|---|---|
| After the first step | 0:38 | 1:17 | 1:55 | 2:34 | 3:12 | 3:51 | 4:29 | 5:08 |

### 4.4 A watcher counts once it has shown one real value

August's second failure was a watcher that asked W&B for a key that did not exist: "nothing yet" and "broken" looked
the same, and it would have waited for ever (journal, lines 38 to 43; `overnight-runs.md`, lines 36 to 40). Before
leaving the runs alone, see each of these, **for each run**:

1. In the launch output: `first logged step:` with a step line. The trainer runs.
2. From W&B: a step above zero that has grown between two looks, under this run's name. The curves arrive.
3. About 40 minutes in, from the Hub: `checkpoints/010000/`. The checkpoint path works: the token can write, the
   repository is the right one. Until this has been seen, the run could still die at its first checkpoint.
4. Step 5, on those two first checkpoints.

A watcher that has only ever said "none yet" has shown nothing.

## 5. After the first checkpoint: are the settings identical?

LeRobot saves the run's whole configuration beside every checkpoint. Do not wait for step 80,000 to compare them.

```bash
for LAYOUT in clustered spread; do
    "$BIN/hf" download "$NAMESPACE/act-tolrad-$LAYOUT-v1" \
        checkpoints/010000/pretrained_model/train_config.json --local-dir "data/checkpoints/act-tolrad-$LAYOUT-v1"
done
A=data/checkpoints/act-tolrad-clustered-v1/checkpoints/010000/pretrained_model/train_config.json
B=data/checkpoints/act-tolrad-spread-v1/checkpoints/010000/pretrained_model/train_config.json
"$RIG_PY" training/compare_train_configs.py "$A" "$B" \
    --must-differ dataset.repo_id --must-differ policy.repo_id --must-differ wandb.run_id
echo "exit status: $?"
```

- **Good:** five lines starting `allowed` (`dataset.repo_id`, `job_name`, `output_dir`, `policy.repo_id`,
  `wandb.run_id`), then `RESULT: identical apart from 5 allowed key(s), 129 key(s) compared.`; exit status **0**.
  **[ran]** on August's real file and two copies edited to this week's plan. The `hf download` lines are **[read]**.
- **Exit status 1:** a line starting `DIFFERS` names each setting that is not the same, with both values; or `SAME`
  says the two files are one run. Then the two runs are not "identical settings": stop and decide before more units
  are spent.
- `DIFFERS  eval.batch_size`: the two machines have different processor counts (the value is 0.7 times the count).
  It is used only for evaluation in simulation. Note it; nothing needs repeating.
- The output hides the values of the allowed keys, which hold the namespace, so it can go into the journal as it is.
  `data/checkpoints/` is ignored by git.

Against August, to show that only the four planned values changed (`settings.md`):

```bash
AUG="$(ls "$HOME"/.cache/huggingface/hub/models--*--act-so101-cube-v1/snapshots/*/checkpoints/075000/pretrained_model/train_config.json)"
"$RIG_PY" training/compare_train_configs.py "$AUG" "$A" \
    --allow dataset.eval_split --allow eval_steps --allow steps --allow save_freq
echo "exit status: $?"
```

- **Good:** `allowed  dataset.eval_split: 0.1 | 0.0`, `allowed  eval_steps: 5000 | 0`,
  `allowed  save_freq: 5000 | 10000`, `allowed  steps: 100000 | 80000`, the name keys, and
  `RESULT: identical apart from ...`; exit status **0**. **[ran]** in the same way.

**Repeat the first comparison on `checkpoints/080000/`** when both runs are done: the sentence in the paper is about
the two checkpoints that are evaluated.

## 6. If a machine is lost

**How it shows:** the W&B run's heartbeat grows old or its state turns to `crashed`; no new checkpoint arrives when
one is due; the `ps` line of step 3.4 no longer lists that session. The same signs, with the session still listed,
mean the trainer has died on a machine that is still there. One way that can happen: LeRobot does not catch a
checkpoint upload that fails, so the error ends the run **[read]** (`scripts/lerobot_train.py:669` in 0.6.0, `:838` at
`6adf515`).

**First make sure the old machine is gone**, whatever state it is in: two trainers must never push into one
repository.

```bash
"$BIN/colab" stop -s "tolrad-$LAYOUT"
"$RIG_PY" "$TOOLS/scripts/ops/colab_orphan.py"
```

- **Good:** `[colab] Session terminated.` or `Session ... not found`, and then **one** line: the other run's machine.
- **Two lines:** the lost run's machine still exists and the tool no longer knows it. The `ps` line of step 3.4 shows
  the healthy run's address beside its session name; the other address is the one to end, with
  `"$RIG_PY" "$TOOLS/scripts/ops/colab_orphan.py" --stop <address>`. Ending a machine cannot be undone: read the
  address twice.

**6.1 Resume from the Hub.** **[read]**; not exercised in August as far as its records show. LeRobot downloads the
highest-numbered `checkpoints/<step>/` of the policy repository and continues from it: weights, optimiser state,
random-number state and step counter are restored, the order of the data is taken up at that step, the settings are
the saved ones, and the W&B run goes on under its old id (`configs/train.py:222`, `common/train_utils.py:542`,
`common/wandb_utils.py:112` at `6adf515`; the same in 0.6.0).

```bash
LAYOUT=spread                         # the run that was lost
S="tolrad-$LAYOUT-r2"                 # a new session name
printf '{"NAMESPACE": "%s", "DATASET_NAME": "tolrad-%s-v1", "RUN_NAME": "act-tolrad-%s-v1", "MODE": "resume_hub"}\n' \
    "$NAMESPACE" "$LAYOUT" "$LAYOUT" > "training/local/$LAYOUT.resume.params.json"
"$BIN/colab" new -s "$S" --gpu A100
"$BIN/colab" upload -s "$S" "$TOOLS/secrets/huggingface.env" /content/hf_token_file
"$BIN/colab" upload -s "$S" "$TOOLS/secrets/wandb.venv" /content/wandb_key_file
"$BIN/colab" upload -s "$S" "training/local/$LAYOUT.resume.params.json" /content/run_params.json
cp training/train_act_colab.ipynb "training/local/resume-$LAYOUT.ipynb"
"$BIN/colab" exec -s "$S" -f "training/local/resume-$LAYOUT.ipynb" --timeout 1800 2>&1 \
    | tee "training/local/resume-$LAYOUT.txt"
```

The notebook then runs `lerobot-train --config_path=$NAMESPACE/act-tolrad-<layout>-v1 --resume=true
--output_dir=/content/train/act-tolrad-<layout>-v1`.

- **Good:** cell 7 prints the checkpoints on the Hub; cell 12 prints `resumed: Resuming data order at epoch N, sample
  M`, a first logged step above the checkpoint's step, and the same `WANDB_RUN` as before. The next checkpoint arrives
  at the next multiple of 10,000.
- **Same GPU type, same LeRobot** (the notebook enforces both). A configuration written by 0.6.1 must be read by 0.6.1.
- **If the resume itself fails**, do not improvise on it at night: the fallback rule below covers a run that stops
  short.
- A resumed run's later configurations differ from the other run's in `resume`, `checkpoint_path`,
  `policy.pretrained_path` and perhaps `output_dir`. Step 5 will print them as `DIFFERS`; that is correct. Add
  `--allow` for exactly those keys, keep the output, and report the interruption: an entry in `docs/journal.md` and
  `docs/decisions.md`, and a sentence in the paper.
- The trainer stopped but its machine is alive (a failed upload, say): that needs `colab exec`, which is unsafe on a
  session more than an hour old. A new machine and `resume_hub` is the route that was read through. (`"MODE":
  "resume_local"` exists for a browser session.)

**Time left after a loss:** (80,000 minus the last step on the Hub) / 4.33 seconds, plus about 15 minutes for the new
machine. From step 40,000 that is 2 h 34 min plus the 15.

**6.2 The fallback rule** (protocol, section 5; `config/protocol.toml`, `[training]`): "if either run cannot reach
80,000 steps before the trials must start, both policies are evaluated at their largest common saved step, and not
below 40,000."

| When the trials must start | Evaluate |
|---|---|
| both repositories hold `080000` | `080000` of each |
| one run stopped short; the largest step that **both** hold is `040000` or above | that step, of each. The finished run keeps all its earlier checkpoints, so its own checkpoint at that step exists |
| the largest common step is below `040000` | nothing, under the protocol as written. A person decides, and it is a deviation |

"Holds" means the folder `checkpoints/<step>/` is listed by the `hf models ls` line of step 4.3 for both
repositories. ACT's learning rate does not depend on the planned number of steps, so a checkpoint at a given step is
the same whether or not the run went on (`settings.md`). Whichever step is used goes into the journal, the trial
records and the paper; a step below 80,000 is logged in `docs/decisions.md`.

## 7. Bring the two step-80,000 checkpoints to the rig

Only `pretrained_model/` is needed to run a policy (about 0.2 GB; August's weights file was 206,699,736 bytes,
*measured*). The other half of a checkpoint, `training_state/`, is the optimiser's state: 0.4 GB, for resuming only.

```bash
for LAYOUT in clustered spread; do
    "$BIN/hf" download "$NAMESPACE/act-tolrad-$LAYOUT-v1" checkpoints/080000/pretrained_model/ \
        --local-dir "data/checkpoints/act-tolrad-$LAYOUT-v1"
    ls -l "data/checkpoints/act-tolrad-$LAYOUT-v1/checkpoints/080000/pretrained_model/"
done
```

**[read]** (a name ending in `/` asks for a folder). 

- **Good:** seven files in each folder: `config.json`, `model.safetensors`, `train_config.json`,
  `policy_preprocessor.json`, `policy_postprocessor.json`, and the two processors' `.safetensors` state files.

**The path the rollout code wants.** `load_policy(repo_id, checkpoint, checkpoint_path, device)`
(`../lerobot-101/scripts/policy/rollout_session.py`, lines 513 to 544) **[read]** works in one of two ways:

- **With `checkpoint_path`** (the script's `--checkpoint-path`): the path is used as it is, and it must be **the
  `pretrained_model` folder itself**, the one that directly holds `config.json` and `model.safetensors`. For this
  experiment:

  ```
  data/checkpoints/act-tolrad-clustered-v1/checkpoints/080000/pretrained_model
  data/checkpoints/act-tolrad-spread-v1/checkpoints/080000/pretrained_model
  ```

  `repo_id` and `checkpoint` are then not used for loading. One folder too high (`.../080000`) fails with
  `ParsingError: Expected a dict with a 'type' key ... got {}` **[ran]**: that message means the path is wrong.
- **Without it:** the function downloads `checkpoints/<checkpoint>/**` of `repo_id` into the Hub cache at every start
  and uses `<cache>/models--<namespace>--<repository>/snapshots/<commit>/checkpoints/<checkpoint>/pretrained_model`.
  That needs the network at the rig and fetches the optimiser's state too. Use the local path.

Two traps in the earlier script, both **[read]**:

- Its `--repo-id` and `--checkpoint` default to August's policy and `075000` (lines 792 and 793). With
  `--checkpoint-path` alone, the right policy runs and the trial is **recorded under August's names**. Whatever starts
  a trial must pass this week's names; a trial log takes the repository's name without the namespace.
- LeRobot loads weights without a strict check of their names (`policies/pretrained.py:174` in 0.6.0): a mismatch
  between the LeRobot that trained and the one that loads would be logged and the policy would run with some untrained
  weights. So check each checkpoint once, with no hardware:

```bash
LAYOUT=clustered                      # then: spread
CKPT="data/checkpoints/act-tolrad-$LAYOUT-v1/checkpoints/080000/pretrained_model"
HF_HUB_OFFLINE=1 "$RIG_PY" - "$TOOLS/scripts/policy/rollout_session.py" "$CKPT" <<'EOF'
import importlib.util, sys
from safetensors import safe_open
spec = importlib.util.spec_from_file_location("rollout_session", sys.argv[1])
rollout = importlib.util.module_from_spec(spec); spec.loader.exec_module(rollout)
policy, _, _ = rollout.load_policy(None, None, sys.argv[2], "cpu")
with safe_open(f"{sys.argv[2]}/model.safetensors", framework="pt") as weights:
    in_file = set(weights.keys())
in_model = set(policy.state_dict())
print(f"tensors in the file: {len(in_file)} | in the model: {len(in_model)} | only in one: {sorted(in_file ^ in_model)[:5]}")
print(f"parameters: {sum(p.numel() for p in policy.parameters()):,}")
sys.exit(0 if in_file == in_model else 1)
EOF
echo "exit status: $?"
shasum -a 256 "$CKPT/model.safetensors"
```

- **Good:** `tensors in the file: 234 | in the model: 234 | only in one: []`, `parameters: 51,597,190`, exit status
  **0**. **[ran]** with exactly this text on August's checkpoint, with the rig's interpreter as it is today (LeRobot
  0.6.0): 2 seconds. The same numbers are expected this week, since the policy and the cameras are the same.
- The `shasum` line gives the hash that `protocol/data-contracts.md` asks to be written into the journal.

**Which LeRobot loads them at the rig** is a separate open point (`docs/reference/lerobot101-interfaces.md`, finding
1): the rig's interpreter now imports release 0.6.0, not the checkout at `6adf515` that ran August's trials. What is
known about a checkpoint written by Colab's 0.6.1, taking August's as the sample:

- under **0.6.0** it loads with every weight matched (the check above, **[ran]**);
- under **`6adf515`** it ran August's 31 trials of 2026-08-19 **[doc]** (the rig then used the checkout directly, and
  the checkout's own log shows it at `6adf515` from 2026-08-16); and by reading, that checkout's ACT model differs
  from 0.6.0's by one class attribute and no weight (`policies/act/modeling_act.py`), and its ACT configuration file
  is identical;
- in both, every key of the saved `config.json` is a field of the configuration class, none is missing, and the five
  saved processor steps are registered (*checked by script*, on the source text).

No difference was found that would stop a 0.6.1 checkpoint from loading in either. After any change of the rig's
interpreter, run the check above again.

**An earlier step** (for a check before the runs end, or for the fallback) is fetched the same way with its number in
place of `080000`. The protocol's Tuesday-night stop line asks for a clustered checkpoint that repeats 8 of 9
demonstrations and does not name the step: decide it beforehand.

Finally, step 5's first comparison on the two `080000` configurations.

## 8. Release the machines, and write the record

A run that has ended leaves its machine idle, at the same cost per hour: two idle A100s cost about 10.6 units an hour
(*computed* from August's 5.3). First make sure the work is on the Hub:

```bash
"$BIN/hf" models ls "$NAMESPACE/act-tolrad-$LAYOUT-v1" --tree
```

- **Good:** `checkpoints/080000/` is there, and `model.safetensors`, `config.json` and `train_config.json` sit at the
  top of the repository: the trainer puts the final policy there after its last step **[read]**. It is the same
  weights as `checkpoints/080000/`; the experiment uses the checkpoint folder, which names its step.

```bash
"$BIN/colab" stop -s "tolrad-$LAYOUT"
```

- **Good:** `[colab] Session terminated.` **`Session ... not found`:** the tool has lost it; it may still exist and
  cost units.
- When both are stopped, ask the server what is left **[read]**:

  ```bash
  "$RIG_PY" "$TOOLS/scripts/ops/colab_orphan.py"
  ```

  **Good:** `no active assignments`. **A line with an address:** a machine is still running; end it with the same
  script and `--stop <address>`. (This script fails loudly when the login has expired, where `colab sessions` would
  report that nothing is running.)
- The trainer's log (`/content/train.log`) dies with the machine. `"$BIN/colab" download -s "$S" /content/train.log
  "training/local/train-$LAYOUT.log"` may still fetch it, or may fail because the connection has expired; the curves
  are in W&B either way.

**Into `docs/journal.md`**, with times from the clock: the results of step 1; when each dataset was uploaded; each
run's `RUN RECORD` block (it names no account); when each checkpoint arrived; the two outputs of step 5; any
interruption and resume; steps per second and hours; the units spent; the two hashes of step 7; which LeRobot loaded
the checkpoints. **Into `docs/decisions.md`**: the pass line used for torn frames, and anything that departed from this
page.

## Time and cost

| Figure | Value | Where it comes from |
|---|---|---|
| Speed | 4.33 steps a second | journal, line 31: August's average over 90,200 steps on an A100, including its evaluation passes and 18 checkpoint saves |
| One run of 80,000 steps | 5 h 08 min | *computed*: 80,000 / 4.33. At the 4.5 steps a second of a 200-step trial on 2026-08-14 (August notebook, cell 7) it would be 4 h 56 min |
| Between checkpoints | 38.5 min | *computed*: 10,000 / 4.33 |
| Price of an A100 | about 5.3 compute units an hour | `overnight-runs.md`, line 23: August's figure. Colab sets it and may have changed it |
| One run | about 27 units | *computed*: 5.13 h x 5.3. August's rule of thumb, steps / 2,900, gives 27.6 |
| Both runs | about 55 units | *computed* |
| Set-up per machine | 10 to 15 min, about 1 unit | *estimate*: not measured |
| An idle machine after the end | 5.3 units an hour, each | the same price |
| In money | about US$2.70 a run | *assumed*: US$9.99 for 100 units, from memory, not checked |
| Dataset upload | about 0.42 GB each | *computed* from the 206 MB of the 27 episodes recorded on 2026-10-06 (*measured*); August's 55-episode set is 686 MB |

Not included: a resumed run's repeated set-up, and the rehearsal of step 0.6 (one to four units).

## How far this page was checked

| Checked by running, on 2026-10-05 | Result |
|---|---|
| `verify_dataset.py` with the rig's interpreter as it is now, on August's set | works; exit 0, and exit 1 with `--expect-seconds 25` or a wrong episode count |
| `scripts/rig/torn_frames.py`, on the first 27 episodes | `NO TORN FRAME`, exit 0, about four minutes |
| `compare_train_configs.py` | 34 tests pass; right verdicts on August's real file |
| the notebook | well-formed (nbformat 4.5), every cell compiles; 61 tests run its cells against stand-ins for the Hub, the dataset and the trainer. **Never run on Colab** |
| the checkpoint check of step 7 | passes on August's checkpoint under LeRobot 0.6.0 |

| Read in source, not run | Where |
|---|---|
| `hf` commands | huggingface_hub 1.26.0, in the rig's environment |
| `colab` commands, their keep-alive and their handling of failed cells | google-colab-cli 0.6.0, in the rig's environment |
| LeRobot's training rules | release 0.6.0 as installed, and the checkout at `6adf515`; see `settings.md` |

| Not verified | What would settle it |
|---|---|
| LeRobot **0.6.1** itself, the version Colab installs | `git diff v0.6.1 6adf515 --stat` on the paths named in `settings.md`, in the LeRobot checkout |
| A resume from the Hub | not done in August as far as its records show; the rehearsal of step 0.6 can include one |
| Whether a Colab machine survives its keep-alive stopping | not knowable from here; step 4.1 |
| Two A100 machines at once on this account | step 3.3 finds out |
| That a dataset written by the rig's present LeRobot loads in 0.6.1 | cell 8 of the notebook; step 0.6 tests it early |

## Terms

To be added to `protocol/glossary.md`, which the task that wrote this page could not edit.

| Term | Meaning |
|---|---|
| **Checkpoint** | The policy's weights and the trainer's state saved at a given step, as `checkpoints/<step>/` |
| **Compute unit** | What Colab charges in; an A100 used about 5.3 an hour in August |
| **Evaluation loss** | The training loss computed on held-out episodes. August used it to pick a checkpoint; this week there is none |
| **Gate** | In the notebook, the record that one check has passed; the trainer starts only when all gates are open |
| **Held-out split** | Episodes kept out of training to compute an evaluation loss. None this week |
| **Hub** | The Hugging Face Hub, where the datasets and the policies' checkpoints are stored, in private repositories |
| **Keep-alive** | The signal that tells Colab a session is in use; sent once a minute by a process on the machine that ran `colab new` |
| **Namespace** | The account or organisation that owns a repository on the Hub. Never written into this repository |
| **Version tag** | The tag `v3.0` on a dataset repository, without which LeRobot will not load it from the Hub |
