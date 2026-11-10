# Training settings

What the August training run used, setting by setting, and what changes this week. The table in the paper's appendix is
taken from here.

**Status: prepared on 2026-10-05, before any run of this week.** The column "This week" is the plan. It becomes a
record when each run's own `train_config.json` has been compared with August's (last section).

## Where each value comes from

Every value carries one of three labels.

- **file**: read from the `train_config.json` that LeRobot saved with the August checkpoint
  (`checkpoints/075000/pretrained_model/train_config.json` in the August policy repository), read on 2026-10-05 from
  this machine's Hub cache. This file is the authority on what was used. It holds 129 values, and all of them are
  listed below: 73 in the first table, 56 in the second. Every entry in a column headed "August value" is a *file*
  value.
- **journal**: read from the earlier project's run journal, `../lerobot-101/docs/run-journal.md` at commit `02b3ced`,
  with the line. These are facts the configuration file does not hold.
- **source**: a default or a rule looked up in LeRobot's source, as `file:line` under `src/lerobot/` in the checkout at
  commit `6adf515`. That checkout is **not** the code that trained in August: Colab installed release 0.6.1, and
  `6adf515` is a later commit. A default is cited only where the August file holds the same value. The two tables were
  produced from the file and the source by a one-off script on 2026-10-05, which refused any row marked "default"
  whose value differed from the cited default, and then checked all 129 rows against the file again.

The column "Set by" says how a value came about:

- `flag`: passed on the command line. The 19 flags are those of the August notebook's training cell
  (`../lerobot-101/notebooks/train_act_colab.ipynb`, cell 9).
- `default`: left at LeRobot's default.
- `derived`: worked out when the run starts, from the dataset, from the policy's preset, from the machine or by W&B.

## What changes this week

Four values, all set by the protocol (`protocol/protocol.md`, section 5). They are two decisions: train on every episode
(the first two rows, which LeRobot ties together) and stop at a fixed step (the last two).

| Key | August | This week | Why |
|---|---|---|---|
| `dataset.eval_split` | `0.1` | **`0.0`** | every episode is trained on; none is held out |
| `eval_steps` | `5000` | **`0`** | no held-out episodes, so no evaluation loss; LeRobot refuses a positive value without a split |
| `steps` | `100000` | **`80000`** | a fixed number of steps, decided before any result |
| `save_freq` | `5000` | **`10000`** | checkpoints at 10,000, 20,000, ... and 80,000 |

Some values differ because they name the run, not because a setting changed: `dataset.repo_id`, `policy.repo_id`,
`output_dir`, `job_name`, `wandb.run_id` and, against August, the W&B project and account. Between this week's two runs
the first five differ, and they are the only values that may.

Two things follow from the four changes and are not settings:

- There is no evaluation loss this week. August chose its checkpoint by that loss; this week the step is fixed in
  advance and nothing is chosen.
- At batch 32, 80,000 steps are 2,560,000 samples: **74.0 passes over the clustered set's 34,582 frames and 76.7
  over the spread set's 33,365** (*measured* 2026-10-07, the dataset check of runbook 2's step 7 on the two finished
  sets of 55 episodes; a take ends on the operator's key, so the two counts differ, by 3.6 %). The step count is the
  same for both runs, the passes are not. August's
  evaluated checkpoint, step 75,000, had
  made about 67 passes over its 49 training episodes (*computed* from the journal's 80.3 passes at 90,200 steps).

## For the appendix

A short form of the first table. Values in bold are this week's changes.

| Setting | Value for both runs this week | Keys |
|---|---|---|
| Policy | ACT, 51,597,190 parameters | `policy.type`; the count was measured (see "Facts") |
| Vision backbone | ResNet-18, started from ImageNet weights (`ResNet18_Weights.IMAGENET1K_V1`) | `policy.vision_backbone`, `policy.pretrained_backbone_weights` |
| Transformer | width 512, 8 heads, feed-forward width 3200, 4 encoder layers, 1 decoder layer, dropout 0.1, ReLU activation, normalisation after each block | `policy.dim_model`, `n_heads`, `dim_feedforward`, `n_encoder_layers`, `n_decoder_layers`, `dropout`, `feedforward_activation`, `pre_norm` |
| Variational encoder | on; latent size 32; 4 layers; KL weight 10 | `policy.use_vae`, `latent_dim`, `n_vae_encoder_layers`, `kl_weight` |
| Action chunk | 100 future actions predicted at a time | `policy.chunk_size` |
| Inputs | 6 joint positions; two colour cameras, front and wrist, 640 x 480; 1 observation step | `policy.input_features`, `policy.n_obs_steps` |
| Output | 6 joint position targets | `policy.output_features` |
| Normalisation | mean and standard deviation for every input and output; ImageNet's statistics for the cameras | `policy.normalization_mapping`, `dataset.use_imagenet_stats` |
| Image augmentation | none | `dataset.image_transforms.enable` |
| Optimiser | AdamW; learning rate 1e-05 (backbone 1e-05); weight decay 0.0001; betas (0.9, 0.999); epsilon 1e-08; gradient norm clipped at 10 | `optimizer.*`, `policy.optimizer_lr_backbone` |
| Learning-rate schedule | none: constant | `scheduler` |
| Batch size | 32 | `batch_size` |
| Steps | **80,000, fixed in advance** (August: a ceiling of 100,000, stopped by hand at 90,200, checkpoint 75,000 evaluated) | `steps` |
| Episodes trained on | **all 55 of each set** (August: 49 of 55, the last 6 held out for an evaluation loss) | `dataset.eval_split`, `eval_steps` |
| Checkpoints | **every 10,000 steps**; step 80,000 is the one evaluated (August: every 5,000) | `save_freq` |
| Seed | 1000, the same for both runs | `seed` |
| Arithmetic | 32-bit floats, no mixed precision | `policy.use_amp` |
| Software and machine | LeRobot 0.6.1 on one A100 GPU, as in August; to be confirmed from the `ENV` line each run prints | notebook, cell 6 |

## Table 1. Settings that define the training (73 values)

| Key | August value | Set by | Default or rule in LeRobot (`src/lerobot/`, commit `6adf515`) | This week | Note |
|---|---|---|---|---|---|
| **Data** | | | | | |
| `dataset.repo_id` | `NAMESPACE/so101-cube-v1` | flag | (none: required) | this run's own |  |
| `dataset.root` | `null` | default | `None` at `configs/default.py:40` | same |  |
| `dataset.episodes` | `null` | default | `None` at `configs/default.py:41` | same | no episode list: every episode, less the held-out split |
| `dataset.revision` | `null` | default | `None` at `configs/default.py:45` | same |  |
| `dataset.eval_split` | `0.1` | flag | `0.0` at `configs/default.py:56` | **`0.0`** | August held out the last 6 episodes (49 to 54) and trained on 49 |
| `dataset.image_transforms.enable` | `false` | default | `False` at `transforms/transforms.py:642` | same |  |
| `dataset.use_imagenet_stats` | `true` | default | `True` at `configs/default.py:46` | same | camera frames are normalised with ImageNet's mean and deviation |
| `dataset.video_backend` | `"torchcodec"` | derived | rule at `utils/import_utils.py:72` | resolved on the machine | `torchcodec` when it can be imported on the machine |
| `dataset.streaming` | `false` | default | `False` at `configs/default.py:54` | same |  |
| `tolerance_s` | `0.0001` | default | `1e-4` at `configs/train.py:150` | same |  |
| `policy.input_features.observation.state.type` | `"STATE"` | derived |  | same | taken from the dataset's features |
| `policy.input_features.observation.state.shape` | `[6]` | derived |  | same | taken from the dataset's features |
| `policy.input_features.observation.images.front.type` | `"VISUAL"` | derived |  | same | taken from the dataset's features |
| `policy.input_features.observation.images.front.shape` | `[3, 480, 640]` | derived |  | same | taken from the dataset's features |
| `policy.input_features.observation.images.wrist.type` | `"VISUAL"` | derived |  | same | taken from the dataset's features |
| `policy.input_features.observation.images.wrist.shape` | `[3, 480, 640]` | derived |  | same | taken from the dataset's features |
| `policy.output_features.action.type` | `"ACTION"` | derived |  | same | taken from the dataset's features |
| `policy.output_features.action.shape` | `[6]` | derived |  | same | taken from the dataset's features |
| **Policy (ACT)** | | | | | |
| `policy.type` | `"act"` | flag | (none: this flag chooses the policy) | same |  |
| `policy.n_obs_steps` | `1` | default | `1` at `policies/act/configuration_act.py:84` | same |  |
| `policy.chunk_size` | `100` | default | `100` at `policies/act/configuration_act.py:85` | same |  |
| `policy.n_action_steps` | `100` | default | `100` at `policies/act/configuration_act.py:86` | same | overridden at evaluation by temporal ensembling |
| `policy.normalization_mapping.VISUAL` | `"MEAN_STD"` | default | `MEAN_STD` at `policies/act/configuration_act.py:88` | same |  |
| `policy.normalization_mapping.STATE` | `"MEAN_STD"` | default | `MEAN_STD` at `policies/act/configuration_act.py:88` | same |  |
| `policy.normalization_mapping.ACTION` | `"MEAN_STD"` | default | `MEAN_STD` at `policies/act/configuration_act.py:88` | same |  |
| `policy.vision_backbone` | `"resnet18"` | default | `"resnet18"` at `policies/act/configuration_act.py:98` | same |  |
| `policy.pretrained_backbone_weights` | `"ResNet18_Weights.IMAGENET1K_V1"` | default | `"ResNet18_Weights.IMAGENET1K_V1"` at `policies/act/configuration_act.py:99` | same |  |
| `policy.replace_final_stride_with_dilation` | `false` | default | `False` at `policies/act/configuration_act.py:100` | same |  |
| `policy.pre_norm` | `false` | default | `False` at `policies/act/configuration_act.py:102` | same |  |
| `policy.dim_model` | `512` | default | `512` at `policies/act/configuration_act.py:103` | same |  |
| `policy.n_heads` | `8` | default | `8` at `policies/act/configuration_act.py:104` | same |  |
| `policy.dim_feedforward` | `3200` | default | `3200` at `policies/act/configuration_act.py:105` | same |  |
| `policy.feedforward_activation` | `"relu"` | default | `"relu"` at `policies/act/configuration_act.py:106` | same |  |
| `policy.n_encoder_layers` | `4` | default | `4` at `policies/act/configuration_act.py:107` | same |  |
| `policy.n_decoder_layers` | `1` | default | `1` at `policies/act/configuration_act.py:111` | same |  |
| `policy.use_vae` | `true` | default | `True` at `policies/act/configuration_act.py:113` | same |  |
| `policy.latent_dim` | `32` | default | `32` at `policies/act/configuration_act.py:114` | same |  |
| `policy.n_vae_encoder_layers` | `4` | default | `4` at `policies/act/configuration_act.py:115` | same |  |
| `policy.temporal_ensemble_coeff` | `null` | default | `None` at `policies/act/configuration_act.py:119` | same | not used in training; set at evaluation (protocol, section 6) |
| `policy.dropout` | `0.1` | default | `0.1` at `policies/act/configuration_act.py:122` | same |  |
| `policy.kl_weight` | `10.0` | default | `10.0` at `policies/act/configuration_act.py:123` | same |  |
| `policy.pretrained_path` | `null` | default | `None` at `configs/policies.py:81` | same | no pretrained policy: only the ResNet-18 backbone starts from ImageNet weights |
| `policy.pretrained_revision` | `null` | default | `None` at `configs/policies.py:83` | same |  |
| `policy.device` | `"cuda"` | flag | `None` at `configs/policies.py:62` | same |  |
| `policy.use_amp` | `false` | default | `False` at `configs/policies.py:65` | same |  |
| **Optimisation** | | | | | |
| `policy.optimizer_lr` | `1e-05` | default | `1e-5` at `policies/act/configuration_act.py:126` | same |  |
| `policy.optimizer_weight_decay` | `0.0001` | default | `1e-4` at `policies/act/configuration_act.py:127` | same |  |
| `policy.optimizer_lr_backbone` | `1e-05` | default | `1e-5` at `policies/act/configuration_act.py:128` | same |  |
| `use_policy_training_preset` | `true` | default | `True` at `configs/train.py:157` | same |  |
| `optimizer.type` | `"adamw"` | derived | rule at `policies/act/configuration_act.py:153` | same | ACT's optimiser preset |
| `optimizer.lr` | `1e-05` | derived | rule at `policies/act/configuration_act.py:155` | same | the preset copies `policy.optimizer_lr` |
| `optimizer.weight_decay` | `0.0001` | derived | rule at `policies/act/configuration_act.py:156` | same | the preset copies `policy.optimizer_weight_decay` |
| `optimizer.grad_clip_norm` | `10.0` | default | `10.0` at `optim/optimizers.py:111` | same | AdamW's default in LeRobot, kept by the preset |
| `optimizer.betas` | `[0.9, 0.999]` | default | `(0.9, 0.999)` at `optim/optimizers.py:108` | same |  |
| `optimizer.eps` | `1e-08` | default | `1e-8` at `optim/optimizers.py:109` | same |  |
| `scheduler` | `null` | derived | rule at `policies/act/configuration_act.py:159` | same | ACT's preset has no schedule: the learning rate is constant |
| `seed` | `1000` | default | `1000` at `configs/train.py:127` | same | the same seed for both runs: one run per dataset, run-to-run variation not measured |
| `cudnn_deterministic` | `false` | default | `False` at `configs/train.py:130` | same |  |
| `batch_size` | `32` | flag | `8` at `configs/train.py:133` | same |  |
| `num_workers` | `8` | flag | `4` at `configs/train.py:132` | same |  |
| `prefetch_factor` | `4` | default | `4` at `configs/train.py:134` | same |  |
| `persistent_workers` | `true` | default | `True` at `configs/train.py:135` | same |  |
| `dataloader_multiprocessing_context` | `"spawn"` | default | `"spawn"` at `configs/train.py:141` | same |  |
| **Steps, logging and checkpoints** | | | | | |
| `steps` | `100000` | flag (equal to the default) | `100_000` at `configs/train.py:142` | **`80000`** | a ceiling in August: the run was stopped by hand at 90,200 |
| `log_freq` | `200` | flag (equal to the default) | `200` at `configs/train.py:145` | same |  |
| `eval_steps` | `5000` | flag | `0` at `configs/train.py:147` | **`0`** | an evaluation loss every 5,000 steps in August; none now |
| `max_eval_samples` | `512` | flag | `0` at `configs/train.py:149` | same (no effect without a split) | read only when a held-out split exists |
| `save_checkpoint` | `true` | default | `True` at `configs/train.py:151` | same |  |
| `save_freq` | `5000` | flag | `20_000` at `configs/train.py:154` | **`10000`** | the last step is saved whatever the frequency |
| `save_checkpoint_to_hub` | `true` | flag | `False` at `configs/train.py:176` | same |  |
| `resume` | `false` | default | `False` at `configs/train.py:124` | same | true in the configuration of a run resumed after an interruption |
| `checkpoint_path` | `null` | default | none at `configs/train.py:183` | same | set by LeRobot when a run is resumed |
| `wandb.enable` | `true` | flag | `False` at `configs/default.py:95` | same | curves are logged to W&B; nothing reads them back into training |

## Table 2. Values in the file that do not shape the training (56 values)

They are listed so that every value of the file is accounted for.

| Key | August value | Set by | Default or rule in LeRobot (`src/lerobot/`, commit `6adf515`) | This week | Note |
|---|---|---|---|---|---|
| **Names and destinations** | | | | | |
| `output_dir` | `"/content/train/act-cube-v1"` | flag | `None` at `configs/train.py:117` | this run's own |  |
| `job_name` | `"act-cube-v1"` | flag | `None` at `configs/train.py:118` | this run's own |  |
| `policy.repo_id` | `NAMESPACE/act-so101-cube-v1` | flag | `None` at `configs/policies.py:71` | this run's own |  |
| `policy.push_to_hub` | `true` | default | `True` at `configs/policies.py:70` | same |  |
| `policy.private` | `true` | flag | `None` at `configs/policies.py:74` | same |  |
| `policy.tags` | `null` | default | `None` at `configs/policies.py:76` | same |  |
| `policy.license` | `null` | default | `None` at `configs/policies.py:78` | same |  |
| `dataset.repo_type` | `"dataset"` | default | `"dataset"` at `configs/default.py:37` | same |  |
| `wandb.project` | `"so101-act"` | flag | `"lerobot"` at `configs/default.py:98` | `"tolrad-act"`, this experiment's project |  |
| `wandb.entity` | (an account name, not reproduced) | flag | `None` at `configs/default.py:99` | the key's default, unless set |  |
| `wandb.run_id` | (a run identifier, not reproduced) | derived | rule at `common/wandb_utils.py:118` | this run's own | assigned by W&B when the run starts |
| `wandb.notes` | `null` | default | `None` at `configs/default.py:100` | same |  |
| `wandb.mode` | `null` | default | `None` at `configs/default.py:102` | same |  |
| `wandb.add_tags` | `true` | default | `True` at `configs/default.py:103` | same |  |
| `wandb.disable_artifact` | `false` | default | `False` at `configs/default.py:97` | same |  |
| **Image augmentation, switched off (`dataset.image_transforms.enable` is false)** | | | | | |
| `dataset.image_transforms.max_num_transforms` | `3` | default | `3` at `transforms/transforms.py:645` | same |  |
| `dataset.image_transforms.random_order` | `false` | default | `False` at `transforms/transforms.py:648` | same |  |
| `dataset.image_transforms.tfs.brightness.weight` | `1.0` | default | defined at `transforms/transforms.py:649` | same |  |
| `dataset.image_transforms.tfs.brightness.type` | `"ColorJitter"` | default | defined at `transforms/transforms.py:649` | same |  |
| `dataset.image_transforms.tfs.brightness.kwargs.brightness` | `[0.8, 1.2]` | default | defined at `transforms/transforms.py:649` | same |  |
| `dataset.image_transforms.tfs.contrast.weight` | `1.0` | default | defined at `transforms/transforms.py:649` | same |  |
| `dataset.image_transforms.tfs.contrast.type` | `"ColorJitter"` | default | defined at `transforms/transforms.py:649` | same |  |
| `dataset.image_transforms.tfs.contrast.kwargs.contrast` | `[0.8, 1.2]` | default | defined at `transforms/transforms.py:649` | same |  |
| `dataset.image_transforms.tfs.saturation.weight` | `1.0` | default | defined at `transforms/transforms.py:649` | same |  |
| `dataset.image_transforms.tfs.saturation.type` | `"ColorJitter"` | default | defined at `transforms/transforms.py:649` | same |  |
| `dataset.image_transforms.tfs.saturation.kwargs.saturation` | `[0.5, 1.5]` | default | defined at `transforms/transforms.py:649` | same |  |
| `dataset.image_transforms.tfs.hue.weight` | `1.0` | default | defined at `transforms/transforms.py:649` | same |  |
| `dataset.image_transforms.tfs.hue.type` | `"ColorJitter"` | default | defined at `transforms/transforms.py:649` | same |  |
| `dataset.image_transforms.tfs.hue.kwargs.hue` | `[-0.05, 0.05]` | default | defined at `transforms/transforms.py:649` | same |  |
| `dataset.image_transforms.tfs.sharpness.weight` | `1.0` | default | defined at `transforms/transforms.py:649` | same |  |
| `dataset.image_transforms.tfs.sharpness.type` | `"SharpnessJitter"` | default | defined at `transforms/transforms.py:649` | same |  |
| `dataset.image_transforms.tfs.sharpness.kwargs.sharpness` | `[0.5, 1.5]` | default | defined at `transforms/transforms.py:649` | same |  |
| `dataset.image_transforms.tfs.affine.weight` | `1.0` | default | defined at `transforms/transforms.py:649` | same |  |
| `dataset.image_transforms.tfs.affine.type` | `"RandomAffine"` | default | defined at `transforms/transforms.py:649` | same |  |
| `dataset.image_transforms.tfs.affine.kwargs.degrees` | `[-5.0, 5.0]` | default | defined at `transforms/transforms.py:649` | same |  |
| `dataset.image_transforms.tfs.affine.kwargs.translate` | `[0.05, 0.05]` | default | defined at `transforms/transforms.py:649` | same |  |
| **Not used by a run like this one** | | | | | |
| `dataset.return_uint8` | `false` | default | `False` at `configs/default.py:50` | same | the trainer asks for 8-bit frames itself (`datasets/factory.py:144`) and scales them |
| `dataset.depth_output_unit` | `"mm"` | default | `DEFAULT_DEPTH_UNIT` (`"mm"`) at `configs/default.py:53` | same | no depth camera |
| `policy.use_peft` | `false` | default | `False` at `configs/policies.py:68` | same |  |
| `peft` | `null` | default | `None` at `configs/train.py:169` | same |  |
| `reward_model` | `null` | default | `None` at `configs/train.py:114` | same |  |
| `sample_weighting` | `null` | default | `None` at `configs/train.py:179` | same |  |
| `rename_map` | `{}` | default | empty at `configs/train.py:182` | same |  |
| `env` | `null` | default | `None` at `configs/train.py:112` | same | no simulation environment |
| `env_eval_freq` | `20000` | default | `20_000` at `configs/train.py:144` | same | needs an environment |
| `eval.n_episodes` | `50` | default | `50` at `configs/default.py:108` | same |  |
| `eval.batch_size` | `8` | derived | rule at `configs/default.py:138` | from the machine's processors | 0.7 x the machine's processors, rounded down: 8 on August's 12; for simulation only |
| `eval.use_async_envs` | `true` | default | `True` at `configs/default.py:114` | same |  |
| `eval.recording` | `false` | default | `False` at `configs/default.py:116` | same |  |
| `eval.recording_repo_id` | `null` | default | `None` at `configs/default.py:119` | same |  |
| `eval.recording_private` | `false` | default | `False` at `configs/default.py:121` | same |  |
| `job.target` | `null` | default | `None` at `configs/default.py:234` | same |  |
| `job.image` | `"huggingface/lerobot-gpu:latest"` | default | `"huggingface/lerobot-gpu:latest"` at `configs/default.py:236` | same |  |
| `job.timeout` | `"2d"` | default | `"2d"` at `configs/default.py:240` | same |  |
| `job.detach` | `false` | default | `False` at `configs/default.py:242` | same |  |
| `job.tags` | `[]` | default | empty at `configs/default.py:245` | same |  |

## Facts of the August run that are not in the configuration file

| Fact | August | Read from |
|---|---|---|
| Steps actually run | 90,200, under a ceiling of 100,000; stopped by hand | journal, line 30 and lines 38 to 43 |
| Checkpoint evaluated | `075000`, the lowest evaluation loss | journal, lines 35 and 36 |
| Episodes | 55 recorded, 40,455 frames; 49 trained on, episodes 49 to 54 held out | journal, lines 30 and 32 |
| Passes over the training frames | 80.3 at step 90,200 | journal, line 33 |
| Machine | Colab, A100-SXM4-40GB, 12 processors | journal, line 31 |
| Speed | 4.33 steps a second on average; 0.215 s per update, 0.0065 s waiting for data | journal, lines 31 and 33 |
| Duration | 5.79 hours | journal, line 31 |
| GPU memory | 13.3 GB | journal, line 33 |
| Final training loss | 0.0314 (L1 part 0.0312) | journal, line 34 |
| Evaluation loss | 0.2866 at 5,000 steps, 0.2545 at 20,000, then between 0.245 and 0.262; lowest 0.2455 at 75,000 | journal, line 35 |
| LeRobot on Colab | `lerobot[training]==0.6.1` | August notebook, cell 2 (not in the journal) |
| Parameters | 51,597,190 | *measured here* on 2026-10-05, by loading the August checkpoint with the rig's interpreter; the journal says "52M" (line 30) |

The command line of the August run was not kept: the notebook in the repository still holds an earlier dataset's names.
"Set by: flag" rests on that notebook's training cell and on the configuration file agreeing with it.

## Rules that sit in LeRobot's code, not in the configuration file

Each was read in two copies of LeRobot: the checkout at `6adf515`, and release 0.6.0 as installed in the rig's
environment (`site-packages/lerobot/`). Release 0.6.1, which Colab installs, lies between the two and was not read (see
"What was not checked"). The two copies state every rule below in the same way.

| Rule | At `6adf515`, under `src/lerobot/` | In 0.6.0, under `lerobot/` |
|---|---|---|
| A positive `eval_steps` needs a held-out split. Nothing refuses a split without evaluation passes: its episodes would simply not be trained on | `configs/train.py:318` | `configs/train.py:255` |
| `eval_split` must be at least 0 and below 1 | `configs/default.py:73` | `configs/default.py:55` |
| With `eval_split` at 0.0 the whole dataset is the training set | `datasets/factory.py:196` | `datasets/factory.py:149` |
| A split holds out the last `ceil(n x eval_split)` episodes of each task, in recorded order, without shuffling | `datasets/factory.py:211` | `datasets/factory.py:164` |
| `max_eval_samples` is read only when a held-out set exists | `scripts/lerobot_train.py:356` | `scripts/lerobot_train.py:479` |
| An evaluation pass needs a positive `eval_steps` and a held-out set | `scripts/lerobot_train.py:761` | `scripts/lerobot_train.py:595` |
| A checkpoint is saved every `save_freq` steps and after the last step | `common/train_utils.py:113`, used at `scripts/lerobot_train.py:759` | `scripts/lerobot_train.py:593` |
| A checkpoint's folder is its step, zero-padded to at least six digits: `080000` | `common/train_utils.py:86` | `common/train_utils.py:45` |
| Each checkpoint is uploaded to `checkpoints/<step>/` of the policy repository, and the commit is tagged with the step | `common/train_utils.py:500` | `common/train_utils.py:290` |
| After the last step the policy is uploaded to the top of the repository | `scripts/lerobot_train.py:909` | `scripts/lerobot_train.py:739` |
| A run resumes from the highest-numbered checkpoint on the Hub when `--config_path` names the repository | `configs/train.py:222`, `common/train_utils.py:542`, `utils/hub.py:31` | `configs/train.py:174`, `common/train_utils.py:322`, `utils/hub.py:28` |
| ACT has no learning-rate schedule, and its optimiser is AdamW with the policy's learning rate and weight decay | `policies/act/configuration_act.py:153` and `:159` | the same file and lines |
| Two parameter groups, the backbone and the rest; both at 1e-5 here | `policies/act/modeling_act.py:74` | `policies/act/modeling_act.py:72` |
| The seed is set once, before the dataset and the policy are built | `scripts/lerobot_train.py:431` | `scripts/lerobot_train.py:243` |
| With `cudnn_deterministic` false, cuDNN picks its fastest algorithms; 32-bit matrix products may use TF32 | `scripts/lerobot_train.py:438` | `scripts/lerobot_train.py:251` |
| Batches are drawn by a shuffled, seeded sampler that knows episode boundaries; the last short batch is kept | `scripts/lerobot_train.py:293` and `:347` | `scripts/lerobot_train.py:420` and `:471` |
| Frames are read as 8-bit, whatever `dataset.return_uint8` says, and scaled to the range 0 to 1 | `datasets/factory.py:144`, `scripts/lerobot_train.py:137` | `datasets/factory.py:99`, `scripts/lerobot_train.py:571` |
| With `use_imagenet_stats`, ImageNet's mean and deviation replace the dataset's own for the cameras | `datasets/factory.py:175` | `datasets/factory.py:129` |
| `wandb.run_id` is written into the configuration when the run starts, and a resumed run rejoins that run | `common/wandb_utils.py:112` and `:118` | the same file and lines |
| A policy's weights are loaded without a strict check of their names: a mismatch is logged, not refused | `policies/pretrained.py:159` | `policies/pretrained.py:174` |

Two consequences for this week:

- Because ACT's learning rate is constant, lowering `steps` from 100,000 to 80,000 changes nothing about the steps
  before 80,000. A checkpoint at 40,000 or 60,000 is what it would have been in a longer run, so the protocol's fallback
  to an earlier common step compares like with like.
- Step 80,000 is saved exactly: it is both a multiple of 10,000 and the last step. The eight checkpoints are `010000`
  to `080000`.

## How to check this table against a run

After the first checkpoint of a run (step 10,000) its `train_config.json` is on the Hub. `README.md`, step 5, fetches
the two files and gives these commands with their paths filled in.

The two runs of this week against each other: nothing may differ outside the allowed names.

```bash
"$RIG_PY" training/compare_train_configs.py CLUSTERED/train_config.json SPREAD/train_config.json \
    --must-differ dataset.repo_id --must-differ policy.repo_id --must-differ wandb.run_id
```

A run of this week against August: exactly the four changed values may differ, and they are printed.

```bash
"$RIG_PY" training/compare_train_configs.py AUGUST/train_config.json CLUSTERED/train_config.json \
    --allow dataset.eval_split --allow eval_steps --allow steps --allow save_freq
```

Both commands were run on 2026-10-05 against the real August file and two copies of it edited to this week's plan:
exit status 0, 129 keys compared. If the second command fails on a real run, this table is wrong for that key: correct
the table, and say so in `docs/decisions.md`.

Two differences would be legitimate and need a word in the record, not a correction:

- `eval.batch_size`: it follows the machine's processor count. A different value means the two machines differ.
- `resume`, `checkpoint_path`, `policy.pretrained_path`, `output_dir`: they change when a run was resumed after an
  interruption. The paper reports a resumed run.

The saved configuration files hold the Hub namespace and the W&B account. They stay out of the paper, its supplement
and any released dataset unless those values are removed first.

## What was not checked

- **LeRobot 0.6.1 itself was not read.** It is on this machine only inside the checkout's git history, and the task
  that wrote this page was not to run git. Each rule above was read in the release before 0.6.1 and in a later
  commit; where those two agree, 0.6.1 is assumed to agree. Two things support that. The August file was written by
  0.6.1 and holds every key of the two tables. And the August notebook's notes describe the same split rule and the
  same resume from the Hub; they were written on 2026-08-14, when the checkout stood at the `v0.6.1` tag (*read* from
  the checkout's `.git/logs/HEAD`: at `v0.6.1` from 2026-08-09 to 2026-08-16, at `6adf515` since). One command, run in
  the LeRobot checkout, would settle the rest:
  `git diff v0.6.1 6adf515 --stat -- src/lerobot/configs src/lerobot/scripts/lerobot_train.py src/lerobot/common src/lerobot/datasets/factory.py src/lerobot/policies/act`
- **The versions of PyTorch and the other packages in the August run are not recorded** for the cube run. This week
  each run prints them (`ENV` line), so the two runs can be shown to match each other, not to match August.
- **"This week" is a plan.** Nothing has been trained with these values yet.
