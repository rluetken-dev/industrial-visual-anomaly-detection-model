# Commit Message Guidelines

This project follows the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification to keep the Git history clean, readable, and suitable for release notes.

## Format

```text
<type>(optional scope): <short summary>

(optional body)

(optional footer)
```

## Types

- `feat`: add a new user-facing or model-development capability
- `fix`: correct a bug, invalid result, or unintended behavior
- `docs`: change documentation only
- `style`: change formatting or whitespace without affecting behavior
- `refactor`: restructure code without changing intended behavior
- `perf`: improve runtime speed, memory use, or artifact size
- `test`: add or update automated tests
- `chore`: change dependencies, tooling, configuration, or repository support files
- `revert`: revert an earlier commit

## Project Scopes

Scopes are optional but recommended when they make the affected area clearer.

Common scopes for this project include:

- `datasets`: dataset discovery, structure validation, inventories, checksums, and reports
- `splits`: deterministic fitting, validation, and test partition manifests
- `preprocessing`: image decoding, resizing, cropping, tensors, and normalization
- `features`: pretrained backbone integration, feature maps, and patch embeddings
- `memory`: feature-memory construction, serialization, sampling, and future coreset selection
- `scoring`: nearest-neighbor distances, patch scores, and image-level scores
- `thresholds`: threshold selection, calibration, and decision rules
- `localization`: anomaly maps, resizing, smoothing, and mask generation
- `evaluation`: metrics, qualitative analysis, and experiment results
- `experiments`: experiment configuration, execution, and reproducibility records
- `onnx`: ONNX export, validation, runtime execution, and parity checks
- `artifacts`: model-package schemas, metadata, checksums, and release artifacts
- `model`: broader model-development behavior spanning several model components
- `scripts`: command-line utilities and development scripts
- `config`: project, dataset, model, and environment configuration
- `docs`: project documentation spanning several documents
- `readme`: repository README
- `spec`: project specification
- `architecture`: architecture documentation
- `strategy`: model-development strategy
- `status`: development-status documentation
- `tests`: shared test infrastructure or broad test changes
- `ci`: automated validation and GitHub Actions
- `deps`: dependency updates

Use the most specific useful scope. Avoid combining unrelated scopes in one commit.

## Examples

```text
feat(datasets): add mvtec ad structure validation
```

```text
feat(splits): create deterministic bottle validation split
```

```text
feat(preprocessing): add resnet18 image transformation pipeline
```

```text
feat(features): extract multiscale resnet18 patch embeddings
```

```text
feat(memory): build normal bottle feature memory
```

```text
feat(scoring): add exact nearest-neighbor anomaly scores
```

```text
feat(localization): generate spatial anomaly maps
```

```text
feat(thresholds): derive decision threshold from normal validation scores
```

```text
feat(evaluation): calculate bottle detection metrics
```

```text
feat(onnx): export resnet18 feature extractor
```

```text
fix(splits): reject overlapping manifest entries
```

```text
fix(preprocessing): preserve rgb channel order
```

```text
feat(memory): add deterministic feature-memory sampling
```

```text
test(datasets): cover missing anomaly masks
```

```text
docs(strategy): document bottle baseline decisions
```

```text
chore(deps): pin onnx runtime version
```

## Breaking Changes

Breaking changes must be identified explicitly when they affect a public command, configuration schema, split-manifest schema, artifact format, or integration contract.

Add an exclamation mark after the type or scope:

```text
feat(artifacts)!: replace model metadata schema
```

Alternatively, add a footer:

```text
feat(config): replace preprocessing field names

BREAKING CHANGE: existing experiment configurations must use the new preprocessing keys.
```

Changing an internal experiment before any stable contract exists is not automatically a breaking change. Use the marker only when consumers or versioned files require migration.

## Reverts

Use `revert` when reverting an earlier commit:

```text
revert: revert "feat(features): extract multiscale resnet18 patch embeddings"
```

The body should reference the reverted commit where practical.

## Guidelines

- Use lowercase for the type and scope.
- Write the summary in imperative mood, for example `add`, not `added`.
- Do not end the summary with a period.
- Keep the summary concise, ideally no longer than 72 characters.
- Keep each commit focused on one logical change.
- Use the body to explain motivation, methodology, or important implementation context.
- Use a footer for breaking changes or issue references.
- Do not include secrets, credentials, private hosts, personal dataset paths, or sensitive data.
- Do not commit dataset archives, extracted benchmark images, masks, generated model files, checkpoints, or local experiment output unless their inclusion has been reviewed explicitly.
- Do not describe a model, evaluation, artifact, or feature as complete when implementation or validation is unfinished.
- Do not claim improved accuracy, latency, memory use, or numerical parity without recorded measurements.
- Separate behavior changes from bulk formatting where practical.
- Keep generated artifacts separate from the source change that produces them unless both are intentionally versioned.

## Model and Experiment Commits

Commit messages must distinguish between implementation and evidence.

Use `feat` or `fix` for code that changes model behavior:

```text
feat(scoring): add maximum patch image score
```

Use `docs`, `test`, or an appropriate scope for recorded validation evidence:

```text
docs(evaluation): record initial bottle baseline results
```

Use `perf` only when a measured implementation change improves performance without changing the intended result:

```text
perf(scoring): batch nearest-neighbor distance calculation
```

If an optimization changes numerical behavior or model quality, use `feat` or `fix` and explain the trade-off in the body.

## Dataset Commits

Dataset-related commits should contain scripts, metadata, documentation, checksums, validation summaries, or deterministic manifests rather than restricted source data.

Examples:

```text
feat(datasets): add mvtec ad 2 validation script
```

```text
docs(datasets): record mvtec loco ad inventory
```

```text
feat(splits): add bottle seed 42 split manifest
```

Do not imply that external datasets are part of the repository when only their validation tooling and documentation are versioned.

## Documentation Commits

Use `docs` when only documentation changes:

```text
docs(architecture): update model artifact boundary
```

```text
docs(status): record completed dataset validation
```

Use `chore` when documentation is only one part of broader repository initialization:

```text
chore: initialize project repository
```

## Test Commits

Use the affected capability as the scope when tests belong to a specific area:

```text
test(preprocessing): verify normalized tensor values
```

```text
test(onnx): verify pytorch runtime parity
```

Use `tests` for shared fixtures, helpers, or broad test infrastructure:

```text
test(tests): add temporary dataset fixture
```

## Dependency Commits

Use `chore(deps)` for dependency changes that do not directly implement product behavior:

```text
chore(deps): update pillow package
```

The body should mention compatibility or security motivation when relevant. Dependency updates that affect model outputs must be validated before the commit is described as safe or equivalent.

## Initial Repository Commit

The initial repository commit should use:

```text
chore: initialize project repository
```

Create the initial commit only after:

- repository hygiene checks have passed;
- the virtual environment is ignored;
- dataset directories and archives are excluded;
- generated ONNX and model artifacts are excluded unless intentionally versioned;
- caches and local editor settings are excluded;
- dependency files are present;
- sensitive-value checks have passed;
- source files compile successfully;
- current validation scripts complete successfully for the configured local datasets.
