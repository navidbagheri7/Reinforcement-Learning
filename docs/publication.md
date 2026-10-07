# Local release and GitHub publication

This repository is prepared locally; the owner publishes it. The public tree includes authored English code, numerical evidence, executed notebooks, documentation, portable policies and release metadata. Large training checkpoints, environments, partial/superseded fits and original course material are excluded.

Choose your GitHub repository name and create an empty repository in your account. From this project directory, after reviewing the release validation:

```bash
git remote add origin git@github.com:YOUR_ACCOUNT/sequential-decision-lab.git
git push -u origin main
git push origin v1.0.0
```

The placeholder account must be replaced by your own. No remote or push is configured automatically. Use GitHub Releases for optional large fitted control/neural weights; publish their recorded SHA256 files with them. The finite-state policies and small MountainCar weights already work from the public tree.

A source archive must exclude `legacy/originals`, `.local`, environments, caches, smoke outputs and private checkpoint directories. A wheel contains the authored Python API and default configs; the source archive carries the research evidence and notebooks. See the release checklist for actual local checks and counts.
