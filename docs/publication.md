# Local release and GitHub publication

This repository is prepared locally; the owner publishes it. The public tree includes authored English code, numerical evidence, executed notebooks, documentation, portable policies and release metadata. Large training checkpoints, environments, partial/superseded fits and original course material are excluded.

Choose your GitHub repository name and create an empty repository in your account. From this project directory, after reviewing the release validation:

```bash
git remote add origin git@github.com:YOUR_ACCOUNT/sequential-decision-lab.git
git push -u origin main
git push origin v1.1.0
```

The placeholder account must be replaced by your own. No remote or push is configured automatically. Use GitHub Releases for optional large fitted control/neural weights; publish their recorded SHA256 files with them. The finite-state policies, small MountainCar weights and new MiniGrid/MuJoCo actors already work from the public tree.

A source archive must exclude `legacy/originals`, `.local`, environments, caches, smoke outputs and private checkpoint directories. A wheel contains the authored Python API and default configs; the source archive carries the research evidence and notebooks. See the release checklist for actual local checks and counts.

## Publish the interactive dashboard

The README displays a real-browser dashboard screenshot and links to navigation and mobile previews. To make the interactive page available through a website, enable GitHub Pages after pushing the public repository:

1. Open the repository's **Settings → Pages**.
2. Under **Build and deployment**, choose **Deploy from a branch**.
3. Select **main** and **/(root)**, then save.
4. Wait for the Pages deployment to finish. The dashboard URL is `https://YOUR_ACCOUNT.github.io/YOUR_REPOSITORY/docs/simulation_dashboard.html`.
5. Replace the README's local dashboard link with that deployed URL and add it to the repository's **About → Website** field.

Use the repository root as the publishing source because the page loads animations from `assets/`, outside `docs/`. The selected account and repository name determine the final URL. See [GitHub's publishing-source instructions](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site).

Visitors use their browser to select tasks and presets, watch prerecorded rollouts, and inspect measured learning curves and comparisons. Reproducing training or replaying an exported policy requires the Python environment described in the README. Pages is not enabled or deployed by the local preparation.
