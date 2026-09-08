# PlannerForge

**LLM Agents for Scenario-Based Testing of Motion Planners in Autonomous Driving**

Accepted to the **EMNLP 2026 Main Conference** (Budapest).

Yuan Gao<sup>1</sup>, Sebastian Müller<sup>1</sup>, Mattia Piccinini<sup>1</sup>,
Marc Kaufeld<sup>1</sup>, Yuchen Zhang<sup>1</sup>, Finn Rasmus Schäfer<sup>1</sup>,
Qunying Song<sup>2</sup>, Johannes Betz<sup>1</sup>

<sup>1</sup>Professorship of Autonomous Vehicle Systems, TUM School of Engineering and Design,
Technical University of Munich, 85748 Garching, Germany;
Munich Institute of Robotics and Machine Intelligence (MIRMI)
<sup>2</sup>University College London, London, United Kingdom

PlannerForge is an LLM-agent framework that unifies all eight stages of
scenario-based testing for autonomous-driving motion planners under a single
chatbot-driven interface: the six stages of the Riedmaier taxonomy plus two
LLM-era stages, ADS Enhancement and ADS Benchmarking.

## This repository

This repository hosts the **project homepage** and the paper. The code release is
in preparation and will be added here.

| Path | Contents |
|------|----------|
| `index.html` | Project homepage — framework overview, results, demo videos, interactive prompt explorer |
| `PlannerForge_EMNLP.pdf` | Camera-ready paper (EMNLP 2026) |
| `assets/` | Figures and the full prompt corpus (`prompts.json`) |
| `*.mp4` | Demo videos for the pipeline stages |

The homepage is a static page. Enable GitHub Pages on the default branch to serve
it, or open `index.html` locally.

## Demos

| Video | Stage |
|-------|-------|
| `Generation_text.mp4` | Scenario generation from natural language |
| `Generation_map.mp4` | Scenario generation from a map input |
| `Selection.mp4` | Scenario selection |
| `Scenario_Modification.mp4` | Scenario modification |
| `Test_Execution.mp4` | Test execution |
| `ADS_Assessment.mp4` | ADS assessment |
| `ADS_Enhancement.mp4` | ADS enhancement |

## Citation

```bibtex
@inproceedings{gao2026plannerforge,
  title     = {PlannerForge: LLM Agents for Scenario-Based Testing of
               Motion Planners in Autonomous Driving},
  author    = {Gao, Yuan and M{\"u}ller, Sebastian and Piccinini, Mattia and
               Kaufeld, Marc and Zhang, Yuchen and Sch{\"a}fer, Finn Rasmus and
               Song, Qunying and Betz, Johannes},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in
               Natural Language Processing (EMNLP)},
  year      = {2026},
  address   = {Budapest, Hungary},
  note      = {Code and data:
               https://github.com/TUM-AVS/PlannerForge}
}
```

## License

Released under the terms in [LICENSE](LICENSE).
