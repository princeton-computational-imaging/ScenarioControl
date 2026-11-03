# scenariocontrol-video

Front-camera driving video from [ScenarioControl](https://github.com/princeton-computational-imaging/ScenarioControl)
scenes: renders a BEV rollout to a wireframe control video and generates the video with our
fine-tuned NVIDIA Cosmos3-Nano models (`pci-lab/scenariocontrol-videogen` on the Hugging Face Hub).

```python
from scenariocontrol_video import ScenarioControlVideoPipeline

pipe = ScenarioControlVideoPipeline.from_hub("pci-lab/scenariocontrol-videogen", variant="prompt-only")
pipe(bev="frames/", prompt="A rainy evening drive through a city.").save("out.mp4")
```

Install, usage, training and licensing: see the "Video Generation" section of the
[ScenarioControl README](https://github.com/princeton-computational-imaging/ScenarioControl#video-generation).
The weights are for non-commercial use only (nuPlan); see `NOTICE`.
