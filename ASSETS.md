# Research and media provenance

All research assets used in this redesign were supplied in the user's local workspace. No illustrative screenshots or model results were fabricated.

## Reported results and author metadata

Source: `../emnlp26_DART/EMNLP26_RLforCUA/EMNLP26/main_camera_ready.tex`.

- Manuscript title and author order/affiliations: title and author block.
- Overall and ten-domain 30/100-step results: `tab:comp_use_res`.
- Throughput (22.6 → 43.6 actions/min), environment utilization (12.2 → 67.7%), and rollout GPU utilization (29.6 → 46.7%): `tab:efficiency`.
- Framework/curation descriptions: abstract and method sections.

Percentage-point changes are differences between DART and its base model **at the same interaction budget**. The 1.9×, 5.5×, and 1.6× efficiency ratios retain the paper's rounded values. No present-day state-of-the-art claim is made.

The citation retains the original arXiv title, identifier, year, and author list from the code repository, while the research section shows the newer supplied manuscript title. No unverified conference acceptance or publication date is asserted.

## Paper figure

`public/research/architecture.png` comes from `../emnlp26_DART/EMNLP26_RLforCUA/figure/Arch_detail.png`. Only transparent exterior padding was cropped; labels and diagram content are unchanged. The interactive SVG is a simplified module-connectivity illustration, not a measured system monitor.

The older files under `public/stats/`, `public/icon.png`, and `public/poster.png` are retained so previously shared asset URLs are not broken. They are not used as updated result charts.

## Trajectory replays

Source: `../iclr-dart-page_1129/trajectories/`, using each selected folder's `final_messages.json`, `reward.txt`, and original screenshots. The featured examples were selected from 147 successful archived runs for visible photo and presentation transformations.

| Example                 | Source trajectory ID                                                 | Recorded reward |
| ----------------------- | -------------------------------------------------------------------- | --------------- |
| Photo color enhancement | `554785e9-4523-4e7a-b8e1-8016f565f56a_trace-2f0c6fab770c-1758785787` | 1.0             |
| Presentation typography | `5c1a6c3d-c1b3-47cb-9b01-8d1b7544ffa1_trace-02fb6b92f81c-1758785787` | 1.0             |
| Husky photo resizing    | `d16c99dc-2a1e-46f2-b350-d97c86c85c15_trace-0d523480f09f-1758785787` | 1.0             |

Transformations:

1. Pair each assistant action and its original reasoning with the preceding screenshot.
2. Resize screenshots to at most 1440 × 810 and encode WebP at quality 87.
3. Preserve task instructions, action text, reward, OS, and source IDs in `manifest.json`; omit the unrelated system prompt.
4. Normalize recorded pixel coordinates to percentages for responsive action markers, including both endpoints of drag actions.
5. Render a 1280 × 860 branded frame with the original screenshot, action label, step number, and source/timing note. Large pulse rings, numbered labels, and drag paths annotate recorded coordinates. Keyboard actions have a dedicated badge. Only these annotations animate; no intermediate application state is invented.
6. Encode silent H.264/yuv420p MP4 at 12 fps, 2 seconds per action, with `faststart`. Generate a 896 × 602 GIF of the full photo enhancement example at the same 2 seconds per action, with four annotation frames per step. Posters and task thumbnails use the final recorded observation paired with the terminal action.

The earlier GIMP export, Calc PDF export, and VS Code file-creation media remain at their original URLs. They came from `../iclr-dart-page_1129/public/data/` and are no longer the featured examples.

All assistant actions, including the terminal `finished` action, are retained. The trailing post-termination screenshot is not a new action. These are earlier archive examples; they do not independently establish or reproduce the camera-ready aggregate scores. Success labels reproduce the archive reward rather than asserting a new evaluation.

## EMNLP 2026 conference identity

`public/brand/emnlp-2026.png` is the official conference wordmark from
[the EMNLP 2026 website](https://2026.emnlp.org/assets/images/logos/emnlp_2026_logo_v1.png).
The source was resized proportionally from 8703 × 1516 to 1200 × 209 for web display;
its original colors and artwork are preserved. The logo links to the conference website
and is displayed alongside the author credits near the top of the page.

## Homepage service icons

- `public/brand/huggingface.svg`: [Hugging Face's hosted logo](https://huggingface.co/front/assets/huggingface_logo-noborder.svg), displayed with its original colors.
- `public/brand/github.svg`: [GitHub's Primer Octicons mark](https://github.com/primer/octicons/blob/main/icons/mark-github-16.svg), displayed in a light monochrome on the dark background. The upstream MIT license is included in `public/brand/octicons-LICENSE.txt`.

These assets are served locally. Homepage buttons link directly to the user-specified GitHub repository and `PengxiangLi/dart-gui-7b` model repository.
