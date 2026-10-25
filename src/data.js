export const modules = [
  {
    name: "Environment cluster",
    type: "INTERACT",
    icon: "▦",
    summary: "A desktop for every rollout.",
    body: "Independent environments execute mouse and keyboard actions. As soon as one trajectory finishes, that environment starts the next one—without waiting for the batch.",
    tag: "Rollout-wise scheduling",
    metric: "5.5×",
    metricLabel: "environment utilization",
  },
  {
    name: "Rollout service",
    type: "GENERATE",
    icon: "↗",
    summary: "Keep the policy moving.",
    body: "Inference workers generate actions independently. Per-worker model synchronization lets other workers keep serving requests while one worker receives new parameters.",
    tag: "Per-worker synchronization",
    metric: "1.6×",
    metricLabel: "GPU utilization",
  },
  {
    name: "Data manager",
    type: "CURATE",
    icon: "≋",
    summary: "Learn from the right experience.",
    body: "A shared data layer connects collection and learning. Adaptive curation operates across tasks, trajectories, steps, and tokens to make long-horizon, sparse-reward experience useful.",
    tag: "Four levels of data curation",
    metric: "4",
    metricLabel: "levels of adaptive curation",
  },
  {
    name: "Trainer",
    type: "LEARN",
    icon: "⟳",
    summary: "Train without the waiting.",
    body: "The trainer consumes curated experience asynchronously and publishes updated checkpoints. Truncated importance sampling helps account for the distribution gap between rollout and training engines.",
    tag: "Asynchronous policy updates",
    metric: "1.9×",
    metricLabel: "training throughput",
  },
];
export const categories = [
  "Chrome",
  "GIMP",
  "Calc",
  "Impress",
  "Writer",
  "Multi-app",
  "OS",
  "Thunderbird",
  "VLC",
  "VS Code",
];
// Source: EMNLP26/main_camera_ready.tex, tab:comp_use_res. Compare equal step budgets.
export const scores = {
  100: {
    baseline: [38.34, 51.92, 9.57, 38.21, 39.13, 8.94, 31.25, 40, 22.44, 47.83],
    ours: [
      53.09, 77.79, 20.62, 50.27, 61.82, 17.61, 63.63, 60.93, 41.03, 70.84,
    ],
    total: [27.52, 43.28],
  },
  30: {
    baseline: [23.91, 50, 6.38, 27.45, 34.77, 9.54, 43.48, 53.33, 17.65, 40.91],
    ours: [
      51.99, 76.99, 19.12, 48.67, 60.82, 16.53, 62.43, 60.03, 39.23, 69.54,
    ],
    total: [24.17, 42.05],
  },
};
export const authors = [
  ["Pengxiang Li", "1,2*", "https://pengxiang-li.github.io"],
  ["Zechen Hu", "3*"],
  ["Zirui Shang", "1,2*"],
  [
    "Jingrong Wu",
    "3*",
    "https://scholar.google.com/citations?user=xr7kNGEAAAAJ&hl=en",
  ],
  ["Yang Liu", "2"],
  ["Hui Liu", "3"],
  ["Zhi Gao", "1,2,5†", "https://zhigao2017.github.io/"],
  ["Chenrui Shi", "1,2"],
  ["Bofei Zhang", "2", "https://bofei5675.github.io/"],
  ["Zihao Zhang", "3"],
  ["Xiaochuan Shi", "3"],
  ["Zedong Yu", "2,4"],
  ["Yuwei Wu", "1,5", "https://wu-yuwei-bit.github.io/"],
  ["Xinxiao Wu", "1,5", "https://wuxinxiao.github.io/"],
  ["Yunde Jia", "5†"],
  ["Liuyu Xiang", "4"],
  ["Zhaofeng He", "4"],
  ["Qing Li", "2†", "https://liqing.io/"],
];
export const bibtex = `@article{li2025dart,
  title = {Efficient Multi-turn RL for GUI Agents via Decoupled Training and Adaptive Data Curation},
  author = {Li, Pengxiang and Hu, Zechen and Shang, Zirui and Wu, Jingrong and Liu, Yang and Liu, Hui and Gao, Zhi and Shi, Chenrui and Zhang, Bofei and Zhang, Zihao and Shi, Xiaochuan and Yu, Zedong and Wu, Yuwei and Wu, Xinxiao and Jia, Yunde and Xiang, Liuyu and He, Zhaofeng and Li, Qing},
  journal = {arXiv preprint arXiv:2509.23866},
  year = {2025},
  url = {https://arxiv.org/abs/2509.23866}
}`;
