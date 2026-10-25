<script setup>
import { computed, nextTick, onMounted, onUnmounted, ref } from "vue";
import { modules, categories, scores, authors, bibtex } from "../data.js";

const asset = (path) => `${import.meta.env.BASE_URL}${path}`;
const demos = ref([]);
const loading = ref(true);
const loadError = ref(false);
const selected = ref(0);
const stepIndex = ref(0);
const mode = ref("video");
const video = ref(null);
const videoError = ref(false);
const playing = ref(false);
const currentTime = ref(0);
const mobileNav = ref(false);
const budget = ref(100);
const activeModule = ref(0);
const copyStatus = ref("Copy BibTeX");
const activeSection = ref("overview");
const demo = computed(() => demos.value[selected.value]);
const step = computed(() => demo.value?.steps[stepIndex.value]);
const currentVideoStep = computed(() =>
  Math.min(
    Math.floor(currentTime.value / 2) + 1,
    demo.value?.steps.length || 1,
  ),
);
const actionName = computed(() => {
  const name = step.value?.action.split("(")[0];
  return (
    {
      click: "CLICK",
      left_double: "DOUBLE-CLICK",
      right_single: "RIGHT-CLICK",
      drag: "DRAG",
      scroll: "SCROLL",
    }[name] || "ACTION"
  );
});
const results = computed(() => scores[budget.value]);
const gain = computed(() =>
  (results.value.total[1] - results.value.total[0]).toFixed(2),
);
let observer;
let copyTimer;

async function loadDemos() {
  loading.value = true;
  loadError.value = false;
  try {
    const response = await fetch(asset("demos/manifest.json"));
    if (!response.ok) throw new Error("Could not load demo data");
    const data = await response.json();
    if (
      !Array.isArray(data) ||
      !data.length ||
      data.some((item) => !item.steps?.length)
    )
      throw new Error("Invalid demo data");
    demos.value = data;
  } catch {
    loadError.value = true;
  } finally {
    loading.value = false;
  }
}
function selectDemo(index) {
  video.value?.pause();
  selected.value = index;
  stepIndex.value = 0;
  currentTime.value = 0;
  videoError.value = false;
  playing.value = false;
}
async function watchDemo() {
  mode.value = "video";
  document.getElementById("demo")?.scrollIntoView({
    behavior: matchMedia("(prefers-reduced-motion: reduce)").matches
      ? "instant"
      : "smooth",
  });
  await nextTick();
  try {
    await video.value?.play();
  } catch {
    /* Native controls remain available. */
  }
}
function setMode(value) {
  video.value?.pause();
  mode.value = value;
  playing.value = false;
  if (value === "video") currentTime.value = 0;
}
function moveStep(delta) {
  stepIndex.value = Math.min(
    Math.max(0, stepIndex.value + delta),
    demo.value.steps.length - 1,
  );
}
async function copyCitation() {
  try {
    await navigator.clipboard.writeText(bibtex);
    copyStatus.value = "Copied!";
  } catch {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(document.getElementById("citation-code"));
    selection.removeAllRanges();
    selection.addRange(range);
    copyStatus.value = "Selected — press Ctrl/Cmd+C";
  }
  clearTimeout(copyTimer);
  copyTimer = setTimeout(() => {
    copyStatus.value = "Copy BibTeX";
  }, 3500);
}
onMounted(() => {
  loadDemos();
  observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries)
        if (entry.isIntersecting) activeSection.value = entry.target.id;
    },
    { rootMargin: "-15% 0px -60% 0px" },
  );
  document
    .querySelectorAll("main > section[id]")
    .forEach((section) => observer.observe(section));
});
onUnmounted(() => {
  observer?.disconnect();
  clearTimeout(copyTimer);
});
</script>

<template>
  <a class="skip-link" href="#main">Skip to content</a>
  <header class="site-header">
    <div class="nav-wrap">
      <a
        class="brand"
        href="#overview"
        aria-label="DART-GUI home"
        @click="mobileNav = false"
        ><svg
          class="brand-mark"
          viewBox="0 0 32 32"
          fill="none"
          aria-hidden="true"
        >
          <path d="M4 5h13l11 11-11 11H4l11-11L4 5Z" fill="currentColor" />
          <path d="m5 12 4 4-4 4" stroke="currentColor" stroke-width="2" /></svg
        >DART<span class="brand-slash">/</span
        ><span class="brand-gui">GUI</span></a
      >
      <button
        class="menu-toggle"
        :aria-expanded="mobileNav"
        aria-controls="main-nav"
        @click="mobileNav = !mobileNav"
      >
        {{ mobileNav ? "Close −" : "Menu +" }}
      </button>
      <nav
        id="main-nav"
        :class="{ open: mobileNav }"
        aria-label="Main navigation"
      >
        <a
          v-for="[id, label] in [
            ['overview', 'Overview'],
            ['demo', 'In action'],
            ['method', 'Method'],
            ['results', 'Results'],
          ]"
          :key="id"
          :href="`#${id}`"
          :class="{ active: activeSection === id }"
          :aria-current="activeSection === id ? 'location' : undefined"
          @click="mobileNav = false"
          >{{ label }}</a
        ><a
          class="nav-code"
          href="https://github.com/Computer-use-agents/dart-gui"
          target="_blank"
          rel="noopener noreferrer"
          >View on GitHub <span aria-hidden="true">↗</span></a
        >
      </nav>
    </div>
  </header>

  <main id="main">
    <section id="overview" class="hero section-wrap">
      <div class="hero-copy">
        <div class="eyebrow">
          <span class="status-dot"></span> OPEN-SOURCE COMPUTER-USE RESEARCH
        </div>
        <h1>Real desktops.<br />Real <span>learning.</span></h1>
        <p class="hero-intro">
          <strong>Meet DART-GUI.</strong> A fully decoupled reinforcement
          learning framework that turns desktop interactions into better GUI
          agents.
        </p>
        <div class="hero-actions">
          <button class="button primary" @click="watchDemo">
            <span class="play-icon" aria-hidden="true">▶</span> Watch it work
            <span aria-hidden="true">↗</span></button
          ><a
            class="button secondary"
            href="https://arxiv.org/abs/2509.23866"
            target="_blank"
            rel="noopener noreferrer"
            >Read the paper <span aria-hidden="true">↗</span></a
          >
          <a
            class="button secondary"
            href="https://huggingface.co/PengxiangLi/dart-gui-7b"
            target="_blank"
            rel="noopener noreferrer"
          >
            <img
              class="service-icon"
              :src="asset('brand/huggingface.svg')"
              alt=""
              width="20"
              height="20"
            />
            Hugging Face <span aria-hidden="true">↗</span>
          </a>
          <a
            class="button secondary"
            href="https://github.com/Computer-use-agents/dart-gui"
            target="_blank"
            rel="noopener noreferrer"
          >
            <img
              class="service-icon github-icon"
              :src="asset('brand/github.svg')"
              alt=""
              width="20"
              height="20"
            />
            GitHub <span aria-hidden="true">↗</span>
          </a>
        </div>
        <div class="hero-footnote">
          <span class="mini-brackets">[ 7B ]</span
          ><span
            >Open weights. Real environments.<br />Learning that keeps
            moving.</span
          >
        </div>
      </div>
      <div
        class="hero-art"
        aria-label="A recorded GIMP desktop connected to the DART asynchronous learning loop"
      >
        <div class="orbit orbit-one"></div>
        <div class="orbit orbit-two"></div>
        <div class="art-coordinate coordinate-top">
          FIG. 01 / THE LEARNING LOOP
        </div>
        <div class="desktop-window">
          <div class="window-top">
            <span class="window-dots"><i></i><i></i><i></i></span
            ><span>agent@desktop:~</span
            ><span class="window-tag">DART-GUI-7B</span>
          </div>
          <div class="desktop-screen">
            <img
              :src="asset('demos/vibrance/06.webp')"
              alt="Recorded GIMP desktop showing a photo with enhanced colors"
              fetchpriority="high"
              width="1440"
              height="810"
            /><span class="screen-crosshair" aria-hidden="true">＋</span>
          </div>
          <div class="desktop-bottom">
            <span><span class="status-dot"></span> OBSERVE → ACT → LEARN</span
            ><span>Ubuntu 22.04</span>
          </div>
        </div>
        <div class="floating-label label-action">
          <span class="label-icon">↗</span>
          <div>
            <span class="mono muted">FEATURED TASK</span
            ><strong>Enhance photo colors</strong>
          </div>
          <span class="tiny-arrow">↗</span>
        </div>
        <div class="floating-label label-learning">
          <span class="loop-icon">⟳</span>
          <div>
            <strong>Always in motion.</strong
            ><span class="mono muted">ASYNCHRONOUS BY DESIGN</span>
          </div>
        </div>
        <div class="art-coordinate coordinate-bottom">
          <span class="cross">+</span> ENVIRONMENT &nbsp; / &nbsp; POLICY &nbsp;
          / &nbsp; EXPERIENCE
        </div>
      </div>
      <div
        class="hero-authors"
        aria-label="Conference, authors, and affiliations"
      >
        <a
          class="conference-logo"
          href="https://2026.emnlp.org/"
          target="_blank"
          rel="noopener noreferrer"
          aria-label="EMNLP 2026 conference website"
        >
          <span class="mono muted">EMNLP 2026</span>
          <span class="conference-logo-image"
            ><img
              :src="asset('brand/emnlp-2026.png')"
              alt="EMNLP 2026 Budapest"
              width="1200"
              height="209"
          /></span>
          <span class="conference-link"
            >Conference website <span aria-hidden="true">↗</span></span
          >
        </a>
        <div class="author-credit">
          <h2 class="hero-paper-title">
            Fully Decoupled Reinforcement Learning with Adaptive Data Curation
            for GUI Agents
          </h2>
          <span class="mono muted">THE AUTHORS</span>
          <div class="authors">
            <span v-for="[name, affiliation, url] in authors" :key="name"
              ><a
                v-if="url"
                :href="url"
                target="_blank"
                rel="noopener noreferrer"
                >{{ name }}</a
              ><span v-else>{{ name }}</span
              ><sup>{{ affiliation }}</sup></span
            >
          </div>
          <div class="affiliations">
            <span><sup>1</sup> Beijing Institute of Technology</span
            ><span><sup>2</sup> BIGAI</span><span><sup>3</sup> DataCanvas</span
            ><span
              ><sup>4</sup> Beijing University of Posts and
              Telecommunications</span
            ><span><sup>5</sup> Shenzhen MSU-BIT University</span>
          </div>
          <p class="author-note">
            * Equal contribution &nbsp; · &nbsp; † Corresponding authors
          </p>
        </div>
      </div>
      <div class="hero-metrics">
        <div>
          <span class="metric-kicker">01 / CAPABILITY</span
          ><strong>43.28<span>%</span></strong>
          <p>OSWorld success rate <small>100 steps</small></p>
        </div>
        <div>
          <span class="metric-kicker">02 / IMPROVEMENT</span
          ><strong>+15.76<span>pp</span></strong>
          <p>Over the base model <small>same budget</small></p>
        </div>
        <div>
          <span class="metric-kicker">03 / THROUGHPUT</span
          ><strong>1.9<span>×</span></strong>
          <p>Training throughput</p>
        </div>
        <div>
          <span class="metric-kicker">04 / UTILIZATION</span
          ><strong>5.5<span>×</span></strong>
          <p>Environment utilization</p>
        </div>
      </div>
    </section>

    <section id="demo" class="demo-section section-wrap section-space">
      <div class="section-heading">
        <div>
          <div class="eyebrow muted">01 / DART IN ACTION</div>
          <h2>
            One agent.<br /><span class="muted">A world of interfaces.</span>
          </h2>
        </div>
        <p>
          No special-purpose APIs. Just pixels, mouse,<br
            class="desktop-only"
          />
          keyboard—and a policy that learns by doing.
        </p>
      </div>
      <div v-if="loading" class="demo-loading" role="status">
        Loading recorded trajectories…
      </div>
      <div v-else-if="loadError" class="demo-loading" role="alert">
        <p>The demo data could not be loaded.</p>
        <button class="button secondary" @click="loadDemos">Try again ↻</button>
      </div>
      <div v-else-if="demo" class="demo-shell">
        <div class="demo-toolbar">
          <div class="terminal-title">
            <span class="status-dot"></span> THE AGENT'S DESKTOP
            <span class="toolbar-app">/ {{ demo.app }}</span>
          </div>
          <div class="segmented" aria-label="Demo view">
            <button
              :class="{ selected: mode === 'video' }"
              :aria-pressed="mode === 'video'"
              @click="setMode('video')"
            >
              Video replay</button
            ><button
              :class="{ selected: mode === 'steps' }"
              :aria-pressed="mode === 'steps'"
              @click="setMode('steps')"
            >
              Step explorer
            </button>
          </div>
        </div>
        <div class="demo-grid">
          <div class="demo-stage">
            <div v-if="mode === 'video'" class="video-wrap">
              <video
                :key="demo.id"
                ref="video"
                :src="asset(demo.video)"
                :poster="asset(demo.poster)"
                controls
                playsinline
                preload="metadata"
                :aria-label="`${demo.app}: ${demo.description} Recorded screenshot replay.`"
                @timeupdate="currentTime = $event.target.currentTime"
                @play="playing = true"
                @pause="playing = false"
                @ended="playing = false"
                @error="videoError = true"
              >
                Your browser does not support video. Download the MP4 using the
                link below.
              </video>
              <div v-if="videoError" class="video-error" role="alert">
                Video playback is unavailable.
                <a :href="asset(demo.video)" download>Download the MP4</a> or
                use the step explorer.
              </div>
            </div>
            <div
              v-else
              class="step-view"
              @keydown.left.prevent="moveStep(-1)"
              @keydown.right.prevent="moveStep(1)"
            >
              <div class="step-screenshot">
                <img
                  :src="asset(step.image)"
                  :alt="`${demo.app}, recorded observation ${stepIndex + 1} of ${demo.steps.length}`"
                  width="1440"
                  height="810"
                />
                <svg
                  v-if="step.endPoint"
                  class="drag-path"
                  viewBox="0 0 1440 810"
                  preserveAspectRatio="none"
                  aria-hidden="true"
                >
                  <defs>
                    <marker
                      id="drag-arrow"
                      viewBox="0 0 10 10"
                      refX="8"
                      refY="5"
                      markerWidth="6"
                      markerHeight="6"
                      orient="auto-start-reverse"
                    >
                      <path d="M0 0L10 5L0 10Z" fill="#c4f378" />
                    </marker>
                  </defs>
                  <line
                    :x1="step.point[0] * 14.4"
                    :y1="step.point[1] * 8.1"
                    :x2="step.endPoint[0] * 14.4"
                    :y2="step.endPoint[1] * 8.1"
                    stroke="#0b100e"
                    stroke-width="10"
                  />
                  <line
                    :x1="step.point[0] * 14.4"
                    :y1="step.point[1] * 8.1"
                    :x2="step.endPoint[0] * 14.4"
                    :y2="step.endPoint[1] * 8.1"
                    stroke="#c4f378"
                    stroke-width="4"
                    marker-end="url(#drag-arrow)"
                  />
                </svg>
                <span
                  v-if="step.point"
                  :key="`${demo.id}-${stepIndex}`"
                  class="action-marker"
                  :style="{
                    left: `${step.point[0]}%`,
                    top: `${step.point[1]}%`,
                  }"
                  aria-label="Recorded action location"
                  ><span
                    class="action-marker-label"
                    :class="{
                      'label-left': step.point[0] > 65,
                      'label-above': step.endPoint,
                    }"
                    aria-hidden="true"
                    >{{ String(stepIndex + 1).padStart(2, "0") }} ·
                    {{ actionName }}</span
                  ></span
                >
              </div>
              <div class="step-controls">
                <button
                  class="icon-button"
                  :disabled="stepIndex === 0"
                  aria-label="Previous step"
                  @click="moveStep(-1)"
                >
                  ←</button
                ><label class="sr-only" for="step-slider">Trajectory step</label
                ><input
                  id="step-slider"
                  v-model.number="stepIndex"
                  type="range"
                  min="0"
                  :max="demo.steps.length - 1"
                  :aria-valuetext="`Step ${stepIndex + 1} of ${demo.steps.length}`"
                /><span class="mono"
                  >{{ String(stepIndex + 1).padStart(2, "0") }} /
                  {{ demo.steps.length }}</span
                ><button
                  class="icon-button"
                  :disabled="stepIndex === demo.steps.length - 1"
                  aria-label="Next step"
                  @click="moveStep(1)"
                >
                  →
                </button>
              </div>
            </div>
            <div class="stage-status">
              <span
                ><i :class="{ pulse: playing }"></i>
                {{
                  mode === "video"
                    ? playing
                      ? "REPLAYING"
                      : "RECORDED TRAJECTORY"
                    : "OBSERVATION + ACTION"
                }}</span
              ><span>{{
                mode === "video"
                  ? `STEP ${String(currentVideoStep).padStart(2, "0")} / ${demo.steps.length}`
                  : "← → TO EXPLORE"
              }}</span>
            </div>
          </div>
          <aside class="demo-sidebar" aria-label="Choose a recorded task">
            <div class="sidebar-label mono muted">CHOOSE A TASK</div>
            <button
              v-for="(item, index) in demos"
              :key="item.id"
              class="task-option"
              :class="{ active: selected === index }"
              :aria-pressed="selected === index"
              @click="selectDemo(index)"
            >
              <span class="task-preview">
                <img
                  :src="asset(item.preview)"
                  alt=""
                  width="1440"
                  height="810"
                  loading="lazy"
                />
                <span class="task-number mono">{{ item.number }}</span>
              </span>
              <span class="task-text"
                ><strong>{{ item.title }}</strong
                ><span>{{ item.description }}</span
                ><small
                  >{{ item.app }} · {{ item.steps.length }} steps
                  <span aria-hidden="true">·</span> {{ item.steps.length * 2 }}s
                  replay</small
                ></span
              ><span class="task-arrow" aria-hidden="true">↗</span>
            </button>
            <div class="recorded-result">
              <span class="check-mark">✓</span>
              <div>
                <strong>Successful recorded run</strong>
                <p>Evaluation reward: {{ demo.reward.toFixed(1) }}</p>
              </div>
            </div>
            <a
              class="text-link download-link"
              :href="asset(demo.video)"
              download
              >Download this demo <span aria-hidden="true">↓</span></a
            >
          </aside>
        </div>
        <div v-if="mode === 'steps'" class="step-details" aria-live="polite">
          <div>
            <span class="mono muted"
              >RECORDED REASONING / {{ stepIndex + 1 }}</span
            >
            <p>{{ step.thought }}</p>
          </div>
          <div>
            <span class="mono muted">ACTION</span><code>{{ step.action }}</code>
          </div>
        </div>
        <div class="demo-caption">
          <p>
            <span class="mono">SOURCE NOTE</span> Replays assembled from
            archived OSWorld screenshots and actions. Each step is shown for 2
            seconds; original execution timing is condensed.
          </p>
          <a :href="asset('demos/dart-gui-demo.gif')" download
            >Get the GIF ↗</a
          >
        </div>
      </div>
      <div class="demo-under">
        <span>See it. Act on it. Learn from it.</span
        ><span class="mono muted">REAL TRAJECTORIES / OPEN RESEARCH</span>
      </div>
    </section>

    <section id="method" class="method-section section-space">
      <div class="section-wrap">
        <div class="section-heading">
          <div>
            <div class="eyebrow muted">02 / THE FRAMEWORK</div>
            <h2>
              Decouple the system.<br /><span class="accent"
                >Unlock the learning.</span
              >
            </h2>
          </div>
          <p>
            Four independent modules. One continuous loop.<br
              class="desktop-only"
            />
            Less waiting, more useful experience.
          </p>
        </div>
        <div class="architecture-layout">
          <div class="architecture-canvas">
            <div class="canvas-top mono muted">
              <span>DART / SYSTEM ARCHITECTURE</span><span>ASYNC</span>
            </div>
            <svg
              class="connector-svg"
              viewBox="0 0 600 390"
              preserveAspectRatio="none"
              aria-hidden="true"
            >
              <path
                d="M145 110H455V280H145M145 110L455 280M145 280L455 110"
                fill="none"
                stroke="var(--line-strong)"
                stroke-width="1.5"
              />
              <path
                class="flow-path"
                d="M145 110H455V280H145M145 280L455 110"
                fill="none"
                stroke="var(--accent)"
                stroke-width="2"
                stroke-dasharray="8 100"
              />
            </svg>
            <div class="architecture-nodes">
              <button
                v-for="(item, index) in modules"
                :key="item.name"
                :class="[
                  'architecture-node',
                  `node-${index}`,
                  { active: activeModule === index },
                ]"
                :aria-pressed="activeModule === index"
                @click="activeModule = index"
              >
                <span class="node-top"
                  ><span class="node-icon">{{ item.icon }}</span
                  ><span class="mono">0{{ index + 1 }}</span></span
                ><strong>{{ item.name }}</strong
                ><span class="mono node-type">{{ item.type }}</span>
              </button>
            </div>
            <div class="loop-center">
              <span>↻</span><span class="mono">CONTINUOUS<br />LEARNING</span>
            </div>
            <div class="canvas-bottom mono muted">
              SELECT A MODULE TO EXPLORE <span>↗</span>
            </div>
          </div>
          <div class="module-detail" aria-live="polite">
            <span class="module-counter mono"
              >0{{ activeModule + 1 }} / 04</span
            >
            <div>
              <div class="eyebrow accent">{{ modules[activeModule].name }}</div>
              <h3>{{ modules[activeModule].summary }}</h3>
              <p>{{ modules[activeModule].body }}</p>
              <span class="module-tag">{{ modules[activeModule].tag }}</span>
            </div>
            <div class="module-metric">
              <strong>{{ modules[activeModule].metric }}</strong
              ><span>{{ modules[activeModule].metricLabel }}</span>
            </div>
          </div>
        </div>
        <div class="curation-strip">
          <div>
            <span class="mono accent">ADAPTIVE DATA CURATION</span>
            <h3>Better experience,<br />at every level.</h3>
          </div>
          <div
            v-for="[num, title, text] in [
              ['01', 'Task', 'Allocate rollout budget by difficulty.'],
              ['02', 'Trajectory', 'Bootstrap hard tasks with experience.'],
              ['03', 'Step', 'Focus updates on high-entropy decisions.'],
              ['04', 'Token', 'Align rollout and training distributions.'],
            ]"
            :key="num"
            class="curation-item"
          >
            <span class="mono muted">{{ num }}</span
            ><strong>{{ title }}</strong>
            <p>{{ text }}</p>
          </div>
        </div>
        <details class="paper-figure">
          <summary>
            Go deeper: the paper’s system diagram <span>+</span>
          </summary>
          <figure>
            <img
              :src="asset('research/architecture.png')"
              alt="DART paper figure showing the decoupled environment cluster, rollout service, data manager, and trainer"
              loading="lazy"
            />
            <figcaption>
              System architecture from the DART manuscript. The interactive
              diagram above is a simplified view.
            </figcaption>
          </figure>
        </details>
      </div>
    </section>

    <section id="results" class="results-section section-wrap section-space">
      <div class="section-heading">
        <div>
          <div class="eyebrow muted">03 / MEASURED PROGRESS</div>
          <h2>
            Small model.<br /><span class="muted">Meaningful gains.</span>
          </h2>
        </div>
        <div class="result-heading-side">
          <p>
            DART-GUI-7B vs. its UI-TARS-1.5-7B base.<br />Same benchmark. Same
            interaction budget.
          </p>
          <div
            class="segmented budget-toggle"
            aria-label="Maximum interaction steps"
          >
            <button
              v-for="value in [30, 100]"
              :key="value"
              :class="{ selected: budget === value }"
              :aria-pressed="budget === value"
              @click="budget = value"
            >
              {{ value }} steps
            </button>
          </div>
        </div>
      </div>
      <div class="result-grid">
        <div class="overall-card">
          <span class="mono muted">OSWORLD / OVERALL SUCCESS</span>
          <div class="overall-value">
            {{ results.total[1].toFixed(2) }}<span>%</span>
          </div>
          <div class="gain-badge">↗ +{{ gain }} percentage points</div>
          <div class="overall-comparison">
            <div>
              <span>UI-TARS-1.5-7B</span
              ><strong>{{ results.total[0].toFixed(2) }}%</strong>
            </div>
            <div class="mini-track">
              <i :style="{ width: `${results.total[0]}%` }"></i>
            </div>
            <div>
              <span class="accent">DART-GUI-7B</span
              ><strong class="accent"
                >{{ results.total[1].toFixed(2) }}%</strong
              >
            </div>
            <div class="mini-track">
              <i class="ours" :style="{ width: `${results.total[1]}%` }"></i>
            </div>
          </div>
          <p>Maximum {{ budget }} agent–environment interactions per task.</p>
        </div>
        <div class="category-chart">
          <div class="chart-heading">
            <span class="mono">ACROSS 10 APPLICATION DOMAINS</span>
            <div class="chart-legend">
              <span><i></i> Base model</span
              ><span><i class="ours"></i> DART</span>
            </div>
          </div>
          <div class="chart-axis mono muted">
            <span>Success rate (%)</span><span>0 — 100</span>
          </div>
          <div
            v-for="(category, index) in categories"
            :key="category"
            class="chart-row"
          >
            <span class="category-label">{{ category }}</span>
            <div
              class="bar-pair"
              role="img"
              :aria-label="`${category}: base ${results.baseline[index]}%, DART ${results.ours[index]}%`"
            >
              <div
                class="bar baseline"
                :style="{ width: `${results.baseline[index]}%` }"
              ></div>
              <div
                class="bar ours"
                :style="{ width: `${results.ours[index]}%` }"
              ></div>
            </div>
            <span class="chart-value mono">{{
              results.ours[index].toFixed(2)
            }}</span>
          </div>
        </div>
      </div>
      <details class="results-table">
        <summary>View the exact numbers <span>+</span></summary>
        <div class="table-scroll">
          <table>
            <caption>
              OSWorld task success rate (%) ·
              {{
                budget
              }}-step maximum
            </caption>
            <thead>
              <tr>
                <th scope="col">Application</th>
                <th scope="col">UI-TARS-1.5-7B</th>
                <th scope="col">DART-GUI-7B</th>
                <th scope="col">Gain (pp)</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(category, index) in categories" :key="category">
                <th scope="row">{{ category }}</th>
                <td>{{ results.baseline[index].toFixed(2) }}</td>
                <td>{{ results.ours[index].toFixed(2) }}</td>
                <td>
                  +{{
                    (results.ours[index] - results.baseline[index]).toFixed(2)
                  }}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </details>
      <p class="source-note">
        Results reported in the EMNLP camera-ready manuscript, OSWorld
        evaluation table. Efficiency gains compare the decoupled and
        non-decoupled systems under identical hardware and rollout settings. See
        the paper for full evaluation protocols.
      </p>
      <div class="efficiency-row">
        <div>
          <span class="mono muted">THROUGHPUT</span>
          <p>
            <strong>22.6 <span>→</span> 43.6</strong
            ><span>actions / minute</span>
          </p>
        </div>
        <div>
          <span class="mono muted">ENVIRONMENT UTILIZATION</span>
          <p>
            <strong>12.2 <span>→</span> 67.7</strong><span>percent</span>
          </p>
        </div>
        <div>
          <span class="mono muted">ROLLOUT GPU UTILIZATION</span>
          <p>
            <strong>29.6 <span>→</span> 46.7</strong><span>percent</span>
          </p>
        </div>
      </div>
    </section>

    <section
      id="resources"
      class="resources-section section-wrap section-space"
    >
      <div class="resources-intro">
        <div class="eyebrow muted">04 / BUILT TO BE BUILT UPON</div>
        <h2>Your next agent<br />starts <span class="accent">here.</span></h2>
        <p>Explore the framework. Run the model.<br />Build on the research.</p>
      </div>
      <div class="resource-links">
        <a
          href="https://arxiv.org/abs/2509.23866"
          target="_blank"
          rel="noopener noreferrer"
          ><span class="resource-number mono">01</span>
          <div>
            <h3>Read the paper</h3>
            <p>Method, experiments, and analysis</p>
          </div>
          <span class="resource-arrow">↗</span></a
        ><a
          href="https://github.com/Computer-use-agents/dart-gui"
          target="_blank"
          rel="noopener noreferrer"
          ><span class="resource-number mono">02</span>
          <div>
            <h3>Explore the code</h3>
            <p>Training, rollout, and evaluation</p>
          </div>
          <span class="resource-arrow">↗</span></a
        ><a
          href="https://huggingface.co/PengxiangLi/dart-gui-7b"
          target="_blank"
          rel="noopener noreferrer"
          ><span class="resource-number mono">03</span>
          <div>
            <h3>Get the model</h3>
            <p>DART-GUI-7B · open weights on Hugging Face</p>
          </div>
          <span class="resource-arrow">↗</span></a
        >
      </div>
    </section>

    <section class="paper-section section-wrap">
      <div class="paper-title">
        <span class="eyebrow muted">THE RESEARCH</span>
        <h2>
          Fully Decoupled Reinforcement Learning with<br class="desktop-only" />
          Adaptive Data Curation for GUI Agents
        </h2>
        <p class="paper-version">
          EMNLP manuscript title · Original arXiv title: “Efficient Multi-turn
          RL for GUI Agents via Decoupled Training and Adaptive Data Curation”
        </p>
      </div>
      <details class="abstract">
        <summary>Abstract <span>+</span></summary>
        <p>
          Vision-language models have catalyzed the development of GUI agents
          for desktop automation. Reinforcement learning remains bottlenecked by
          idle time inside the rollout stage and imbalanced learning signals
          from long-horizon, sparse-reward interactions. DART fully decouples
          the pipeline into four independent modules: environment cluster,
          rollout service, data manager, and trainer. Rollout-wise scheduling
          and per-worker model synchronization improve system efficiency, while
          adaptive curation at the task, trajectory, step, and token levels
          makes collected experience more useful. DART-GUI-7B achieves a 43.28%
          task success rate on OSWorld, a 15.76 percentage-point improvement
          over its base model at the same 100-step budget.
        </p>
      </details>
      <div class="citation-block">
        <div class="citation-top">
          <span class="mono">CITE THIS WORK</span
          ><button class="copy-button" @click="copyCitation" aria-live="polite">
            {{ copyStatus }} <span aria-hidden="true">⧉</span>
          </button>
        </div>
        <pre id="citation-code" tabindex="0"><code>{{ bibtex }}</code></pre>
      </div>
    </section>
  </main>
  <footer class="site-footer section-wrap">
    <a class="brand" href="#overview"
      >DART<span class="brand-slash">/</span
      ><span class="brand-gui">GUI</span></a
    ><span>Learning to act. Built in the open.</span
    ><a
      href="https://github.com/Computer-use-agents/dart-gui"
      target="_blank"
      rel="noopener noreferrer"
      >GitHub ↗</a
    ><a href="mailto:dylan.liqing@gmail.com">Get in touch ↗</a
    ><a href="#overview" class="back-top" aria-label="Back to top">↑</a>
  </footer>
</template>
