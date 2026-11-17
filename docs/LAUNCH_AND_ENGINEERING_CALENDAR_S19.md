# Project HiveMind — Season 19 Google Calendar, GitHub Milestones & GitHub Project v2 (`Oct 28 – Nov 8, 2026`)

> **Synchronized Across Google Calendar (`callensxavier@gmail.com`), GitHub Milestones (`#1`–`#4`), and GitHub Project v2 (`#7`)**

---

## 1. Quick Access Links (Google Calendar, GitHub Project & iCal Feed)

| Platform | Direct Link | Access / Purpose |
| :--- | :--- | :--- |
| **Google Calendar (1-Click Add to `callensxavier@gmail.com` / Any Google Account)** | **[Add Project HiveMind S19 Calendar to Google Calendar](https://calendar.google.com/calendar/u/0/r?cid=NjUyNTMzZTUyZGFlOTU2NDI4Mzk5OTIyNTAzYjI4NGI0ZTY3ZmQyNGU3MDBjZmE3ZTJmNmJhMzJmZDJlNmY3ZEBncm91cC5jYWxlbmRhci5nb29nbGUuY29t)** | Shared directly with `callensxavier@gmail.com` (`owner`) and publicly readable by all Bee & Hive Miners |
| **Google Calendar Public Web View (Embed)** | **[Open Interactive Google Calendar (Europe/Paris)](https://calendar.google.com/calendar/embed?src=652533e52dae956428399922503b284b4e67fd24e700cfa7e2f6ba32fd2e6f7d%40group.calendar.google.com&ctz=Europe%2FParis)** | Live browser calendar view (`10` scheduled milestones & launch waves) |
| **Google Calendar Public `.ics` Feed** | **[Google Calendar iCal (`basic.ics`)](https://calendar.google.com/calendar/ical/652533e52dae956428399922503b284b4e67fd24e700cfa7e2f6ba32fd2e6f7d%40group.calendar.google.com/public/basic.ics)** | Live iCal subscription URL for Apple Calendar, Outlook, Thunderbird, or Proton Calendar |
| **GitHub Repository `.ics` Feed** | **[`docs/calendar/hivemind_s19_international_launch.ics`](calendar/hivemind_s19_international_launch.ics)** / **[Public Site `.ics`](https://hivemindathome.github.io/calendar/hivemind_s19_international_launch.ics)** | Version-controlled RFC 5545 `.ics` file in GitHub |
| **GitHub Project v2 Roadmap & Board** | **[GitHub Project #7 — HiveMind S19 Calendar & Roadmap](https://github.com/users/xaviercallens/projects/7)** | Public GitHub Project v2 linked to `xaviercallens/HiveMind` with `Start Date`, `Target Date`, `Launch Phase`, and `Workstream` |
| **GitHub Repository Milestones** | **[Milestone #1 (Oct 28–31)](https://github.com/xaviercallens/HiveMind/milestone/1)** · **[Milestone #2 (Sun Nov 1)](https://github.com/xaviercallens/HiveMind/milestone/2)** · **[Milestone #3 (Nov 2–5)](https://github.com/xaviercallens/HiveMind/milestone/3)** · **[Milestone #4 (Nov 6–8)](https://github.com/xaviercallens/HiveMind/milestone/4)** | Dated GitHub Milestones grouping Issues `#238`–`#244`, `#247`, and `#251` |

---

## 2. Visual Schedule (`Oct 28 – Nov 8, 2026`)

```mermaid
flowchart TD
    subgraph P0["Phase 0: Core Qwen 3.5, Idle MoE & AMCP/EDA Prep (Wed Oct 28 – Sat Oct 31)"]
        D1["Wed Oct 28 (09:00–18:00 CET)<br/>Day 1 (#239): Producer Hardware Auto-Tiering<br/>(std_8gb / big_12gb / xl_16gb_plus)"]
        D2["Thu Oct 29 (09:00–18:00 CET)<br/>Day 2 (#240): Usage-Driven 13-Bee MoE Allocator<br/>(65% Coding / 25% Science / 10% BYOB)"]
        D3["Fri Oct 30 (09:00–18:00 CET)<br/>Day 3 (#241): Daytime Micro-Idle Harvester<br/>(>=30s Idle, <1.0s Preemption) & Nocturnal DiLoCo"]
        D4["Sat Oct 31 (09:00–19:00 CET)<br/>Day 4 (#242): Big-to-8GB SVD Rank-16 LoRA Projection<br/>& AMCP v1.5 EDA Pre-Launch Gate"]
        D1 --> D2 --> D3 --> D4
    end

    subgraph P1["Phase 1: Sunday Producer-First International Launch (Sun Nov 1)"]
        D5["Sun Nov 1 (10:00–14:00 CET)<br/>Day 5 (#243): Speculative Decoding Rollout<br/>(Qwen3.5-Coder-3B Draft -> 14B / 30B-A3B Target)"]
        SUN["🚀 Sun Nov 1 (14:00–22:00 CET / 13:00 UTC)<br/>Producer-First Onboarding Wave (#251, #247)<br/>r/BOINC, r/homelab, r/gpumining, CrowdGPT (Target: >=38 Active GPUs)"]
        NOC["🌙 Sun Nov 1 (22:00) – Mon Nov 2 (07:00 CET)<br/>Nocturnal Swarm DiLoCo + MoE Training & 06:30 SVD Broadcast"]
        D5 --> SUN --> NOC
    end

    subgraph P2["Phase 2: English Prosumer & Open-Weights Dev Mix (Mon Nov 2 – Thu Nov 5)"]
        D6["Mon Nov 2 (09:00–17:00 CET)<br/>Day 6–7 (#244): Bring Your Own Bee (BYOB),<br/>Crowd-v1 & 3,000-Dev IDE Rollout"]
        W2["🌍 Mon Nov 2 (17:00 CET) – Thu Nov 5 (21:00 CET)<br/>Phase 2 Prosumer/Dev Mix Launch (#238)<br/>r/LocalLLaMA, r/SelfHosted (AMCP Elastic Scale -> 55 GPUs)"]
        D6 --> W2
    end

    subgraph P3["Phase 3: Global Show HN & Full Consumer/Prosumer Mix (Fri Nov 6 – Sun Nov 8)"]
        W3["🔥 Fri Nov 6 (15:00 CET) – Sun Nov 8 (22:00 CET)<br/>Phase 3 Global Show HN Launch (#247)<br/>AMCP + CloudEvents EDA (up to 114 GPUs, Kelly Surge Pricing, 2-Lane QoS)"]
    end

    P0 --> P1 --> P2 --> P3
```

---

## 3. Master Event Table with Direct Google Calendar & GitHub Links

| Date & Time (`Europe/Paris` / `CET`) | Launch Phase & Milestone | Event Title | GitHub Issue & Milestone | Google Calendar Links |
| :--- | :--- | :--- | :--- | :--- |
| **Wed, Oct 28, 2026**<br/>`09:00–18:00 CET` | **Phase 0**<br/>[Milestone #1](https://github.com/xaviercallens/HiveMind/milestone/1) | **[HiveMind S19 Day 1]** Producer Hardware Auto-Tiering (`std_8gb` / `big_12gb` / `xl_16gb_plus`) | **[#239](https://github.com/xaviercallens/HiveMind/issues/239)** | [View Event](https://www.google.com/calendar/event?eid=bDU4cHRrb25zNjlzbmZha2oxNjZqdmJyZG8gNjUyNTMzZTUyZGFlOTU2NDI4Mzk5OTIyNTAzYjI4NGI0ZTY3ZmQyNGU3MDBjZmE3ZTJmNmJhMzJmZDJlNmY3ZEBn) · [1-Click Add](https://calendar.google.com/calendar/render?action=TEMPLATE&text=%5BHiveMind+S19+Day+1%5D+Producer+Hardware+Auto-Tiering+%28std_8gb+%2F+big_12gb+%2F+xl_16gb_plus%29&dates=20261028T080000Z%2F20261028T170000Z) |
| **Thu, Oct 29, 2026**<br/>`09:00–18:00 CET` | **Phase 0**<br/>[Milestone #1](https://github.com/xaviercallens/HiveMind/milestone/1) | **[HiveMind S19 Day 2]** Usage-Driven 13-Bee MoE Workunit Allocator (`65%` Coding / `25%` Science / `10%` BYOB) | **[#240](https://github.com/xaviercallens/HiveMind/issues/240)** | [View Event](https://www.google.com/calendar/event?eid=aGk3MjhlZGJxYXVmdWlkbmFzbDE2NzFoMzAgNjUyNTMzZTUyZGFlOTU2NDI4Mzk5OTIyNTAzYjI4NGI0ZTY3ZmQyNGU3MDBjZmE3ZTJmNmJhMzJmZDJlNmY3ZEBn) · [1-Click Add](https://calendar.google.com/calendar/render?action=TEMPLATE&text=%5BHiveMind+S19+Day+2%5D+Usage-Driven+13-Bee+MoE+Workunit+Allocator+%2865%25+Coding+%2F+25%25+Science+%2F+10%25+BYOB%29&dates=20261029T080000Z%2F20261029T170000Z) |
| **Fri, Oct 30, 2026**<br/>`09:00–18:00 CET` | **Phase 0**<br/>[Milestone #1](https://github.com/xaviercallens/HiveMind/milestone/1) | **[HiveMind S19 Day 3]** Daytime Micro-Idle Harvester ($\ge 30\text{s}$ Idle, $< 1.0\text{s}$ Preemption) & Nocturnal Scheduler | **[#241](https://github.com/xaviercallens/HiveMind/issues/241)** | [View Event](https://www.google.com/calendar/event?eid=MGdpcGRlZzZxcXVmcWI4bHRlODBuMXJjbXMgNjUyNTMzZTUyZGFlOTU2NDI4Mzk5OTIyNTAzYjI4NGI0ZTY3ZmQyNGU3MDBjZmE3ZTJmNmJhMzJmZDJlNmY3ZEBn) · [1-Click Add](https://calendar.google.com/calendar/render?action=TEMPLATE&text=%5BHiveMind+S19+Day+3%5D+Daytime+Micro-Idle+Harvester+%28%3E%3D30s+Idle%2C+%3C1.0s+Preemption%29+%26+Nocturnal+Scheduler&dates=20261030T080000Z%2F20261030T170000Z) |
| **Sat, Oct 31, 2026**<br/>`09:00–19:00 CET` | **Phase 0**<br/>[Milestone #1](https://github.com/xaviercallens/HiveMind/milestone/1) | **[HiveMind S19 Day 4]** Big-Producer Teacher-to-Student SVD LoRA Projection & AMCP v1.5 EDA Pre-Launch Gate | **[#242](https://github.com/xaviercallens/HiveMind/issues/242)** | [View Event](https://www.google.com/calendar/event?eid=bW9mODFpNTI5MDdrdmN1aTQwZXQ2cDB1bXMgNjUyNTMzZTUyZGFlOTU2NDI4Mzk5OTIyNTAzYjI4NGI0ZTY3ZmQyNGU3MDBjZmE3ZTJmNmJhMzJmZDJlNmY3ZEBn) · [1-Click Add](https://calendar.google.com/calendar/render?action=TEMPLATE&text=%5BHiveMind+S19+Day+4%5D+Big-Producer+Teacher-to-Student+SVD+LoRA+Projection+%26+AMCP+v1.5+EDA+Pre-Launch+Gate&dates=20261031T080000Z%2F20261031T180000Z) |
| **Sun, Nov 1, 2026**<br/>`10:00–14:00 CET` | **Phase 1**<br/>[Milestone #2](https://github.com/xaviercallens/HiveMind/milestone/2) | **[HiveMind S19 Day 5]** Speculative Decoding Rollout (`Qwen3.5-Coder-3B` Draft $\to$ `14B` / `30B-A3B` Target) | **[#243](https://github.com/xaviercallens/HiveMind/issues/243)** | [View Event](https://www.google.com/calendar/event?eid=b2ZlczdkMXY4ZWlkMTkwMmRwZmkwbnVvdGsgNjUyNTMzZTUyZGFlOTU2NDI4Mzk5OTIyNTAzYjI4NGI0ZTY3ZmQyNGU3MDBjZmE3ZTJmNmJhMzJmZDJlNmY3ZEBn) · [1-Click Add](https://calendar.google.com/calendar/render?action=TEMPLATE&text=%5BHiveMind+S19+Day+5%5D+Speculative+Decoding+Rollout+%28Qwen3.5-Coder-3B+Draft+-%3E+14B+%2F+30B-A3B+Target%29&dates=20261101T090000Z%2F20261101T130000Z) |
| **Sun, Nov 1, 2026**<br/>`14:00–22:00 CET`<br/>(`13:00–21:00 UTC`) | **Phase 1**<br/>[Milestone #2](https://github.com/xaviercallens/HiveMind/milestone/2) | 🚀 **[SUNDAY LAUNCH — PHASE 1]** Producer-First International Onboarding Wave (`r/BOINC`, `r/homelab`, `r/gpumining`, CrowdGPT) | **[#251](https://github.com/xaviercallens/HiveMind/issues/251)** · **[#247](https://github.com/xaviercallens/HiveMind/issues/247)** | [View Event](https://www.google.com/calendar/event?eid=a3UxaWNpZnI1YWQydXRpZjkyYnA3cGwxbW8gNjUyNTMzZTUyZGFlOTU2NDI4Mzk5OTIyNTAzYjI4NGI0ZTY3ZmQyNGU3MDBjZmE3ZTJmNmJhMzJmZDJlNmY3ZEBn) · [1-Click Add](https://calendar.google.com/calendar/render?action=TEMPLATE&text=%F0%9F%9A%80+%5BSUNDAY+LAUNCH+%E2%80%94+PHASE+1%5D+Producer-First+International+Onboarding+Wave+%28r%2FBOINC%2C+r%2Fhomelab%2C+r%2Fgpumining%2C+CrowdGPT%29&dates=20261101T130000Z%2F20261101T210000Z) |
| **Sun, Nov 1 – Mon, Nov 2**<br/>`22:00–07:00 CET` | **Phase 1**<br/>[Milestone #2](https://github.com/xaviercallens/HiveMind/milestone/2) | 🌙 **[Nocturnal Swarm Window]** DiLoCo + Evolutionary MoE Training & Rank-32$\to$16 SVD Broadcast | **[#238](https://github.com/xaviercallens/HiveMind/issues/238)** | [View Event](https://www.google.com/calendar/event?eid=a3Zpc3Rpb2NuOTA3cTZlcXRtOGJwNzZrNjAgNjUyNTMzZTUyZGFlOTU2NDI4Mzk5OTIyNTAzYjI4NGI0ZTY3ZmQyNGU3MDBjZmE3ZTJmNmJhMzJmZDJlNmY3ZEBn) · [1-Click Add](https://calendar.google.com/calendar/render?action=TEMPLATE&text=%F0%9F%8C%99+%5BNocturnal+Swarm+Window%5D+DiLoCo+%2B+Evolutionary+MoE+Training+%26+Rank-32-%3E16+SVD+Broadcast+%2822%3A00-07%3A00%29&dates=20261101T210000Z%2F20261102T060000Z) |
| **Mon, Nov 2, 2026**<br/>`09:00–17:00 CET` | **Phase 2**<br/>[Milestone #3](https://github.com/xaviercallens/HiveMind/milestone/3) | **[HiveMind S19 Day 6–7]** Bring Your Own Bee (`BYOB`) & `Crowd-v1` Swarm Integration (`@Vxtzq` + `3,000` Devs) | **[#244](https://github.com/xaviercallens/HiveMind/issues/244)** | [View Event](https://www.google.com/calendar/event?eid=YWZyMm9ta2JldmlpaXRpZ3U1MnA1cHZrajAgNjUyNTMzZTUyZGFlOTU2NDI4Mzk5OTIyNTAzYjI4NGI0ZTY3ZmQyNGU3MDBjZmE3ZTJmNmJhMzJmZDJlNmY3ZEBn) · [1-Click Add](https://calendar.google.com/calendar/render?action=TEMPLATE&text=%5BHiveMind+S19+Day+6-7%5D+Bring+Your+Own+Bee+%28BYOB%29+%26+Crowd-v1+Swarm+Integration+%28%40Vxtzq+%2B+3%2C000+Devs%29&dates=20261102T080000Z%2F20261102T160000Z) |
| **Mon, Nov 2 – Thu, Nov 5**<br/>`17:00 CET – 21:00 CET` | **Phase 2**<br/>[Milestone #3](https://github.com/xaviercallens/HiveMind/milestone/3) | 🌍 **[PHASE 2 LAUNCH]** English Prosumer & Open-Weights Dev Mix Wave (`r/LocalLLaMA`, `r/SelfHosted`) | **[#238](https://github.com/xaviercallens/HiveMind/issues/238)** | [View Event](https://www.google.com/calendar/event?eid=Z2ZtaWlqOG5lZmYwYTZ0N3NyYWhuMzhicW8gNjUyNTMzZTUyZGFlOTU2NDI4Mzk5OTIyNTAzYjI4NGI0ZTY3ZmQyNGU3MDBjZmE3ZTJmNmJhMzJmZDJlNmY3ZEBn) · [1-Click Add](https://calendar.google.com/calendar/render?action=TEMPLATE&text=%F0%9F%8C%8D+%5BPHASE+2+LAUNCH%5D+English+Prosumer+%26+Open-Weights+Dev+Mix+Wave+%28r%2FLocalLLaMA%2C+r%2FSelfHosted%29&dates=20261102T160000Z%2F20261105T200000Z) |
| **Fri, Nov 6 – Sun, Nov 8**<br/>`15:00 CET – 22:00 CET` | **Phase 3**<br/>[Milestone #4](https://github.com/xaviercallens/HiveMind/milestone/4) | 🔥 **[PHASE 3 GLOBAL LAUNCH]** `Show HN` + Global English Consumer/Prosumer Mix Wave | **[#247](https://github.com/xaviercallens/HiveMind/issues/247)** | [View Event](https://www.google.com/calendar/event?eid=Y2Vhc3MwZGFhMzQ2czRwZWFucTJoN21sMTAgNjUyNTMzZTUyZGFlOTU2NDI4Mzk5OTIyNTAzYjI4NGI0ZTY3ZmQyNGU3MDBjZmE3ZTJmNmJhMzJmZDJlNmY3ZEBn) · [1-Click Add](https://calendar.google.com/calendar/render?action=TEMPLATE&text=%F0%9F%94%A5+%5BPHASE+3+GLOBAL+LAUNCH%5D+Show+HN+%2B+Global+English+Consumer%2FProsumer+Mix+Wave&dates=20261106T140000Z%2F20261108T210000Z) |
