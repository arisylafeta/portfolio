import type { Project } from "../types/projects";

export const PROJECTS: Project[] = [
  {
    id: "rebattery",
    title: "ReBattery",
    tagline: "Battery market infrastructure, built for liquidity.",
    period: { start: "08.2025" },
    link: "https://rebattery.io/",
    skills: ["Energy Markets", "Market Infrastructure", "AI Systems", "Data"],
    description: `A product platform focused on making the battery market more liquid, discoverable, and execution-ready.
- Built to connect fragmented market data, customer workflows, and transaction intelligence into one operating layer.
- Combines practical AI systems with domain-specific product design to accelerate matching, qualification, and commercial decision cycles.
- Designed with operator-first constraints: reliable data foundations, clear execution paths, and tooling that scales with real market complexity.`,
    logo: "/images/linkedin/organizations/rebattery.jpg",
  },
  {
    id: "easyclaw",
    title: "EasyClaw",
    tagline: "Open-source agents, cloud-simple deployment.",
    period: { start: "10.2025", end: "02.2026" },
    link: "https://easyclaw-navy.vercel.app/",
    skills: [
      "Infrastructure",
      "Agent Harnesses",
      "Open Source",
      "Cloud Automation",
    ],
    isExpanded: true,
    description: `An infrastructure layer for deploying OpenClaw agents without the operational overhead.
- Built to remove setup friction through one-click provisioning, managed updates, and isolated runtime environments.
- Focused on robust agent harnesses so builders can experiment with tools, memory, and automation safely while preserving developer control.
- Bridges open-source flexibility with production-grade operability, giving teams a faster route from agent idea to stable deployment.`,
    logo: "https://easyclaw-navy.vercel.app/rounded-logo.png",
  },
  {
    id: "reoutfit",
    title: "Reoutfit",
    tagline: "Try on the future of e-commerce styling.",
    period: { start: "07.2025", end: "10.2025" },
    link: "https://www.reoutfit.me/",
    skills: ["E-commerce", "AI Styling", "Personalization", "Retail UX"],
    description: `An AI-assisted e-commerce experience for discovering and trying outfits before purchase.
- Combines conversational styling, virtual try-on mechanics, and product discovery across multiple fashion sources in one flow.
- Designed to reduce decision fatigue and buyer's remorse by helping users test fit and style intent earlier in the journey.
- Product work centered on conversion-quality UX: making exploration feel playful while still optimizing for trust, clarity, and checkout-ready confidence.`,
    logo: "https://www.google.com/s2/favicons?domain=reoutfit.me&sz=128",
  },
  {
    id: "salespeak",
    title: "Salespeak",
    tagline: "Pipeline growth, engineered for modern sales teams.",
    period: { start: "01.2025", end: "05.2025" },
    link: "https://salespeak-seven.vercel.app/",
    skills: ["Sales", "Lead Generation", "Outbound", "Voice AI"],
    description: `A sales execution platform focused on helping teams generate pipeline with less manual lift.
- Combines outbound structure, messaging systems, and AI-assisted workflows to improve top-of-funnel consistency.
- Designed for practical sales outcomes: better connect rates, cleaner qualification, and more predictable meeting flow.
- Built with an operator mindset where go-to-market feedback directly informs product iteration, so the system improves with every campaign and every call block.`,
    logo: "https://salespeak-seven.vercel.app/favicon.png",
  },
  {
    id: "twenty-punches",
    title: "20Punches",
    tagline: "AI investing copilots for clearer market decisions.",
    period: { start: "08.2025", end: "10.2025" },
    link: "https://www.20punches.co.uk/",
    skills: [
      "Investments",
      "AI MCP",
      "Multi-Agent Systems",
      "Portfolio Intelligence",
    ],
    description: `An AI-native investing product designed to turn financial anxiety into confident decision-making.
- Built around portfolio intelligence, practical market context, and conversational analysis instead of static dashboards.
- Implemented AI MCP patterns and skills-based agent workflows so users can query opportunities, stress-test ideas, and reason across signals without context-switching.
- Experimented deeply with multi-agent orchestration to separate research, critique, and execution loops; the result is a cleaner path from noisy market data to useful investor actions.`,
    logo: "https://www.google.com/s2/favicons?domain=20punches.co.uk&sz=128",
  },
];
