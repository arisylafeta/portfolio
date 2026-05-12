import type { Project } from "../types/projects";

export const PROJECTS: Project[] = [
  {
    id: "kai-voice-agent",
    title: "KAI Voice Agent",
    period: { start: "03.2026" },
    link: "https://github.com/arisylafeta",
    skills: ["Next.js", "LiveKit", "Gemini", "Vercel"],
    isExpanded: true,
    description: `A voice-first assistant for everyday workflows.
- Real-time conversational interactions
- Low-latency responses for practical tasks
- Production-focused product architecture`,
    logo: "https://api.dicebear.com/7.x/shapes/svg?seed=KAI+Voice+Agent",
  },
  {
    id: "salespeak",
    title: "Salespeak",
    period: { start: "12.2024" },
    link: "https://github.com/arisylafeta",
    skills: ["TypeScript", "LLMs", "Voice AI", "Automation"],
    description: `Voice AI platform for SMB outreach and lead qualification.
- Automated top-of-funnel workflows
- Qualification flows for inbound and outbound leads
- Product analytics to improve conversion`,
    logo: "https://api.dicebear.com/7.x/shapes/svg?seed=Salespeak",
  },
  {
    id: "twenty-punches",
    title: "20 Punches",
    period: { start: "01.2026" },
    link: "https://github.com/arisylafeta",
    skills: ["React", "OpenAI API", "Speech", "Three.js"],
    description: `An experimental multi-modal AI assistant.
- Combines text, voice, and UI interactions
- Tests conversational and GUI agent handoff patterns
- Built as an R&D playground for product ideas`,
    logo: "https://api.dicebear.com/7.x/shapes/svg?seed=20+Punches",
  },
];
