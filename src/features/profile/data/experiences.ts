import type { Experience } from "../types/experiences";

export const EXPERIENCES: Experience[] = [
  {
    id: "rebattery",
    companyName: "ReBattery",
    companyLogo: "/images/linkedin/organizations/rebattery.jpg",
    isCurrentEmployer: true,
    positions: [
      {
        id: "rebattery-cto",
        title: "Chief Technology Officer",
        employmentPeriod: {
          start: "08.2025",
        },
        employmentType: "Full-time",
        icon: "business",
        isExpanded: true,
        description: `- At ReBattery, I lead product and engineering to help teams make better battery decisions faster using reliable data and practical AI. I stay hands-on in architecture and execution so we can ship quickly without compromising quality or trust.`,
        skills: ["Technology Strategy", "AI Systems", "Leadership"],
      },
    ],
  },
  {
    id: "stealth-ai-startup",
    companyName: "Stealth AI Startup",
    companyLogo: "/images/linkedin/organizations/stealth-ai-startup.jpg",
    positions: [
      {
        id: "stealth-founder",
        title: "Founder",
        employmentPeriod: {
          start: "09.2023",
          end: "12.2025",
        },
        employmentType: "Full-time",
        icon: "idea",
        description: `- This was a concentrated period of building fast, close to clients, and testing the waters across multiple AI startup ideas. I worked directly with operators to understand workflow pain, then translated that into product bets that could be validated in-market, not just in prototypes.
- Over this era, I built four full platforms end-to-end and pushed each toward real commercial traction: **Salespeak**, **Reoutfit**, **EasyClaw**, and **20Punches**. The focus was always the same: ship something useful quickly, iterate from usage, and keep tightening the path from product to revenue.
- Beyond product and engineering, I led sales-heavy cycles and customer discovery loops in parallel - demos, outreach, qualification, closing, and feedback capture. It was equal parts structured execution and AI tinkering, and it shaped a durable founder playbook for building things people will actually pay for.`,
        skills: ["Entrepreneurship", "Product", "AI"],
      },
    ],
  },
  {
    id: "quantera",
    companyName: "Quantera",
    companyLogo: "/images/linkedin/organizations/quantera.jpg",
    positions: [
      {
        id: "quantera-quant-analyst",
        title: "Quantitative Analyst",
        employmentPeriod: {
          start: "10.2022",
          end: "09.2023",
        },
        employmentType: "Full-time",
        icon: "code",
        description: `- At Quantera, I built and stress-tested quantitative research workflows aimed at finding tradable signal, not just statistically pretty charts. The work blended market intuition with disciplined experimentation, from hypothesis design through backtesting and post-trade review.
- I explored intermarket momentum strategies and incorporated alternative datasets, including sentiment and event-driven inputs, to improve forecasting confidence. Every model had to earn its place by surviving realistic assumptions around noise, latency, and changing regimes.
- Beyond analysis, I helped turn research into action: clearer communication with stakeholders, stronger evaluation standards, and better visibility into what was actually driving performance. The outcome was a tighter research loop and more grounded portfolio decision-making.`,
        skills: [
          "Quantitative Research",
          "Statistical Analysis",
          "Portfolio Management",
          "Trading Systems",
        ],
      },
    ],
  },
  {
    id: "greenify",
    companyName: "Greenify LLC",
    companyLogo: "/images/linkedin/organizations/greenify.jpg",
    positions: [
      {
        id: "greenify-fullstack-dev",
        title: "Full Stack Developer",
        employmentPeriod: {
          start: "09.2021",
          end: "05.2022",
        },
        employmentType: "Full-time",
        icon: "code",
        description: `- At Greenify, I helped build an ESG-focused platform that needed to be both credible for analysts and usable for everyday client workflows. I worked across the stack to deliver product experiences that made complex sustainability data easier to navigate and trust.
- On the frontend, I focused on clean interaction patterns and performance-minded implementation in React and Next.js. On the backend, I supported data structures and service behavior that could scale without creating operational drag as usage increased.
- I also drove practical improvements in search visibility and product clarity, which helped the platform reach more relevant users and convert attention into usage. The core principle was straightforward: thoughtful engineering should make meaningful products feel simple.`,
        skills: ["React.js", "Next.js", "MongoDB", "SASS"],
      },
    ],
  },
  {
    id: "ecotek",
    companyName: "ECOTEK KS",
    companyLogo: "/images/linkedin/organizations/ecotek.jpg",
    positions: [
      {
        id: "ecotek-sales-marketing-associate",
        title: "Sales and Marketing Associate",
        employmentPeriod: {
          start: "05.2020",
          end: "09.2021",
        },
        employmentType: "Full-time",
        icon: "business",
        description: `- At ECOTEK, I learned the fundamentals of growth the unglamorous way: direct outreach, customer conversations, and consistent execution. We built demand by showing up daily, communicating clear value, and making sure prospects understood why our offer mattered.
- I ran campaigns across social, email, and outbound channels, balancing brand presence with practical conversion goals. The work sharpened my instincts for messaging, audience fit, and the difference between activity metrics and actual business outcomes.
- I also supported onboarding for new clients and helped build trust through responsiveness and clarity. That early operator experience still shapes how I build products today: distribution is part of the product, and attention is earned through usefulness, not noise.`,
        skills: ["Facebook Ads", "Social Media Marketing", "Digital Media"],
      },
    ],
  },
];
