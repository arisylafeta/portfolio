import type { User } from "@/features/profile/types/user";

export const USER: User = {
  firstName: "Arianit",
  lastName: "Sylafeta",
  displayName: "Arianit Sylafeta",
  username: "arisylafeta",
  gender: "male",
  pronouns: "he/him",
  bio: "Founder building AI-native products with a strong product and execution mindset.",
  timeZone: "Europe/London",
  flipSentences: [
    "Founder",
    "AI Engineer",
    "Voice AI Builder",
    "Fullstack Product Developer",
  ],
  address: "London, United Kingdom",
  phoneNumber: "",
  secondPhoneNumber: "",
  // base64-string-converter)
  email: "YXJpYW5pdC5zeWxhZmV0YUBnbWFpbC5jb20=", // base64 encoded
  website: "https://arisylafeta.com",
  jobTitle: "Co-Founder and CTO @ ReBattery",
  jobs: [
    {
      title: "Co-Founder and CTO",
      company: "ReBattery",
      website: "https://rebattery.co.uk",
    },
    {
      title: "Quantitative Analyst",
      company: "Quantera",
      website: "#",
    },
  ],
  about: `
- I am currently CTO at **ReBattery**.
- I help make the battery market more liquid by applying a technical lens to battery data and customer transactions.
- I enjoy turning complex ideas into clear product experiences, from prototype to production.
- My interests include energy markets, agent commerce, and AI organizational management.
- Outside work, I spend time on chess and Brazilian jiu-jitsu.
`,
  avatar: "/images/me.png",
  ogImage: "/images/og-image-light.png",
  namePronunciationUrl: "/audio/abdulrehman.mp3",
  keywords: [
    "arianit sylafeta",
    "arisylafeta",
    "founder",
    "ai engineer",
    "voice ai",
    "salespeak",
    "fullstack developer",
    "nextjs",
    "react",
  ],
  dateCreated: "2026-05-12", // YYYY-MM-DD
};
