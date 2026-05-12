import type { Experience } from "../types/experiences";

export const EXPERIENCES: Experience[] = [
  {
    id: "salespeak",
    companyName: "Salespeak",
    companyLogo: "https://api.dicebear.com/7.x/shapes/svg?seed=Salespeak",
    isCurrentEmployer: true,
    positions: [
      {
        id: "salespeak-cofounder",
        title: "Co-Founder",
        employmentPeriod: {
          start: "12.2024",
        },
        employmentType: "Full-time",
        icon: "business",
        isExpanded: true,
        description: `- Building a Voice AI product for SMB lead outreach and qualification.
- Defining product strategy, growth loops, and technical direction.
- Shipping fast experiments and production-grade features.`,
        skills: [
          "Product Strategy",
          "Voice AI",
          "LLMs",
          "TypeScript",
          "Next.js",
        ],
      },
    ],
    theme: true,
  },
  {
    id: "quantera",
    companyName: "Quantera",
    companyLogo: "https://api.dicebear.com/7.x/shapes/svg?seed=Quantera",
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
        description: `- Built and maintained data-driven models for market analysis.
- Worked closely with stakeholders to turn analysis into decisions.
- Improved reporting pipelines and internal workflows.`,
        skills: ["Python", "Data Analysis", "Modeling", "Research"],
      },
    ],
  },
  {
    id: "solaborate",
    companyName: "Solaborate",
    companyLogo: "https://api.dicebear.com/7.x/shapes/svg?seed=Solaborate",
    positions: [
      {
        id: "solaborate-ml-intern",
        title: "Machine Learning Intern",
        employmentPeriod: {
          start: "07.2022",
          end: "09.2022",
        },
        employmentType: "Internship",
        icon: "code",
        description: `- Prototyped ML features and evaluated model behavior.
- Supported data preparation and experimentation cycles.
- Collaborated with engineering on integration and delivery.`,
        skills: ["Machine Learning", "Python", "Experimentation"],
      },
    ],
  },
  {
    id: "education",
    companyName: "Education",
    positions: [
      {
        id: "birkbeck-msc",
        title: "Birkbeck, University of London",
        employmentPeriod: {
          start: "2023",
          end: "2024",
        },
        icon: "education",
        description: `- MSc, Quantitative Finance with Data Science.`,
        skills: ["Quantitative Finance", "Data Science"],
      },
      {
        id: "rit-baasc",
        title: "Rochester Institute of Technology",
        employmentPeriod: {
          start: "2018",
          end: "2022",
        },
        icon: "education",
        description: `- BAASc, Web and Mobile Computing in IT.`,
        skills: ["Web Development", "Mobile Development"],
      },
    ],
  },
];
