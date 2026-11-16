import { PROJECTS } from "@/features/profile/data/projects";

const content = `# Projects

${PROJECTS.map((item) => {
  const skills = `\n\nSkills: ${item.skills.join(", ")}`;
  const description = item.description ? `\n\n${item.description.trim()}` : "";
  const period = `\n\nPeriod: ${item.period.start} - ${item.period.end || "Present"}`;
  return `## ${item.title}\n\nProject URL: ${item.link}${period}${skills}${description}`;
}).join("\n\n")}
`;

export const dynamic = "force-static";

export async function GET() {
  return new Response(content, {
    headers: {
      "Content-Type": "text/markdown;charset=utf-8",
    },
  });
}
