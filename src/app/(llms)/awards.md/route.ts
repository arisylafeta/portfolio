import { AWARDS } from "@/features/profile/data/awards";

const content = `# Awards

${AWARDS.map((item) => {
  const period =
    item.startDate && item.endDate
      ? `\n\nPeriod: ${item.startDate} - ${item.endDate}`
      : "";
  return `## ${item.prize} | ${item.title}${period}\n\n${item.description}`;
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
