import { ArrowRight } from "lucide-react";
import type { Metadata } from "next";
import Image from "next/image";
import Link from "next/link";

import { Button } from "@/components/ui/button";
import { FlickeringGrid } from "@/components/ui/flickering-grid-hero";
import { cn } from "@/lib/utils";

export const metadata: Metadata = {
  title: "Products - Arianit Sylafeta",
  description:
    "Explore my products and platforms across AI, sales, infrastructure, and e-commerce.",
};

const products = [
  {
    id: "rebattery",
    name: "ReBattery",
    icon: "/images/linkedin/organizations/rebattery.jpg",
    description:
      "Market infrastructure product helping make battery transactions more liquid through better data and AI-enabled execution workflows.",
    href: "https://rebattery.io/",
    status: "Live",
    tech: ["Energy Markets", "Market Infrastructure", "AI Systems", "Data"],
    image: "/images/products/rebattery.png",
  },
  {
    id: "easyclaw",
    name: "EasyClaw",
    icon: "https://easyclaw-navy.vercel.app/rounded-logo.png",
    description:
      "Managed OpenClaw infrastructure for deploying personal AI agents without DevOps friction.",
    href: "https://easyclaw-navy.vercel.app/",
    status: "Deprecated",
    tech: [
      "Infrastructure",
      "Agent Harnesses",
      "Open Source",
      "Cloud Automation",
    ],
    image: "/images/products/easyclaw.png",
  },
  {
    id: "reoutfit",
    name: "Reoutfit",
    icon: "https://www.reoutfit.me/logo.png",
    description:
      "AI-powered e-commerce styling and virtual try-on experience built to reduce buyer hesitation.",
    href: "https://www.reoutfit.me/",
    status: "Deprecated",
    tech: ["E-commerce", "AI Styling", "Personalization", "Retail UX"],
    image: "/images/products/reoutfit.png",
  },
  {
    id: "salespeak",
    name: "Salespeak",
    icon: "https://salespeak-seven.vercel.app/favicon.png",
    description:
      "Sales product focused on outbound momentum, lead qualification, and consistent pipeline generation.",
    href: "https://salespeak-seven.vercel.app/",
    status: "Deprecated",
    tech: ["Sales", "Lead Generation", "Outbound", "Voice AI"],
    image: "/images/products/salespeak.png",
  },
  {
    id: "twenty-punches",
    name: "20Punches",
    icon: "https://www.20punches.co.uk/favicon.svg",
    description:
      "Investment intelligence product using AI MCP patterns, skills, and multi-agent analysis loops.",
    href: "https://www.20punches.co.uk/",
    status: "Deprecated",
    tech: [
      "Investments",
      "AI MCP",
      "Multi-Agent Systems",
      "Portfolio Intelligence",
    ],
    image: "/images/products/20punches.png",
  },
];

export default function ProductsPage() {
  return (
    <div className="mx-auto border-x border-edge md:max-w-3xl">
      <div className="screen-line-after px-4 py-2.5">
        <p className="font-mono text-xs tracking-wide text-muted-foreground uppercase">
          Products
        </p>
      </div>

      <SectionDivider />

      <div className="relative aspect-2/1 bg-white text-black sm:aspect-3/1 dark:bg-black dark:text-white">
        <FlickeringGrid
          className="absolute inset-0"
          squareSize={3}
          gridGap={5}
          flickerChance={0.12}
          color="rgb(0, 0, 0)"
          maxOpacity={0.12}
        />

        <div className="relative z-10 flex h-full flex-col items-center justify-center px-6 text-center">
          <h1 className="text-3xl font-semibold tracking-tight md:text-4xl">
            My Products
          </h1>
          <p className="mt-2 max-w-xl text-xs text-muted-foreground md:text-sm">
            Five products built across AI, commerce, sales, market
            infrastructure, and agent tooling.
          </p>
        </div>

        <div className="pointer-events-none absolute right-0 bottom-0 left-0 h-px bg-edge" />
      </div>

      <SectionDivider />

      <div className="border-t border-edge">
        {products.map((product) => (
          <div
            key={product.id}
            className="group relative border-b border-edge bg-background transition-colors hover:bg-accent2/40"
          >
            <div className="pointer-events-none absolute -bottom-px -left-[100vw] h-px w-[200vw] bg-[repeating-linear-gradient(315deg,var(--pattern-foreground)_0,var(--pattern-foreground)_1px,transparent_0,transparent_50%)] bg-size-[10px_10px] [--pattern-foreground:var(--color-edge)]/56" />

            <div className="grid gap-6 p-6 md:grid-cols-3 md:p-8">
              <div className="flex items-center justify-center overflow-hidden rounded-lg border border-edge bg-muted md:col-span-1">
                <Image
                  src={product.image}
                  alt={`${product.name} preview`}
                  width={800}
                  height={520}
                  className="h-full w-full object-cover"
                  unoptimized
                />
              </div>

              <div className="md:col-span-2">
                <div className="mb-4 flex items-center gap-3">
                  <Image
                    src={product.icon}
                    alt=""
                    width={22}
                    height={22}
                    className="size-5 rounded-sm object-contain"
                    unoptimized
                    aria-hidden="true"
                  />
                  <h2 className="text-2xl font-bold">{product.name}</h2>
                  <span
                    className={`rounded-full px-3 py-1 text-xs font-medium ${
                      product.status === "Live"
                        ? "bg-green-500/10 text-green-600 dark:text-green-400"
                        : "bg-amber-500/10 text-amber-700 dark:text-amber-400"
                    }`}
                  >
                    {product.status}
                  </span>
                </div>

                <p className="mb-4 text-base text-muted-foreground">
                  {product.description}
                </p>

                <div className="mb-6 flex flex-wrap gap-2">
                  {product.tech.map((tech) => (
                    <span
                      key={tech}
                      className="rounded-md border bg-muted px-2 py-1 text-xs font-medium"
                    >
                      {tech}
                    </span>
                  ))}
                </div>

                <Button asChild>
                  <Link href={product.href}>
                    Open Product
                    <ArrowRight className="ml-2 h-4 w-4" />
                  </Link>
                </Button>
              </div>
            </div>
          </div>
        ))}
      </div>

      <SectionDivider />
    </div>
  );
}

function SectionDivider({ className }: { className?: string }) {
  return (
    <div
      className={cn(
        "relative flex h-8 w-full",
        "before:absolute before:-left-[100vw] before:-z-1 before:h-8 before:w-[200vw]",
        "before:bg-[repeating-linear-gradient(315deg,var(--pattern-foreground)_0,var(--pattern-foreground)_1px,transparent_0,transparent_50%)] before:bg-size-[10px_10px] before:[--pattern-foreground:var(--color-edge)]/56",
        className
      )}
    />
  );
}
