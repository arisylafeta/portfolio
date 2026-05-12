import dayjs from "dayjs";
import { FileCheckIcon } from "lucide-react";
import Image from "next/image";

import { Icons } from "@/components/icons";
import { Markdown } from "@/components/markdown";
import {
  CollapsibleChevronsIcon,
  CollapsibleContent,
  CollapsibleTrigger,
  CollapsibleWithContext,
} from "@/components/ui/collapsible";
import { Separator } from "@/components/ui/separator";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { Prose } from "@/components/ui/typography";

import type { Award } from "../../types/awards";

export function AwardItem({
  className,
  award,
}: {
  className?: string;
  award: Award;
}) {
  const canExpand = !!award.description;
  const period =
    award.startDate && award.endDate
      ? `${award.startDate} - ${award.endDate}`
      : dayjs(award.date).format("MM.YYYY");

  return (
    <CollapsibleWithContext disabled={!canExpand} asChild>
      <div className={className}>
        <div className="flex items-center hover:bg-accent2">
          <div
            className="mx-4 flex size-6 shrink-0 items-center justify-center overflow-hidden rounded-md border border-muted-foreground/15 bg-muted ring-1 ring-edge ring-offset-1 ring-offset-background"
            aria-hidden
          >
            {award.logo ? (
              <Image
                src={award.logo}
                alt=""
                width={24}
                height={24}
                className="size-6 object-cover"
                unoptimized
              />
            ) : (
              <Icons.award className="pointer-events-none size-4 text-muted-foreground" />
            )}
          </div>

          <div className="flex-1 border-l border-dashed border-edge">
            <CollapsibleTrigger className="flex w-full items-center gap-4 p-4 pr-2 text-left select-none">
              <div className="flex-1">
                <h3 className="mb-1 leading-snug font-medium text-balance">
                  {award.title}
                </h3>

                <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-sm text-muted-foreground">
                  <dl>
                    <dt className="sr-only">Prize</dt>
                    <dd>{award.prize}</dd>
                  </dl>

                  <Separator
                    className="data-[orientation=vertical]:h-4"
                    orientation="vertical"
                  />

                  <dl>
                    <dt className="sr-only">Period</dt>
                    <dd>
                      <time dateTime={dayjs(award.date).toISOString()}>
                        {period}
                      </time>
                    </dd>
                  </dl>

                  <Separator
                    className="data-[orientation=vertical]:h-4"
                    orientation="vertical"
                  />

                  <dl>
                    <dt className="sr-only">Received in Grade</dt>
                    <dd>{award.grade}</dd>
                  </dl>
                </div>
              </div>

              {award.referenceLink && (
                <SimpleTooltip content="Open Reference Attachment">
                  <a
                    className="relative flex size-6 shrink-0 items-center justify-center text-muted-foreground after:absolute after:-inset-2 hover:text-foreground"
                    href={award.referenceLink}
                    target="_blank"
                    rel="noopener"
                  >
                    <FileCheckIcon
                      className="pointer-events-none size-4"
                      aria-hidden
                    />
                    <span className="sr-only">Open Reference Attachment</span>
                  </a>
                </SimpleTooltip>
              )}

              {canExpand && (
                <div
                  className="shrink-0 text-muted-foreground [&_svg]:size-4"
                  aria-hidden
                >
                  <CollapsibleChevronsIcon />
                </div>
              )}
            </CollapsibleTrigger>
          </div>
        </div>

        {canExpand && (
          <CollapsibleContent className="group overflow-hidden duration-300 data-[state=closed]:animate-collapsible-up data-[state=open]:animate-collapsible-down">
            <div className="border-t border-edge shadow-inner">
              <Prose className="p-4 duration-300 group-data-[state=closed]:animate-fade-out group-data-[state=open]:animate-fade-in">
                <Markdown>{award.description}</Markdown>
              </Prose>
            </div>
          </CollapsibleContent>
        )}
      </div>
    </CollapsibleWithContext>
  );
}
