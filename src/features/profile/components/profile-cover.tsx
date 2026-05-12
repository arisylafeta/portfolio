import { BrandContextMenu } from "@/components/brand-context-menu";
import { FlickeringGrid } from "@/components/ui/flickering-grid-hero";
import { cn } from "@/lib/utils";

export function ProfileCover() {
  return (
    <BrandContextMenu>
      <div
        id="js-cover-mark"
        className={cn(
          "relative aspect-2/1 border-x border-edge select-none sm:aspect-3/1",
          "flex items-center justify-center overflow-hidden text-black dark:text-white",
          "screen-line-before screen-line-after before:-top-px after:-bottom-px",
          "bg-white dark:bg-black"
        )}
      >
        <FlickeringGrid
          className="absolute inset-0"
          squareSize={3}
          gridGap={5}
          flickerChance={0.12}
          color="rgb(0, 0, 0)"
          maxOpacity={0.12}
        />

        <div className="pointer-events-none absolute right-0 bottom-0 left-0 h-px bg-edge" />
      </div>
    </BrandContextMenu>
  );
}
