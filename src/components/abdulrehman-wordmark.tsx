const UNIT = 32;
const GAP = 2; // 2 units = 64px, matching the A-S gap in the mark

// prettier-ignore
const LETTER_A: number[][] = [
  [1,1,1,1,1,1],
  [1,1,0,0,1,1],
  [1,1,0,0,1,1],
  [1,1,1,1,1,1],
  [1,1,0,0,1,1],
  [1,1,0,0,1,1],
  [1,1,0,0,1,1],
  [1,1,0,0,1,1],
];

// prettier-ignore
const LETTER_R: number[][] = [
  [1,1,1,1,1,1],
  [1,1,0,0,1,1],
  [1,1,0,0,1,1],
  [1,1,1,1,1,1],
  [1,1,0,1,1,0],
  [1,1,0,0,1,1],
  [1,1,0,0,1,1],
  [1,1,0,0,1,1],
];

// prettier-ignore
const LETTER_I: number[][] = [
  [1,1,1,1],
  [0,1,1,0],
  [0,1,1,0],
  [0,1,1,0],
  [0,1,1,0],
  [0,1,1,0],
  [0,1,1,0],
  [1,1,1,1],
];

// prettier-ignore
const LETTER_S: number[][] = [
  [1,1,1,1,1,1,1,1],
  [1,1,1,1,1,1,1,1],
  [0,0,0,0,0,0,0,0],
  [1,1,1,1,1,1,0,0],
  [1,1,1,1,1,1,0,0],
  [0,0,0,0,0,0,1,1],
  [1,1,1,1,1,1,1,1],
  [1,1,1,1,1,1,1,1],
];

// prettier-ignore
const LETTER_Y: number[][] = [
  [1,1,0,0,1,1],
  [1,1,0,0,1,1],
  [0,1,1,1,1,0],
  [0,0,1,1,0,0],
  [0,0,1,1,0,0],
  [0,0,1,1,0,0],
  [0,0,1,1,0,0],
  [0,0,1,1,0,0],
];

// prettier-ignore
const LETTER_L: number[][] = [
  [1,1,0,0],
  [1,1,0,0],
  [1,1,0,0],
  [1,1,0,0],
  [1,1,0,0],
  [1,1,0,0],
  [1,1,0,0],
  [1,1,1,1],
];

// prettier-ignore
const LETTER_F: number[][] = [
  [1,1,1,1,1,1],
  [1,1,0,0,0,0],
  [1,1,0,0,0,0],
  [1,1,1,1,1,1],
  [1,1,0,0,0,0],
  [1,1,0,0,0,0],
  [1,1,0,0,0,0],
  [1,1,0,0,0,0],
];

// prettier-ignore
const LETTER_E: number[][] = [
  [1,1,1,1,1,1],
  [1,1,0,0,0,0],
  [1,1,0,0,0,0],
  [1,1,1,1,1,1],
  [1,1,0,0,0,0],
  [1,1,0,0,0,0],
  [1,1,0,0,0,0],
  [1,1,1,1,1,1],
];

// prettier-ignore
const LETTER_T: number[][] = [
  [1,1,1,1,1,1],
  [0,0,1,1,0,0],
  [0,0,1,1,0,0],
  [0,0,1,1,0,0],
  [0,0,1,1,0,0],
  [0,0,1,1,0,0],
  [0,0,1,1,0,0],
  [0,0,1,1,0,0],
];

const WORD = [
  { grid: LETTER_A, width: 6 },
  { grid: LETTER_R, width: 6 },
  { grid: LETTER_I, width: 4 },
  { grid: LETTER_S, width: 8 },
  { grid: LETTER_Y, width: 6 },
  { grid: LETTER_L, width: 4 },
  { grid: LETTER_A, width: 6 },
  { grid: LETTER_F, width: 6 },
  { grid: LETTER_E, width: 6 },
  { grid: LETTER_T, width: 6 },
  { grid: LETTER_A, width: 6 },
];

function gridToPath(grid: number[][], offsetX: number): string {
  const parts: string[] = [];
  for (let row = 0; row < grid.length; row++) {
    let col = 0;
    while (col < grid[row].length) {
      while (col < grid[row].length && grid[row][col] === 0) col++;
      if (col >= grid[row].length) break;
      const startCol = col;
      while (col < grid[row].length && grid[row][col] === 1) col++;
      const x = offsetX + startCol * UNIT;
      const y = row * UNIT;
      const w = (col - startCol) * UNIT;
      parts.push(`M${x},${y}h${w}v${UNIT}h-${w}z`);
    }
  }
  return parts.join("");
}

function buildWordmarkPath(): string {
  let offset = 0;
  const paths: string[] = [];

  for (let i = 0; i < WORD.length; i++) {
    const letter = WORD[i];
    paths.push(gridToPath(letter.grid, offset));
    offset += letter.width * UNIT;
    if (i < WORD.length - 1) {
      offset += GAP * UNIT;
    }
  }

  return paths.join("");
}

function getViewBox(): { width: number; height: number } {
  let totalWidth = 0;
  for (let i = 0; i < WORD.length; i++) {
    totalWidth += WORD[i].width * UNIT;
    if (i < WORD.length - 1) {
      totalWidth += GAP * UNIT;
    }
  }
  return { width: totalWidth, height: 8 * UNIT };
}

export function AriSylafetaWordmark(props: React.ComponentProps<"svg">) {
  const { width, height } = getViewBox();
  const path = buildWordmarkPath();

  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox={`0 0 ${width} ${height}`}
      fill="none"
      {...props}
    >
      <path fill="currentColor" d={path} />
    </svg>
  );
}

export const AbdulRehmanWordmark = AriSylafetaWordmark;

export function getWordmarkSVG(color: string) {
  const { width, height } = getViewBox();
  const path = buildWordmarkPath();
  return `<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 ${width} ${height}"><path fill="${color}" d="${path}"/></svg>`;
}
