export function AriSylafetaMark(props: React.ComponentProps<"svg">) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      fill="none"
      viewBox="0 0 512 256"
      {...props}
    >
      <path
        fill="currentColor"
        d="
            M0 0h64v256H0V0Z
            M128 0h64v256h-64V0Z
            M0 0h192v64H0V0Z
            M0 96h192v64H0V96Z
            M256 0h256v64H256V0Z
            M256 96h192v64H256V96Z
            M448 160h64v32h-64v-32Z
            M256 192h256v64H256v-64Z
          "
      ></path>
    </svg>
  );
}

export const AbdulRehmanMark = AriSylafetaMark;

export function getMarkSVG(color: string) {
  return `<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 512 256"><path fill="${color}" d="M0 0h64v256H0V0ZM128 0h64v256h-64V0ZM0 0h192v64H0V0ZM0 96h192v64H0V96ZM256 0h256v64H256V0ZM256 96h192v64H256V96ZM448 160h64v32h-64v-32ZM256 192h256v64H256v-64Z"/></svg>`;
}
