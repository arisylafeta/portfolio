import { USER } from "@/features/profile/data/user";
import type { NavItem } from "@/types/nav";

export const SITE_INFO = {
  name: USER.displayName,
  url: process.env.APP_URL || "https://arisylafeta.com",
  ogImage: USER.ogImage,
  description: USER.bio,
  keywords: USER.keywords,
};

export const META_THEME_COLORS = {
  light: "#ffffff",
  dark: "#09090b",
};

export const MAIN_NAV: NavItem[] = [
  {
    title: "Portfolio",
    href: "/",
  },
  {
    title: "Blog",
    href: "/blog",
  },
  {
    title: "Products",
    href: "/products",
  },
  // {
  //   title: "Components",
  //   href: "/components",
  // },
];

export const GITHUB_USERNAME = "arisylafeta";
export const SOURCE_CODE_GITHUB_REPO = "arisylafeta/resume-2025";
export const SOURCE_CODE_GITHUB_URL =
  "https://github.com/arisylafeta/resume-2025";

export const UTM_PARAMS = {
  utm_source: "https://arisylafeta.com",
  utm_medium: "portfolio_website",
  utm_campaign: "referral",
};
