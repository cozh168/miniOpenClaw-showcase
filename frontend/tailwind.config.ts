import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}"
  ],
  theme: {
    extend: {
      colors: {
        red: {
          50: "#fef2f2",
          600: "#dc2626",
          700: "#b91c1c"
        },
        amber: {
          500: "#f59e0b",
          700: "#b45309"
        },
        ink: "var(--color-ink)",
        sand: "var(--color-sand)",
        frost: "var(--color-frost)",
        ocean: "var(--color-ocean)",
        ember: "var(--color-ember)"
      },
      boxShadow: {
        sm: "0 1px 2px 0 rgb(0 0 0 / 0.05)",
        panel: "0 16px 60px rgba(8, 35, 44, 0.18)"
      },
      borderRadius: {
        sm: "0.125rem",
        panel: "28px"
      }
    }
  },
  plugins: []
};

export default config;
