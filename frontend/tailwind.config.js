/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['"Plus Jakarta Sans"', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'Monaco', 'Consolas', 'monospace'],
      },
      colors: {
        blue_slate: {
          DEFAULT: '#495867',
          100: '#0f1215',
          200: '#1e242a',
          300: '#2c363f',
          400: '#3b4754',
          500: '#495867',
          600: '#667a8f',
          700: '#8b9cac',
          800: '#b1bdc8',
          900: '#d8dee3',
        },
        glaucous: {
          DEFAULT: '#577399',
          100: '#11171e',
          200: '#232e3d',
          300: '#34455b',
          400: '#465c7a',
          500: '#577399',
          600: '#768eb0',
          700: '#98aac4',
          800: '#bac7d8',
          900: '#dde3eb',
        },
        pale_sky: {
          DEFAULT: '#bdd5ea',
          100: '#142c40',
          200: '#295881',
          300: '#3d83c1',
          400: '#7dacd6',
          500: '#bdd5ea',
          600: '#cadeee',
          700: '#d7e6f3',
          800: '#e5eef7',
          900: '#f2f7fb',
        },
        ghost_white: {
          DEFAULT: '#f7f7ff',
          100: '#000064',
          200: '#0000c8',
          300: '#2d2dff',
          400: '#9191ff',
          500: '#f7f7ff',
          600: '#f7f7ff',
          700: '#f9f9ff',
          800: '#fbfbff',
          900: '#fdfdff',
        },
      },
    },
  },
  plugins: [
    require('daisyui'),
  ],
  daisyui: {
    themes: [
      {
        'studio-dark': {
          "primary": "#577399",         // glaucous DEFAULT
          "primary-content": "#f7f7ff", // ghost_white DEFAULT
          "secondary": "#7dacd6",       // pale_sky 400
          "secondary-content": "#0f1215",
          "accent": "#bdd5ea",          // pale_sky DEFAULT
          "accent-content": "#0f1215",
          "neutral": "#1e242a",         // blue_slate 200
          "neutral-content": "#f7f7ff", // ghost_white DEFAULT
          "base-100": "#0f1215",        // blue_slate 100 (deep dark slate background)
          "base-200": "#1e242a",        // blue_slate 200 (panel background)
          "base-300": "#2c363f",        // blue_slate 300 (cards, inputs)
          "base-content": "#f7f7ff",    // ghost_white DEFAULT (high legibility text)
          "info": "#768eb0",            // glaucous 600
          "info-content": "#f7f7ff",
          "success": "#52c41a",
          "success-content": "#f7f7ff",
          "warning": "#faad14",
          "warning-content": "#0f1215",
          "error": "#ff4d4f",
          "error-content": "#f7f7ff",
        },
        'cool-sky': {
          "primary": "#577399",         // glaucous DEFAULT
          "primary-content": "#f7f7ff",
          "secondary": "#3d83c1",       // pale_sky 300
          "secondary-content": "#f7f7ff",
          "accent": "#295881",          // pale_sky 200
          "accent-content": "#f7f7ff",
          "neutral": "#2c363f",         // blue_slate 300
          "neutral-content": "#f7f7ff",
          "base-100": "#f7f7ff",        // ghost_white DEFAULT (clean crisp light)
          "base-200": "#f2f7fb",        // pale_sky 900
          "base-300": "#d8dee3",        // blue_slate 900
          "base-content": "#0f1215",    // blue_slate 100
          "info": "#577399",
          "info-content": "#f7f7ff",
          "success": "#389e0d",
          "success-content": "#f7f7ff",
          "warning": "#d48806",
          "warning-content": "#0f1215",
          "error": "#cf1322",
          "error-content": "#f7f7ff",
        },
        // Backward compatibility theme aliases
        'warm-linen': {
          "primary": "#577399",
          "primary-content": "#f7f7ff",
          "secondary": "#3d83c1",
          "secondary-content": "#f7f7ff",
          "accent": "#295881",
          "accent-content": "#f7f7ff",
          "neutral": "#2c363f",
          "neutral-content": "#f7f7ff",
          "base-100": "#f7f7ff",
          "base-200": "#f2f7fb",
          "base-300": "#d8dee3",
          "base-content": "#0f1215",
          "info": "#577399",
          "info-content": "#f7f7ff",
          "success": "#389e0d",
          "success-content": "#f7f7ff",
          "warning": "#d48806",
          "warning-content": "#0f1215",
          "error": "#cf1322",
          "error-content": "#f7f7ff",
        },
        'warm-parchment': {
          "primary": "#577399",
          "primary-content": "#f7f7ff",
          "secondary": "#3d83c1",
          "secondary-content": "#f7f7ff",
          "accent": "#295881",
          "accent-content": "#f7f7ff",
          "neutral": "#2c363f",
          "neutral-content": "#f7f7ff",
          "base-100": "#f7f7ff",
          "base-200": "#f2f7fb",
          "base-300": "#d8dee3",
          "base-content": "#0f1215",
          "info": "#577399",
          "info-content": "#f7f7ff",
          "success": "#389e0d",
          "success-content": "#f7f7ff",
          "warning": "#d48806",
          "warning-content": "#0f1215",
          "error": "#cf1322",
          "error-content": "#f7f7ff",
        },
      },
    ],
    darkTheme: "studio-dark",
    base: true,
    styled: true,
    utils: true,
  },
}
