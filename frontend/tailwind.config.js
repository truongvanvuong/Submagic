/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ['selector', '[data-theme="studio-dark"]'],
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: [
          '"Plus Jakarta Sans"',
          'system-ui',
          '-apple-system',
          'BlinkMacSystemFont',
          '"Segoe UI"',
          'Roboto',
          '"Noto Sans SC"',
          '"PingFang SC"',
          '"Microsoft YaHei"',
          'sans-serif',
        ],
        mono: [
          '"JetBrains Mono"',
          'ui-monospace',
          'SFMono-Regular',
          'Menlo',
          'Monaco',
          'Consolas',
          'monospace',
        ],
      },
      colors: {
        parchment: {
          DEFAULT: '#edede9',
          100: '#34342a',
          200: '#676754',
          300: '#989881',
          400: '#c2c2b5',
          500: '#edede9',
          600: '#f0f0ed',
          700: '#f4f4f2',
          800: '#f8f8f6',
          900: '#fbfbfb',
        },
        dust_grey: {
          DEFAULT: '#d6ccc2',
          100: '#312921',
          200: '#625241',
          300: '#937a62',
          400: '#b6a391',
          500: '#d6ccc2',
          600: '#ded6ce',
          700: '#e7e0da',
          800: '#efebe7',
          900: '#f7f5f3',
        },
        linen: {
          DEFAULT: '#f5ebe0',
          100: '#473017',
          200: '#8e5f2e',
          300: '#c78f52',
          400: '#debd99',
          500: '#f5ebe0',
          600: '#f7efe6',
          700: '#f9f3ed',
          800: '#fbf7f3',
          900: '#fdfbf9',
        },
        powder_petal: {
          DEFAULT: '#e3d5ca',
          100: '#38291e',
          200: '#70523b',
          300: '#a87b59',
          400: '#c5a891',
          500: '#e3d5ca',
          600: '#e8ddd4',
          700: '#eee6df',
          800: '#f4eeea',
          900: '#f9f7f4',
        },
        almond_silk: {
          DEFAULT: '#d5bdaf',
          100: '#33241b',
          200: '#664735',
          300: '#986b50',
          400: '#ba937c',
          500: '#d5bdaf',
          600: '#ddcabf',
          700: '#e6d7cf',
          800: '#eee4df',
          900: '#f7f2ef',
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
          "primary": "#c78f52",         // linen 300 (warm caramel accent)
          "primary-content": "#1f1b17", // deep espresso
          "secondary": "#d5bdaf",       // almond_silk 500
          "secondary-content": "#33241b",
          "accent": "#debd99",          // linen 400
          "accent-content": "#312921",
          "neutral": "#34342a",         // parchment 100
          "neutral-content": "#edede9", // parchment 500
          "base-100": "#1f1b17",        // deep warm dark background
          "base-200": "#2b251f",        // elevated panel
          "base-300": "#38291e",        // powder_petal 100 (cards/toolbars)
          "base-content": "#f5ebe0",    // linen 500 (high legibility text)
          "info": "#b6a391",            // dust_grey 400
          "info-content": "#312921",
          "success": "#52c41a",
          "success-content": "#1f1b17",
          "warning": "#c78f52",         // linen 300
          "warning-content": "#1f1b17",
          "error": "#ef4444",
          "error-content": "#fbfbfb",
        },
        'cool-sky': {
          "primary": "#8e5f2e",         // linen 200 (rich bronze, >5.3:1 contrast)
          "primary-content": "#fdfbf9", // linen 900
          "secondary": "#664735",       // almond_silk 200
          "secondary-content": "#f7f2ef",
          "accent": "#a87b59",          // powder_petal 300
          "accent-content": "#f9f7f4",
          "neutral": "#34342a",         // parchment 100
          "neutral-content": "#edede9",
          "base-100": "#fbfbfb",        // parchment 900 (crisp light background)
          "base-200": "#f5ebe0",        // linen 500 (warm panel)
          "base-300": "#e3d5ca",        // powder_petal 500 (toolbars/inputs)
          "base-content": "#312921",    // dust_grey 100 (high contrast text)
          "info": "#625241",            // dust_grey 200
          "info-content": "#f7f5f3",
          "success": "#389e0d",
          "success-content": "#fbfbfb",
          "warning": "#a87b59",         // powder_petal 300
          "warning-content": "#f9f7f4",
          "error": "#cf1322",
          "error-content": "#fbfbfb",
        },
        'warm-linen': {
          "primary": "#8e5f2e",
          "primary-content": "#fdfbf9",
          "secondary": "#664735",
          "secondary-content": "#f7f2ef",
          "accent": "#a87b59",
          "accent-content": "#f9f7f4",
          "neutral": "#34342a",
          "neutral-content": "#edede9",
          "base-100": "#fbfbfb",
          "base-200": "#f5ebe0",
          "base-300": "#e3d5ca",
          "base-content": "#312921",
          "info": "#625241",
          "info-content": "#f7f5f3",
          "success": "#389e0d",
          "success-content": "#fbfbfb",
          "warning": "#a87b59",
          "warning-content": "#f9f7f4",
          "error": "#cf1322",
          "error-content": "#fbfbfb",
        },
        'warm-parchment': {
          "primary": "#8e5f2e",
          "primary-content": "#fdfbf9",
          "secondary": "#664735",
          "secondary-content": "#f7f2ef",
          "accent": "#a87b59",
          "accent-content": "#f9f7f4",
          "neutral": "#34342a",
          "neutral-content": "#edede9",
          "base-100": "#fbfbfb",
          "base-200": "#f5ebe0",
          "base-300": "#e3d5ca",
          "base-content": "#312921",
          "info": "#625241",
          "info-content": "#f7f5f3",
          "success": "#389e0d",
          "success-content": "#fbfbfb",
          "warning": "#a87b59",
          "warning-content": "#f9f7f4",
          "error": "#cf1322",
          "error-content": "#fbfbfb",
        },
      },
    ],
    darkTheme: "studio-dark",
    base: true,
    styled: true,
    utils: true,
  },
}
