/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'light-bronze': {
          50: '#f8f1ed', 100: '#f0e2db', 200: '#e2c5b6', 300: '#d3a992', 400: '#c58c6d',
          500: '#b66f49', 600: '#92593a', 700: '#6d432c', 800: '#492c1d', 900: '#24160f', 950: '#1a100a'
        },
        'desert-sand': {
          50: '#f8f1ed', 100: '#f0e3db', 200: '#e2c8b6', 300: '#d3ac92', 400: '#c5906d',
          500: '#b67549', 600: '#925d3a', 700: '#6d462c', 800: '#492f1d', 900: '#24170f', 950: '#1a100a'
        },
        'powder-petal': {
          50: '#fff1e5', 100: '#ffe2cc', 200: '#ffc599', 300: '#ffa866', 400: '#ff8b33',
          500: '#ff6e00', 600: '#cc5800', 700: '#994200', 800: '#662c00', 900: '#331600', 950: '#240f00'
        },
        'ash-grey': {
          50: '#f4f4f1', 100: '#e9e9e2', 200: '#d2d2c6', 300: '#bcbca9', 400: '#a5a58d',
          500: '#8f8f70', 600: '#72725a', 700: '#565643', 800: '#39392d', 900: '#1d1d16', 950: '#141410'
        },
        'dry-sage': {
          50: '#f4f4f1', 100: '#e9e9e2', 200: '#d2d2c6', 300: '#bcbca9', 400: '#a5a58d',
          500: '#8f8f70', 600: '#72725a', 700: '#565643', 800: '#39392d', 900: '#1d1d16', 950: '#141410'
        },
        'dusty-olive': {
          50: '#f3f4f1', 100: '#e7e8e3', 200: '#cfd1c7', 300: '#b6baab', 400: '#9ea38f',
          500: '#868c73', 600: '#6b705c', 700: '#505445', 800: '#36382e', 900: '#1b1c17', 950: '#131410'
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
        bronze: {
          "primary": "#92593a",
          "primary-content": "#f8f1ed",
          "secondary": "#505445",
          "secondary-content": "#f3f4f1",
          "accent": "#ff6e00",
          "accent-content": "#240f00",
          "neutral": "#39392d",
          "neutral-content": "#e9e9e2",
          "base-100": "#f4f4f1",
          "base-200": "#e9e9e2",
          "base-300": "#d2d2c6",
          "base-content": "#1d1d16",
          "info": "#6b705c",
          "info-content": "#f3f4f1",
          "success": "#565643",
          "success-content": "#f4f4f1",
          "warning": "#ff8b33",
          "warning-content": "#240f00",
          "error": "#b3412e",
          "error-content": "#ffffff",
        },
        "bronze-dark": {
          "primary": "#c58c6d",
          "primary-content": "#1a100a",
          "secondary": "#9ea38f",
          "secondary-content": "#131410",
          "accent": "#ff6e00",
          "accent-content": "#240f00",
          "neutral": "#d2d2c6",
          "neutral-content": "#1d1d16",
          "base-100": "#1d1d16",
          "base-200": "#141410",
          "base-300": "#39392d",
          "base-content": "#e9e9e2",
          "info": "#b6baab",
          "info-content": "#131410",
          "success": "#a5a58d",
          "success-content": "#141410",
          "warning": "#ff8b33",
          "warning-content": "#240f00",
          "error": "#e05238",
          "error-content": "#ffffff",
        },
      },
    ],
    darkTheme: "bronze-dark",
    base: true,
    styled: true,
    utils: true,
  },
}
