/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./**/*.{html,js}",
  ],
  theme: {
    extend: {
      colors: {
        'dark-background': '#0C0A09',
        'dark-surface': '#1C1917',
        'dark-border': '#292524',
        'dark-text': '#E7E5E4',
        'dark-muted': '#A8A29E',
        'dark-primary': '#4F46E5', // Indigo-600
        'dark-primary-hover': '#4338CA', // Indigo-700
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
        heading: ['Plus Jakarta Sans', 'sans-serif'],
      },
      animation: {
        'pulse-fast': 'pulse 1.2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'spin-fast': 'spin 0.8s linear infinite',
      },
    },
  },
  plugins: [],
}