// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  devtools: { enabled: false },
  srcDir: "src/",

  routeRules: {
    '/api/**': {
      proxy: process.env.NODE_ENV === "development" ? "http://127.0.0.1:8000/api/**" : "/api/**",
    },
    '/docs': {
      proxy: "http://127.0.0.1:8000/docs",
    },
    '/openapi.json': {
      proxy: "http://127.0.0.1:8000/openapi.json",
    }
  },
  runtimeConfig: {
    public: {
      apiBaseUrl: process.env.NUXT_PUBLIC_API_BASE_URL,
    }
  },

  compatibilityDate: "2025-03-16",
  components: [
    {
      path:'~/components',
      pathPrefix: false,
    }
  ],
  modules:['@nuxt/ui', "@nuxtjs/color-mode"],
  css:['./assets/css/main.css'],
  build: {
    standalone: true,
    parallel: false,
    cache: true,
    hardSource: false
  },
  nitro: {
    devServer: {
      // My files are under src, if yours are in the root you can change this to ./
      watch: ['./src']
    }
  },
    features: {
    devLogs: false,
    transitions: false
  }

})