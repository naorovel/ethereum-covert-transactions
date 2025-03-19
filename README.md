## Introduction

Visualization of Ethereum transaction data using interactive network graph structures.


## Deployed Demo (not fully functional yet)

https://ethereum-covert-transactions.vercel.app/


## Getting Started (Locally)

First, install the dependencies (Note: you must have Node.js and npm installed):

```bash
npm install
# or
yarn
# or
pnpm install
```

Make sure you have pip3 working.
Then, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

The FastApi server will be running on [http://127.0.0.1:8000](http://127.0.0.1:8000) 


## How It Works

The Python/FastAPI server is mapped into to Nuxt.js app under `/api/`.

The server API of Nuxt3 has been relocated to `/backend/` to make it compatible with the Vercel API routes.

This is implemented using [`nuxt.config.js` rewrites](https://github.com/tutorfx/nuxtjs-fastapi/blob/main/nuxt.config.ts) to map any request to `/api/:path*` to the FastAPI API, which is hosted in the `/api` folder.

On localhost, the rewrite will be made to the `127.0.0.1:8000` port, which is where the FastAPI server is running.

In production, the FastAPI server is hosted as [Python serverless functions](https://vercel.com/docs/concepts/functions/serverless-functions/runtimes/python) on Vercel.




