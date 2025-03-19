# Visualization of Ethereum transaction data using interactive network graph structures.

## Introduction




## Demo

https://ethereum-covert-transactions.vercel.app/


## Dataset

We have used the following dataset for the demo:

https://github.com/salam-ammari/Labeled-Transactions-based-Dataset-of-Ethereum-Network

The dataset utilized by this project is the Benchmark Labeled Transactions of Ethereum Network (BLTE) dataset, which is designed specifically to perform blockchain security research.

The dataset is collected from the Ethereum Classic (ETC) network, a public, and open-source blockchain platform. It overcomes the limitations of previous blockchain datasets by ensuring a well-defined transaction representation and including labelled data to improve detection accuracy.

By leveraging the BLTE dataset, our project aims to visualize Ethereum transactions and explore suspicious transaction patterns to enhance blockchain security analysis through further expansions to this visualization.

More detailed description can be found in their repo.


## Developing Locally

You can clone & create this repo with the following command

```bash
npx create-nuxt-app nuxtjs-fastapi --example "https://github.com/naorovel/ethereum-covert-transactions.git"
```

## Getting Started

First, install the dependencies:

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




