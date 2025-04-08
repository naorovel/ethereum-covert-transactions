<template>
  <div>
    <div v-if="pending && !graphData">Loading initial data...</div>
    <div v-else-if="error">
      <p>Error: {{ error }}</p>
      <button @click="retryFetch">Retry</button>
    </div>
    <div v-else-if="graphData">
      <div v-if="refreshing" class="refresh-indicator">Refreshing...</div>
      <ForceGraph
        :nodes="graphData.nodes"
        :links="graphData.links"
      />
    </div>
    <div v-else>No data available</div>
  </div>
</template>

<script setup lang="ts">
import { onMounted, onUnmounted, ref } from 'vue'

type Node = {
  id: string
}

type Link = {
  source: string,
  target: string
}

type GraphData = {
  nodes: Node[],
  links: Link[]
}

// Define API URLs first
const SETUP_API_URL = 'http://localhost:8000/load_init_graph_transactions?num_transactions=1000'
const API_URL = 'http://localhost:8000/get_graph_transactions'

// Declare graph data ref to store the fetched data
const graphData = ref<GraphData | null>(null)
const error = ref<string | null>(null)
const refreshing = ref(false)

// Set up useFetch with immediate: false to manually control fetching
const { data, pending, execute } = useFetch<GraphData>(API_URL, {
  headers: { 'Accept': '*/*' },
  transform: (res: any) => ({
    nodes: res.nodes.map((node: Node) => ({ id: node.id })),
    links: res.links.map((link: Link) => ({
      source: link.source,
      target: link.target
    }))
  }),
  immediate: false // Prevent automatic fetch on component creation
})

// Declare interval variable with proper type
let intervalId: number | undefined

// Function to call our API and refresh the data
const fetchData = async () => {
  console.log('Fetching graph data...')
  error.value = null
  refreshing.value = true
  try {
    await execute()
    if (data.value) {
      console.log('Fetched data:', data)
      console.log('Move fetched data to graphData ...')
      graphData.value = data.value
      console.log('Current data:', graphData)
    }
  } catch (err) {
    const errorMessage = err instanceof Error ? err.message : 'Unknown error'
    console.error('Error fetching graph data:', errorMessage)
    error.value = `Failed to fetch graph data: ${errorMessage}`
  } finally {
    refreshing.value = false
  }
}

// Function to retry fetching after an error
const retryFetch = () => {
  error.value = null
  if (graphData.value) {
    // We already have some data, just refresh it
    fetchData()
  } else {
    // We need to set up initial data first
    setupInitialData()
  }
}

// Optional function to call the setup API if needed
const setupInitialData = async () => {
  error.value = null
  try {
    console.log('Setting up initial graph data...')
    const response = await fetch(SETUP_API_URL)
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`)
    }
    const result = await response.json()
    console.log('Setup complete:', result)
    // Now fetch the graph data
    await fetchData()
  } catch (err) {
    const errorMessage = err instanceof Error ? err.message : 'Unknown error'
    console.error('Error setting up initial data:', errorMessage)
    error.value = `Failed to set up initial data: ${errorMessage}`
  }
}

onMounted(() => {
  // Call setupInitialData on mount (if you want to initialize first)
  setupInitialData()
  // Set up the interval to call fetchData every second
  intervalId = window.setInterval(() => {
    // Only fetch new data if we're not currently refreshing
    // and there's no error
    if (!refreshing.value && !error.value) {
      fetchData()
    }
  }, 3000)
})

onUnmounted(() => {
  // Clean up the interval when the component is unmounted
  if (intervalId !== undefined) {
    clearInterval(intervalId)
  }
})
</script>

<style scoped>
.refresh-indicator {
  position: absolute;
  top: 10px;
  right: 10px;
  background-color: rgba(0, 0, 0, 0.6);
  color: white;
  padding: 5px 10px;
  border-radius: 4px;
}
</style>

