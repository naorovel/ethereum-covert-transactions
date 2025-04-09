<template>
  <div>
    <div v-if="pending">Loading initial data...</div>
    <div v-else-if="data">
      <ForceGraph
        :nodes="data.nodes"
        :links="data.links"
      />
    </div>
    <div v-else>No data available</div>
  </div>
</template>

<script setup lang="ts">

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
const API_URL = 'http://localhost:8000/get_graph_transactions'


// Set up useFetch with immediate: false to manually control fetching
const { data, pending, error } = useFetch<GraphData>(API_URL, {
  headers: { 'Accept': '*/*' },
  transform: (res: any) => {
    console.log(res)
    return {
      nodes: res.nodes.map((node: Node) => ({ id: node.id })),
      links: res.links.map((link: Link) => ({
        source: link.source,
        target: link.target
      }))
    }
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

