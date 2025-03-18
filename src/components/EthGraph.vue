<template>
  <div>
    <div v-if="pending">Loading...</div>
    <div v-else-if="data">
      <!-- <pre>{{ data }}</pre> -->
      <!-- Display structured data -->
      <!-- <div v-for="node in data.nodes" :key="node.id">
        Node: {{ node.id }}
      </div> -->
      <ForceGraph
      :nodes="data.nodes"
      :links="data.links"/>
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

const API_URL = 'http://localhost:8000/get_graph_transactions?num_transactions=10'

const { data, pending } = useFetch<GraphData>(API_URL, { // Note changed to GraphData (not array)
  headers: { 'Accept': '*/*' },
  transform: (res: any) => ({
    nodes: res.nodes.map((node: Node) => ({ id: node.id })),
    links: res.links.map((link: Link) => ({ 
      source: link.source, 
      target: link.target 
    }))
  })
})

</script>