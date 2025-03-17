<template>
  <div class="p-4">
    <ForceGraph
      :nodes="nodes"
      :links="links"
    />
  </div>
</template>

<script>
export default {
  data() {
    return {
      // Nodes must have unique IDs
        nodes: [
        { id: 'A', x: 100, y: 100 },
        { id: 'B', x: 200, y: 200 }
        ],
        links: [{ source: 'A', target: 'B' }]
    }
  }
}
</script>