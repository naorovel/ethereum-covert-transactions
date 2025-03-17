<template>
  <client-only>
    <UCard class="graph-card">
      <div ref="graphContainer" class="graph-container" />
    </UCard>
    
    <!-- Debug output -->
    <div class="p-4 text-sm">
      <pre>Processed Links: {{ processedLinks }}</pre>
    </div>
  </client-only>
</template>

<script>
export default {
  props: {
    nodes: { type: Array, required: true },
    links: { type: Array, required: true }
  },
  data() {
    return {
      d3: null,
      simulation: null,
      processedLinks: []
    }
  },
  mounted() {
    if (process.client) {
      import('d3').then(module => {
        this.d3 = module
        this.processLinks()
        this.initGraph()
      })
    }
  },
  methods: {
    processLinks() {
      // Convert link references to node objects
      const nodeMap = new Map(this.nodes.map(node => [node.id, node]))
      
      this.processedLinks = this.links.map(link => {
        const source = nodeMap.get(link.source)
        const target = nodeMap.get(link.target)
        
        if (!source || !target) {
          console.error('Invalid link:', link)
          return null
        }
        
        return {
          source: source,
          target: target
        }
      }).filter(Boolean)

      console.log('Processed links:', this.processedLinks)
    },
    initGraph() {
      const container = this.$refs.graphContainer
      if (!container) return

      container.innerHTML = ''
      const { width, height } = this.getDimensions()
      
      const svg = this.d3.select(container)
        .append('svg')
        .attr('width', width)
        .attr('height', height)
        .style('background', '#f8fafc')

      // Create simulation
      this.simulation = this.d3.forceSimulation()
        .force('charge', this.d3.forceManyBody().strength(-200))
        .force('center', this.d3.forceCenter(width / 2, height / 2))
        .force('link', this.d3.forceLink(this.processedLinks).id(d => d.id).distance(100))
        .force('collision', this.d3.forceCollide().radius(20))

      // Draw links
      const link = svg.append('g')
        .selectAll('line')
        .data(this.processedLinks)
        .join('line')
        .attr('stroke', '#3b82f6')
        .attr('stroke-width', 2)

      // Draw nodes
      const node = svg.append('g')
        .selectAll('circle')
        .data(this.nodes)
        .join('circle')
        .attr('r', 12)
        .attr('fill', '#ef4444')
        .call(this.dragHandler())

      // Simulation handler
      this.simulation.nodes(this.nodes)
        .on('tick', () => {
          link
            .attr('x1', d => d.source.x)
            .attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x)
            .attr('y2', d => d.target.y)

          node
            .attr('cx', d => d.x)
            .attr('cy', d => d.y)
        })
    },
    getDimensions() {
      return {
        width: this.$refs.graphContainer.clientWidth,
        height: 600
      }
    },
    dragHandler() {
      //if (!this.d3 || !this.simulation) return;
      console.log("Drag handler being called!")
      return this.d3.drag()
        .on('start', event => {
          console.log('Drag start', event.subject);
          if (!event.active) this.simulation.alphaTarget(0.3).restart();
          event.subject.fx = event.subject.x;
          event.subject.fy = event.subject.y;
        })
        .on('drag', event => {
          event.subject.fx = event.x;
          event.subject.fy = event.y;
          this.simulation.alpha(0.5).restart();
        })
        .on('end', event => {
          if (!event.active) this.simulation.alphaTarget(0);
          event.subject.fx = null;
          event.subject.fy = null;
        });
    }
    
  }
}
</script>

<style>
.graph-container circle {
  cursor: grab;
  transition: fill 0.2s;
}

.graph-container circle:active {
  cursor: grabbing;
  pointer-events: all;
}
</style>