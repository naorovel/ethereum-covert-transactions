<template>
  <client-only>
    <div ref="graphContainer" class="graph-container">
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
      processedLinks: [],
      zoom: null,
      zoomGroup: null
    }
  },
  mounted() {
    if (process.client) {
      import('d3').then(module => {
        this.d3 = module
        this.processLinks()
        this.initGraph()
        this.initZoom()
      })
    }
  },
  methods: {
    // Zoom methods
    initZoom() {
      const svg = this.d3.select(this.$refs.graphContainer).select('svg')

      // Initialize zoomGroup if it doesn't exist
      if (!this.zoomGroup) {
        this.zoomGroup = svg.append('g')
      }

      this.zoom = this.d3.zoom()
        .scaleExtent([0.1, 5])
        .on('zoom', (event) => {
          if (this.zoomGroup) {
            this.zoomGroup.attr('transform', event.transform)
          }
        })

      svg.call(this.zoom)
    },
    resetZoom() {
      const svg = this.d3.select(this.$refs.graphContainer).select('svg')
      svg.transition()
        .duration(250)
        .call(this.zoom.transform, this.d3.zoomIdentity)
    },
    processLinks() {
      const nodeMap = new Map(this.nodes.map(node => [node.id, node]))
      this.processedLinks = this.links.map(link => {
        const source = nodeMap.get(link.source)
        const target = nodeMap.get(link.target)
        return source && target ? { source, target } : null
      }).filter(Boolean)
    },
    getDimensions() {
      return {
        width: this.$refs.graphContainer.clientWidth,
        height: 1080
      };
    },

    initGraph() {
       const container = this.$refs.graphContainer
      if (!container) return

      container.innerHTML = ''
      const { width, height } = this.getDimensions()
      
      // Create SVG first
      const svg = this.d3.select(container)
        .append('svg')
        .attr('width', width)
        .attr('height', height)
        .style('background', '#f7ede2')

      // Initialize zoom group before creating graph elements
      this.zoomGroup = svg.append('g')

      
      // Create simulation
      this.simulation = this.d3.forceSimulation()
        .force('charge', this.d3.forceManyBody().strength(-0.00))
        .force('center', this.d3.forceCenter(width / 2, height / 2))
        .force('link', this.d3.forceLink(this.processedLinks).id(d => d.id).distance(2))
        .force('collision', this.d3.forceCollide().radius(0.5))

      // Draw links inside zoom group
      const link = this.zoomGroup
        .append('g')
        .selectAll('line')
        .data(this.processedLinks)
        .join('line')
        .attr('stroke', '#f6bd60')
        .attr('stroke-width', 1)

      // Draw nodes inside zoom group
      const node = this.zoomGroup
        .append('g')
        .selectAll('circle')
        .data(this.nodes)
        .join('circle')
        .attr('r', 1)
        .attr('fill', '#84a59d')
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

    // Modified drag handler to work with zoom
    dragHandler() {
      return this.d3.drag()
        .on('start', event => {
          if (!event.active) this.simulation.alphaTarget(0.3).restart()
          event.subject.fx = event.x
          event.subject.fy = event.y
        })
        .on('drag', event => {
          const [x, y] = this.d3.pointer(event, this.zoomGroup.node())
          event.subject.fx = x
          event.subject.fy = y
          this.simulation.alpha(0.5).restart()
        })
        .on('end', event => {
          if (!event.active) this.simulation.alphaTarget(0)
          event.subject.fx = null
          event.subject.fy = null
        })
    }
  }
}
</script>

<style>
.reset-zoom {
  position: absolute;
  top: 10px;
  left: 10px;
  z-index: 100;
  padding: 5px 10px;
  cursor: pointer;
  background: white;
  border: 1px solid #ccc;
  border-radius: 4px;
}
</style>