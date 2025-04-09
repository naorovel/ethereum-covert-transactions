<template>
  <client-only>
    <div class="force-graph-container">
      <!-- Filter Controls -->
       
      <div class="filter-controls">
        <div class="dropdown">
          <button class="dropdown-toggle">Filter Links ▼</button>
          <div class="dropdown-menu">
            <div v-for="(values, attr) in possibleFilters" :key="attr">
              <div class="filter-group">
                <h6>{{ attr }}</h6>
                <div v-for="value in values" :key="value">
                  <label>
                    <input
                      type="checkbox"
                      :value="value"
                      v-model="selectedFilters[attr]"
                    />
                    {{ value }}
                  </label>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Color Legend -->
      <div class="color-legend">
        <div class="legend-item">
          <span class="legend-color" style="background-color: #2a9d8f"></span>
            Correctly classified
        </div>
        <div class="legend-item">
          <span class="legend-color" style="background-color: #e76f51"></span>
            Incorrectly classified
        </div>
      </div>

      <!-- Graph Container -->
      <div ref="graphContainer" class="graph-container"></div>
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
      zoomGroup: null,
      selectedFilters: {
        covert: [],
        covert_generated: [],
        type: []
      },
      tooltip: null
    }
  },
  computed: {
    filteredLinks() {
      return this.links.filter(link => {
        const matchesCovert = this.selectedFilters.covert.length === 0 || 
          this.selectedFilters.covert.includes(link.covert)
        const matchesCovertGen = this.selectedFilters.covert_generated.length === 0 || 
          this.selectedFilters.covert_generated.includes(link.covert_generated)
        const matchesType = this.selectedFilters.type.length === 0 || 
          this.selectedFilters.type.includes(link.type)
        return matchesCovert && matchesCovertGen && matchesType
      })
    },
    possibleFilters() {
      const types = new Set()
      this.links.forEach(link => types.add(link.type))
      return {
        covert: [true, false],
        covert_generated: [true, false],
        type: Array.from(types)
      }
    }
  },
  watch: {
    filteredLinks: {
      handler() {
        this.processLinks()
        if (this.d3) this.initGraph()
      },
      deep: true
    }
  },
  mounted() {
    if (process.client) {
      import('d3').then(module => {
        this.d3 = module
        this.tooltip = this.d3.select('body')
          .append('div')
          .attr('class', 'tooltip')
          .style('position', 'absolute')
          .style('background', '#fff')
          .style('padding', '5px')
          .style('border', '1px solid #ccc')
          .style('opacity', 0)
        this.processLinks()
        this.initGraph()
        this.initZoom()
      })
    }
  },
  methods: {
    initZoom() {
      const svg = this.d3.select(this.$refs.graphContainer).select('svg')
      if (!this.zoomGroup) this.zoomGroup = svg.append('g')

      this.zoom = this.d3.zoom()
        .scaleExtent([0.1, 20])
        .on('zoom', (event) => {
          this.zoomGroup.attr('transform', event.transform)
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
      this.processedLinks = this.filteredLinks
        .map(link => {
          const source = nodeMap.get(link.source)
          const target = nodeMap.get(link.target)
          return source && target ? { ...link, source, target } : null
        })
        .filter(Boolean)
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
        .style('background', '#f7ede2')

      this.zoomGroup = svg.append('g')
      this.simulation = this.d3.forceSimulation()
        .force('charge', this.d3.forceManyBody().strength(-0.0))
        .force('center', this.d3.forceCenter(width / 2, height / 2))
        .force('link', this.d3.forceLink(this.processedLinks).id(d => d.id).distance(0.0001))
        .force('collision', this.d3.forceCollide().radius(0.0001))

      // Links
      const link = this.zoomGroup.append('g')
        .selectAll('line')
        .data(this.processedLinks)
        .join('line')
        .attr('stroke', d => d.covert === d.covert_generated ? '#2a9d8f' : '#e76f51')
        .attr('stroke-width', 0.5)
        .on('click', (event, d) => this.$emit('link-selected', d))

      // Nodes
      const node = this.zoomGroup.append('g')
        .selectAll('circle')
        .data(this.nodes)
        .join('circle')
        .attr('r', 1)
        .attr('fill', '#264653')
        .on('mouseover', (event, d) => {
          this.tooltip
            .style('left', `${event.pageX + 10}px`)
            .style('top', `${event.pageY + 10}px`)
            .style('opacity', 1)
            .text(d.id)
        })
        .on('mouseout', () => this.tooltip.style('opacity', 0))
        .call(this.dragHandler())

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
    dragHandler() {
      return this.d3.drag()
        .on('start', event => {
          // if (!event.active) this.simulation.alphaTarget(0.3).restart()
          event.subject.fx = event.x
          event.subject.fy = event.y
        })
        .on('drag', event => {
          const [x, y] = this.d3.pointer(event, this.zoomGroup.node())
          event.subject.fx = x
          event.subject.fy = y
          // this.simulation.alpha(0.5).restart()
        })
        .on('end', event => {
          if (!event.active) this.simulation.alphaTarget(0)
          event.subject.fx = null
          event.subject.fy = null
        })
    },
    getDimensions() {
      return {
        width: this.$refs.graphContainer?.clientWidth || 800,
        height: 1080
      }
    }
  }
}
</script>

<style>
.force-graph-container {
  position: relative;
  width: 100%;
  height: 100%;
}

.filter-controls {
  position: absolute;
  top: 10px;
  left: 10px;
  z-index: 100;
  background: white;
  padding: 10px;
  border-radius: 4px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.dropdown-toggle {
  padding: 5px 10px;
  background: #fff;
  border: 1px solid #ccc;
  border-radius: 4px;
  cursor: pointer;
}

.dropdown-menu {
  display: none;
  position: absolute;
  background: white;
  border: 1px solid #ccc;
  padding: 10px;
  margin-top: 5px;
  border-radius: 4px;
  min-width: 200px;
}

.dropdown:hover .dropdown-menu {
  display: block;
}

.filter-group {
  margin-bottom: 10px;
}

.filter-group h6 {
  margin: 0 0 5px 0;
  color: #333;
  font-size: 0.9em;
}

.filter-group label {
  display: block;
  font-size: 0.8em;
  margin: 2px 0;
}

.color-legend {
  position: absolute;
  top: 10px;
  right: 10px;
  background: white;
  padding: 10px;
  border: 1px solid #ccc;
  border-radius: 4px;
  z-index: 100;
}

.legend-item {
  display: flex;
  align-items: center;
  margin-bottom: 5px;
  font-size: 0.8em;
}

.legend-color {
  width: 16px;
  height: 16px;
  margin-right: 8px;
  border-radius: 3px;
}

.graph-container {
  width: 100%;
  height: 1080px;
}

.tooltip {
  pointer-events: none;
  font-size: 0.8em;
}
</style>