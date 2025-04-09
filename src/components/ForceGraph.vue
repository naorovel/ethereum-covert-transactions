<template>
  <client-only>
    <div class="graph-wrapper">

      <div class="filter-panel">
        <h4>Filters</h4>
        
        <!-- Existing bias filter -->
        <!-- <div class="filter-group">
          <label>Bias Types:</label>
          <select v-model="selectedBiases" multiple class="filter-select">
             <option 
                v-for="bias in allBiasTypes" 
                :key="bias" 
                :value="bias"
                :style="{ color: biasColorScale(bias) }"
              >
                █ {{ formatBiasName(bias) }}
              </option>
          </select>
        </div> -->

        <!-- New filters -->
        <!-- <div class="filter-group">
          <label>Detection Classification:</label>
          <select v-model="selectedCovert" multiple class="filter-select">
            <option v-for="alg in allCovert" :key="alg" :value="alg">
              {{ alg }}
            </option>
          </select>
        </div>

        <div class="filter-group">
          <label>Generated Covert:</label>
          <select v-model="selectedCovertGenerated" multiple class="filter-select">
            <option v-for="src in allCovertGenerated" :key="src" :value="src">
              {{ src }}
            </option>
          </select>
        </div> -->

        <div class="filter-group">
          <label>Transaction Types:</label>
          <select v-model="selectedTypes" multiple class="filter-select">
            <option v-for="type in allTypes" :key="type" :value="type">
              {{ type }}
            </option>
          </select>
        </div>
      </div>
      <div class="legend">
        <h4>Detection Color Legend</h4>
        <div class="legend-item">
          <div class="legend-color" style="background-color: #2a9d8f"></div>
          <div class="legend-label">Non-covert Links</div>
        </div>
        <div class="legend-item">
          <div class="legend-color" style="background-color: #e76f51"></div>
          <div class="legend-label">Covert Links</div>
        </div>
        <!-- <div class="legend-item">
          <div class="legend-color" style="background-color: #264653"></div>
          <div class="legend-label">Other Links</div>
        </div> -->
      </div>

      <div class="side-menu">
        <div class="menu-header">
          <h2>
            {{ selectedNode ? 'Node Details' : 'Connection Details' }}
          </h2>
          <button @click="clearSelection">×</button>
        </div>
        <div class="menu-content">
          <!-- Node Details -->
          <div v-if="selectedNode">
            <p>ID: {{ selectedNode.id }}</p>
            <p>Connections: {{ selectedNode.connections?.length || 0 }}</p>
            <pre>{{ selectedNode }}</pre>
          </div>
          
          <!-- Connection Details -->
          <div v-if="selectedConnection">
            <h3>Between:</h3>
            <ul>
              <li>{{ selectedConnection.nodes[0].id }}</li>
              <li>{{ selectedConnection.nodes[1].id }}</li>
            </ul>
            
            <h4>All Links ({{ selectedConnection.links.length }}):</h4>
            <div 
              v-for="(link, index) in selectedConnection.links" 
              :key="index"
              class="link-item"
            >
              <div class="link-properties">
                <div v-if="link.type">
                  Type: <strong>{{ link.type }}</strong>
                </div>
              </div>
              <!-- Optional: Keep raw data view -->
              <pre v-if="showRawLinkData">{{ link }}</pre>
            </div>
            <!-- <div 
              v-for="(link, index) in selectedConnection.links" 
              :key="index"
              class="link-item"
            >
              <pre>{{ link }}</pre>
            </div> -->
          </div>
        </div>
      </div>



      <div ref="graphContainer" class="graph-container" @click="handleBackgroundClick">
        <div 
          v-if="hoveredElement" 
          class="tooltip"
          :style="{
            left: mousePosition.x + 10 + 'px',
            top: mousePosition.y + 10 + 'px'
          }"
        >
          <div v-if="hoveredElement.type === 'node'">
            Node: {{ hoveredElement.data.id }}
          </div>
          <div v-else>
            Link: {{ hoveredElement.data.source.id }} → {{ hoveredElement.data.target.id }}
          </div>
        </div>
      </div>
    </div>
    
  </client-only>
</template>


<script>

import * as d3 from 'd3';
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
      hoveredElement: null,
      mousePosition: { x: 0, y: 0 },
      selectedNode: null,
      selectedConnection: null,
      selectedCovert: [],
      selectedCovertGenerated: [],
      selectedTypes: [],
      showRawLinkData: false,
      filteredNodes: [],

    }
  },
  computed: {
    filteredLinks() {
      return this.links.filter(link => {
        // Bias filter

        // Algorithm filter
        const hasCovert = this.selectedCovert.length === 0 ||
                            (link.covert && 
                            this.selectedCovert.some(a => 
                              link.covert === a || // Handle string
                              link.covert.includes(a) // Handle array
                            ));

        // Source filter (exact match)
        const hasCovertGenerated = this.selectedCovertGenerated.length === 0 ||
                        (link.covert_generated && 
                          this.selectedCovertGenerated.includes(link.covert_generated));

        // Type filter (exact match)
        const hasType = this.selectedTypes.length === 0 ||
                      (link.type && 
                        this.selectedTypes.includes(link.type));

        return hasCovert && hasCovertGenerated && hasType;
      });
    },
      allTypes() {
      return [...new Set(this.links.flatMap(l => l.type || []))];
      },
      allCovert() {
        return [...new Set(this.links.flatMap(l => l.covert || []))];
      },
      allCovertGenerated() {
        return [...new Set(this.links.flatMap(l => l.covert_generated || []))];
      },
  },
  watch: {
    selectedCovert: 'refreshGraph',
    selectedCovertGenerated: 'refreshGraph',
    selectedTypes: 'refreshGraph',
    filteredLinks: {
      handler(newLinks) {
        try {
          if (this.d3) {
            this.processLinks();
            this.updateGraphData(newLinks);
          }
        } catch (error) {
          console.error('Filter update error:', error);
        }
      },
      immediate: true,
      deep: true
    },
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
    refreshGraph() {
      this.processLinks();
      this.updateGraphData(this.filteredLinks);
    },

    resetSimulation() {
        // if (!this.d3 || !this.simulation) return;

        // // Stop existing simulation
        // this.simulation.stop();
        
        // // Get current dimensions
        // const { width, height } = this.getDimensions();

        // // Reinitialize forces
        // this.simulation
        //   .force('charge', this.d3.forceManyBody().strength(-10000))
        //   .force('center', this.d3.forceCenter(width / 2, height / 2))
        //   .force('link', this.d3.forceLink(this.processedLinks).id(d => d.id).distance(100))
        //   .force('collision', this.d3.forceCollide().radius(5));

        // // Restart with fresh alpha
        // this.simulation.alpha(1).restart();
      if (this.simulation) {
        this.simulation.stop();
        this.simulation.nodes([]);
        this.simulation.force('link', null);
        this.simulation.force('charge', null);
        this.simulation.force('center', null);
        this.simulation = null;
      }
    },
    initZoom() {
      const svg = this.d3.select(this.$refs.graphContainer).select('svg')

      // Initialize zoomGroup if it doesn't exist
      if (!this.zoomGroup) {
        this.zoomGroup = svg.append('g')
      }

      this.zoom = this.d3.zoom()
        .scaleExtent([0.05, 20])
        .on('zoom', (event) => {
          if (this.zoomGroup) {
            this.zoomGroup.attr('transform', event.transform)
          }
        })

      svg.call(this.zoom)
    },
    clearSelection() {
      this.selectedNode = null
      this.selectedConnection = null
    },
    resetZoom() {
      const svg = this.d3.select(this.$refs.graphContainer).select('svg')
      svg.transition()
        .duration(250)
        .call(this.zoom.transform, this.d3.zoomIdentity)
    },
    processLinks() {
      const nodeMap = new Map(this.nodes.map(node => [node.id, node]));


      this.processedLinks = this.filteredLinks
        .map(link => {
          try {
            const source = nodeMap.get(link.source?.id || link.source);
            const target = nodeMap.get(link.target?.id || link.target);
            
            if (!source || !target) {
              console.warn('Invalid link:', link);
              return null;
            }
            
            return {
              ...link,
              source,
              target
            };
          } catch (error) {
            console.error('Link processing error:', error);
            return null;
          }
        })
        .filter(Boolean);

        // Collect nodes from processedLinks
        const nodeIds = new Set();
        this.processedLinks.forEach(link => {
          nodeIds.add(link.source.id);
          nodeIds.add(link.target.id);
        });
        this.filteredNodes = this.nodes.filter(node => nodeIds.has(node.id));
    },
    getDimensions() {
      return {
        width: this.$refs.graphContainer.clientWidth,
        height: this.$refs.graphContainer.clientHeight,
      };
    },

    initGraph() {
      const container = this.$refs.graphContainer;
      if (!container) return;

      const { width, height } = this.getDimensions();

      // Clear existing SVG
      container.innerHTML = '';
      
      const svg = this.d3.select(container)
        .append("svg")
        .attr("width", width)
        .attr("height", height)
        .style("background", "#f7ede2");

      this.zoomGroup = svg.append("g");
      this.initZoom();

      // Initialize simulation
      this.simulation = this.d3.forceSimulation()
        .force("link", this.d3.forceLink().id(d => d.id).distance(10))
        .force("charge", this.d3.forceManyBody().strength(-10))
        .force("center", this.d3.forceCenter(width / 2, height / 2))
        .force("collision", this.d3.forceCollide().radius(5))
        .on("tick", () => this.tickHandler());

      // Initial data bind
      this.updateGraphData(this.filteredLinks);

    },
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
    },

    handleBackgroundClick() {
      this.clearSelection()
    },

    updateMousePosition(event) {
      const container = this.$refs.graphContainer.getBoundingClientRect()
      const transform = this.d3.zoomTransform(this.zoomGroup.node())
      
      this.mousePosition = {
        x: (event.clientX - container.left - transform.x) / transform.k,
        y: (event.clientY - container.top - transform.y) / transform.k
      }
    },
    handleNodeHover(event, d) {
      this.hoveredElement = { type: "node", data: d };
      this.updateMousePosition(event);
    },

    handleNodeClick(event, d) {
      this.selectedNode = d;
      this.selectedConnection = null;
      event.stopPropagation();
    },

    handleLinkHover(event, d) {
      this.hoveredElement = { type: "link", data: d };
      this.updateMousePosition(event);
    },

    updateGraphData(filteredLinks) {
      if (!this.d3 || !this.simulation) return;

      // Process links while maintaining node references
      const nodeMap = new Map(this.nodes.map(n => [n.id, n]));
      this.processedLinks = filteredLinks.map(link => ({
        ...link,
        source: nodeMap.get(link.source?.id || link.source),
        target: nodeMap.get(link.target?.id || link.target)
      })).filter(link => link.source && link.target);

      // Preserve existing node positions
      const existingNodes = new Map(this.nodes.map(n => [n.id, n]));

      // Update simulation forces
      this.simulation
        .force("link", this.d3.forceLink(this.processedLinks).id(d => d.id).distance(100))
        .force("charge", this.d3.forceManyBody().strength(-100));

      // Update nodes
      const nodes = this.zoomGroup.selectAll(".node-group")
        .data(this.nodes, d => d.id);

      nodes.exit().remove();

      const nodeEnter = nodes.enter()
        .append("g")
        .attr("class", "node-group")
        .call(this.dragHandler());

      // Add node elements
      nodeEnter.append("circle")
        .attr("r", 15)
        .attr("fill", "transparent")
        .attr("pointer-events", "visible");

      nodeEnter.append("circle")
        .attr("r", 5)
        .attr("fill", "#264653");

      // nodeEnter.append("text")
      //   .text(d => d.id)
      //   .attr("dx", 8)
      //   .attr("dy", 4)
      //   .style("font-size", "10px")
      //   .style("fill", "#264653");

      // Merge nodes
      const nodeMerge = nodeEnter.merge(nodes)
        .on("mouseover", (event, d) => this.handleNodeHover(event, d))
        .on("click", (event, d) => this.handleNodeClick(event, d));

      // Update links
      const links = this.zoomGroup.selectAll("line")
        .data(this.processedLinks, d => `${d.source.id}-${d.target.id}`);

      links.exit()
        .transition()
        .duration(500)
        .style("opacity", 0)
        .remove();

      const linkEnter = links.enter()
        .append("line")
        .attr("stroke", d => this.getLinkColor(d))
        .attr("stroke-width", 7)
        .style("opacity", 0)
        .on("mouseover", (event, d) => this.handleLinkHover(event, d))
        .on("click", (event, d) => this.handleLinkClick(event, d));

      links.merge(linkEnter)
        .transition()
        .duration(500)
        .style("opacity", 1)
        .attr("stroke", d => this.getLinkColor(d));

      // Restore positions and restart simulation
      this.nodes.forEach(n => {
        const existing = existingNodes.get(n.id);
        if (existing) {
          n.x = existing.x;
          n.y = existing.y;
        }
      });

      this.simulation
        .nodes(this.filteredNodes)
        .alpha(0.5)
        .restart();
    },
    tickHandler() {
      this.zoomGroup.selectAll("line")
        .attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x)
        .attr("y2", d => d.target.y);

      this.zoomGroup.selectAll(".node-group")
        .attr("transform", d => `translate(${d.x},${d.y})`);
    },
    handleLinkClick(event, clickedLink) {
      event.stopPropagation();
      const nodePair = [
        clickedLink.source.id, 
        clickedLink.target.id
      ].sort().join('|');
      
      const allLinks = this.processedLinks.filter(l => {
        const currentPair = [l.source.id, l.target.id].sort().join('|');
        return currentPair === nodePair;
      });

      this.selectedConnection = {
        nodes: [clickedLink.source, clickedLink.target],
        links: allLinks
      };
      this.selectedNode = null;
    },

    getLinkColor(link) {
      console.log(link.covert, link.covert_generated)
      // if (link.covert === false && link.covert_generated === false) {
      //   return '#264653'; // Dark blue for non-covert and non-generated 
      // } else if (link.covert === false && link.covert_generated === true) {
      //   return '#fffff'; // White for non-covert but generated
      // } else if (link.covert === true && link.covert_generated === false) {
      //   return '#2a9d8f'; // Teal for covert but not generated
      // } else if (link.covert === true && link.covert_generated === true) {
      //   return '#e76f51'; // Coral for covert and generated
      // }
      if (link.covert === false) {
        return '#2a9d8f'; // teal
      } else if (link.covert === true) {
        return '#e76f51'; // Coral for covert and generated
      } else {
        return '#264653'; // Dark blue for non-covert and non-generated 
      }
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

.tooltip {
  position: absolute;
  background: white;
  border: 1px solid #ccc;
  padding: 8px 12px;
  border-radius: 4px;
  pointer-events: none;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  font-size: 14px;
  z-index: 100;
  max-width: 400px;
  color: #264653;
}

.graph-container {
  position: relative;
}

.graph-wrapper {
  display: flex;
  position: relative;
}

.side-menu {
  width: 400px;
  background: white;
  border-right: 1px solid #ccc;
  height: 100vh;
  overflow-y: auto;
  box-shadow: 2px 0 5px rgba(0,0,0,0.1);
  z-index: 100;
  position: fixed;
  left: 0;
  top: 0;
}

.menu-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 15px;
  border-bottom: 1px solid #eee;
  color:#264653
}

.menu-header button {
  background: none;
  border: none;
  font-size: 1.5em;
  cursor: pointer;
}

.menu-content {
  padding: 15px;
  color: #264653;

}

.graph-container {
  flex-grow: 1;
  margin-left: 300px; /* Match side menu width */
  height: 100vh;
  position: relative;
}

.link-item {
  margin: 10px 0;
  padding: 10px;
  border: 1px solid #eee;
  border-radius: 4px;
}

.link-item pre {
  margin: 0;
  font-size: 0.9em;
}

.node-group text {
  user-select: none;
  -webkit-user-select: none;
}

.link-properties {
  margin-bottom: 8px;
}

.link-properties div {
  margin: 4px 0;
  font-size: 0.9em;
  color: #2a9d8f;
}

pre {
  font-size: 0.8em;
  opacity: 0.7;
}

.filter-panel {
  position: fixed;
  right: 20px;
  top: 20px;
  background: white;
  padding: 15px;
  border: 1px solid #ccc;
  border-radius: 4px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  z-index: 1000;
  max-height: 80vh;
  overflow-y: auto;
  max-width: 300px;
}

.filter-item {
  margin: 5px 0;
}

.filter-item label {
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
  color:#264653
}

.filter-item input {
  margin: 0;
}

.bias-item {
  margin: 8px 0;
  padding: 6px;
  background: #f7f7f7;
  border-radius: 4px;
}

.bias-value {
  font-size: 0.85em;
  color: #666;
  margin-left: 12px;
}

.bias-color-indicator {
  display: inline-block;
  width: 12px;
  height: 12px;
  border-radius: 3px;
  margin-right: 8px;
  border: 1px solid #ddd;
}

.filter-item label {
  color: #264653; /* Dark blue */
  font-weight: 500;
}

.filter-item label {
  transition: color 0.2s ease;
}

.filter-item label:hover {
  color: #e76f51; /* Your coral color on hover */
}

.filter-panel h4 {
  color: #2a9d8f; /* Your teal color */
  border-bottom: 1px solid #eee;
  padding-bottom: 8px;
}

.filter-item input[type="checkbox"] {
  accent-color: #2a9d8f; /* Match your color scheme */
}

.legend {
  position: fixed;
  right: 20px;
  bottom: 20px;
  background: white;
  padding: 15px;
  border: 1px solid #ccc;
  border-radius: 4px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  z-index: 1000;
  max-height: 50vh;
  overflow-y: auto;
}

.legend-item {
  display: flex;
  align-items: center;
  margin: 5px 0;
}

.legend-color {
  width: 15px;
  height: 15px;
  border-radius: 3px;
  margin-right: 10px;
  border: 1px solid #ddd;
}

.legend-label {
  font-size: 0.9em;
}

line {
  stroke-opacity: 0.7 !important;
  stroke-width: 3px !important;
}

.filter-panel {
  width: 300px;
  padding: 15px;
}

.filter-group {
  margin-bottom: 15px;
}

.filter-group label {
  display: block;
  margin-bottom: 5px;
  font-weight: bold;
  color: #2a9d8f;
}

.filter-select {
  width: 100%;
  min-height: 100px;
  padding: 5px;
  border: 1px solid #ccc;
  border-radius: 4px;
  background: white;
  color:#264653

}

.filter-select option {
  padding: 3px;
  cursor: pointer;
}

.filter-select option:hover {
  background-color: #f0f0f0;
}

line.exit {
  opacity: 0 !important;
  pointer-events: none;
}

.bias-chart-panel {
  position: fixed;
  left: 320px;
  top: 20px;
  background: white;
  padding: 15px;
  border: 1px solid #ccc;
  border-radius: 4px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  z-index: 1000;
  max-width: 300px;
}

.bar-item {
  margin: 8px 0;
}

.bar-label {
  font-size: 0.9em;
  color: #264653;
  margin-bottom: 4px;
}

.bar-container {
  height: 8px;
  background: #eee;
  border-radius: 4px;
  overflow: hidden;
}

.bar {
  height: 100%;
  transition: width 0.3s ease;
}

/* Style the select box itself */
.filter-select {
  border: 1px solid #a5d6a7; /* Light muted green border */
  background-color: #f1f8e9; /* Very light green background */
}

/* Style when dropdown is open */
.filter-select:active, 
.filter-select:focus {
  border-color: #66bb6a; /* Medium muted green */
}

/* Scrollbar styling for the dropdown */
.filter-select::-webkit-scrollbar {
  width: 8px;
}

.filter-select::-webkit-scrollbar-thumb {
  background: #a5d6a7; /* Muted green scrollbar */
  border-radius: 4px;
}

/* Special styling for the bias filter */
.bias-filter option:checked {
  background-color: #e8f5e9 !important;
  color: #2e7d32 !important;
  font-weight: 600;
  position: relative;
}

/* Add checkmark indicator */
.bias-filter option:checked::after {
  content: "✓";
  position: absolute;
  right: 8px;
  color: #388e3c; /* Slightly brighter green */
}
</style>
