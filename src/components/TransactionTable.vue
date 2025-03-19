<script setup lang="ts">
import type { TableColumn } from '@nuxt/ui'

const UBadge = resolveComponent('UBadge')
const UButton = resolveComponent('UButton')
const UDropdownMenu = resolveComponent('UDropdownMenu')

type Transaction = {
  hash: number
  transaction_index: number
  from_address: string
  to_address: string
  value: number
  block_timestamp: string
  from_scam: boolean
  to_scam: boolean
}

const API_URL = 'http://localhost:8000/get_table_transactions?num_transactions=10000'

const searchTerm = ref('')
const columnFilters = ref<Record<string, any>>({
  value: { min: null, max: null },
  from_scam: 'all',
  to_scam: 'all'
})

const sortColumn = ref<keyof Transaction | null>(null)
const sortDirection = ref<'asc' | 'desc'>('asc')
const filteredCount = computed(() => filteredData.value?.length || 0)

const { data, status } = await useFetch<Transaction[]>(API_URL, {
  method: 'GET',
  headers: { 'Accept': '*/*' },
  transform: (res: any) => res.map(item => ({
    ...item,
    value: Number(item.value),
    from_scam: item.from_scam === 1, // Convert to boolean
    to_scam: item.to_scam === 1      // Convert to boolean
  })),
  lazy: true
})

const filteredData = computed(() => {
  if (!data.value) return []
  
  return data.value.filter(item => {
    // Global search
    const searchMatch = Object.entries(item).some(([key, value]) => {
      const stringValue = key === 'value' 
        ? (Number(value) / 1e6).toString()
        : String(value).toLowerCase()
      return stringValue.includes(searchTerm.value.toLowerCase())
    })

    // Value range filter (convert millions to original value for search)
    const value = item.value
    const min = columnFilters.value.value.min ? 
      Number(columnFilters.value.value.min) * 1e6 : -Infinity
    const max = columnFilters.value.value.max ? 
      Number(columnFilters.value.value.max) * 1e6 : Infinity
    const valueMatch = value >= min && value <= max

    // Scam filters
    const fromScamMatch = columnFilters.value.from_scam === 'all' ? true : 
      item.from_scam === (columnFilters.value.from_scam === 'yes')
      
    const toScamMatch = columnFilters.value.to_scam === 'all' ? true : 
      item.to_scam === (columnFilters.value.to_scam === 'yes')

    return searchMatch && valueMatch && fromScamMatch && toScamMatch
  }).sort((a, b) => {
    if (!sortColumn.value) return 0
    const modifier = sortDirection.value === 'asc' ? 1 : -1
    return a[sortColumn.value] > b[sortColumn.value] ? modifier : -modifier
  })
})

const columns: TableColumn<Transaction>[] = [
  {
    accessorKey: 'transaction_index',
    header: 'ID',
    sortable: true
  },
  {
    accessorKey: 'hash',
    header: 'Transaction Hash',
    sortable: true
  },
  {
    accessorKey: 'from_address',
    header: 'From Address',
    sortable: true
  },
  {
    accessorKey: 'to_address',
    header: 'To Address',
    sortable: true
  },
    {
    accessorKey: 'value',
    header: 'Value (Millions of ETH)',
    sortable: true,
    format: (value) => `$${(value / 1e6).toLocaleString()}M`
  },
  {
    accessorKey: 'from_scam',
    header: 'From Scam',
    sortable: true,
    format: (value) => Boolean(value) ? 'Yes' : 'No'
  },
  {
    accessorKey: 'to_scam',
    header: 'To Scam',
    sortable: true,
    format: (value) => Boolean(value) ? 'Yes' : 'No'
  }
]

function toggleSort(column: keyof Transaction) {
  if (sortColumn.value === column) {
    sortDirection.value = sortDirection.value === 'asc' ? 'desc' : 'asc'
  } else {
    sortColumn.value = column
    sortDirection.value = 'asc'
  }
}
</script>

<template>
  <client-only>
    <UCard class="mt-4">
      <!-- Header Content -->
     <template #header>
        <div class="flex flex-col gap-4">
          <!-- Search and Count -->
          <div class="flex items-center justify-between">
            <UInput
              v-model="searchTerm"
              placeholder="Global search..."
              icon="i-heroicons-magnifying-glass-20-solid"
              class="flex-1"
            />
            <div class="ml-4 text-sm text-gray-500">
              Showing {{ filteredCount }} of {{ data?.length || 0 }}
            </div>
          </div>

          <!-- Filters -->
          <div class="flex flex-wrap gap-4 items-center">
            <!-- Value Range -->
            <div class="flex items-center gap-2">
              <UInput
                v-model="columnFilters.value.min"
                placeholder="Min (M)"
                type="number"
                class="w-32"
                suffix="M"
              />
              <span class="text-gray-500">to</span>
              <UInput
                v-model="columnFilters.value.max"
                placeholder="Max (M)"
                type="number"
                class="w-32"
                suffix="M"
              />
            </div>
          </div>
        </div>
      </template>

      <!-- Table -->
      <UTable
        sticky
        :data="filteredData"
        :columns="columns"
        :loading="status === 'pending'"
        class="flex-1"
      >
        <!-- Compact Slot Templates -->
        <template #value-data="{ value }"><span class="font-mono">${{ (value / 1e6).toLocaleString() }}M</span></template>
        <template #from_scam-data="{ value }"><UBadge :label="value ? 'Yes' : 'No'" :color="value ? 'red' : 'green'"/></template>
        <template #to_scam-data="{ value }"><UBadge :label="value ? 'Yes' : 'No'" :color="value ? 'red' : 'green'"/></template>
      </UTable>
    </UCard>
  </client-only>
</template>