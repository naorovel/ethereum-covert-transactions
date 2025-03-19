<script setup lang="ts">
import type { TableColumn } from '@nuxt/ui'

type Transaction = {
  hash: number
  transaction_index: number
  from_address: string
  to_address: string
  value: number
  block_timestamp: string
  from_scam: number
  to_scam: number
}

const API_URL = 'http://localhost:8000/get_table_transactions?num_transactions=10000'

const { data, status } = await useFetch<Transaction[]>(API_URL,{
    method: 'GET',
    headers: {
        'Accept': '*/*',
    },
    transform: (res) => {
        return res
    },
    lazy: true
})

const columns: TableColumn<Transaction>[] = [
    {
        accessorKey: 'transaction_index',
        header: 'ID'
    },
    {
        accessorKey: 'hash',
        header: 'Hash'
    },
    {
        accessorKey: 'from_address',
        header: 'From Address',
    },
    {
        accessorKey: 'to_address',
        header: 'To Address'
    },
    {
        accessorKey: 'value',
        header: 'Value'
    }
]
</script>

<template>
  <client-only>
    <UCard class="mt-4">
      <UTable sticky :data="data" :columns="columns" :loading="status === 'pending'" class="flex-1" />
    </UCard>
  </client-only>
</template>
