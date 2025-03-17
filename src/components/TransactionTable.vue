<script setup lang="ts">
import { h, resolveComponent } from 'vue'
import { upperFirst } from 'scule'

const UButton = resolveComponent('UButton')
const UCheckbox = resolveComponent('UCheckbox')
const UBadge = resolveComponent('UBadge')
const UDropdownMenu = resolveComponent('UDropdownMenu')

// const data = ref([
//   {
//     id: '4600',
//     date: '2024-03-11T15:30:00',
//     status: 'paid',
//     email: 'james.anderson@example.com',
//     amount: 594
//   },
//   {
//     id: '4599',
//     date: '2024-03-11T10:10:00',
//     status: 'failed',
//     email: 'mia.white@example.com',
//     amount: 276
//   },
//   {
//     id: '4598',
//     date: '2024-03-11T08:50:00',
//     status: 'refunded',
//     email: 'william.brown@example.com',
//     amount: 315
//   },
//   {
//     id: '4597',
//     date: '2024-03-10T19:45:00',
//     status: 'paid',
//     email: 'emma.davis@example.com',
//     amount: 529
//   },
//   {
//     id: '4596',
//     date: '2024-03-10T15:55:00',
//     status: 'paid',
//     email: 'ethan.harris@example.com',
//     amount: 639
//   }
// ])
interface ApiItem {
  id: number
  name: string
}

const API_URL = 'http://localhost:8000/items'

// Reactive state
const data = ref<ApiItem[] | null>(null)
const error = ref<Error | null>(null)
const pending = ref(false)

const fetchData = async () => {
  try {
    pending.value = true
    error.value = null
    data.value = null

    console.log('Starting fetch...')
    
    const { data: response, error: fetchError } = await useFetch<ApiItem[]>(API_URL, {
      method: 'GET',
      headers: {
        'Accept': '*/*',
      },
      transform: (res) => {
        console.log('[Transform] Raw response:', res)
        return res
      },
      onRequest({ request, options }) {
        console.log('[onRequest]', request, options)
      },
      onResponse({ response }) {
        console.log('[onResponse] Status:', response.status)
        console.log('[onResponse] Headers:', response.headers)
        console.log('[onResponse] Body:', response._data)
      },
      onResponseError({ response }) {
        console.error('[onResponseError]', response)
      }
    })

    console.log('Fetch completed. Response:', response.value)
    console.log('Fetch error:', fetchError.value)

    if (fetchError.value) {
      throw fetchError.value
    }

    data.value = response.value
    console.log('Data assigned:', data.value)
    
  } catch (err) {
    error.value = err as Error
    console.error('[Catch] Error:', err)
  } finally {
    pending.value = false
    console.log('Final state:', { data: data.value, error: error.value })
  }
}


onMounted(async () => {
  try {
    const result = await $fetch('https://httpbin.org/get')
    console.log('External fetch test:', result)
  } catch (err) {
    console.error('External fetch failed:', err)
  }
})


</script>

<template>
    <UTable sticky :data="data"/>
     <button @click="fetchData">Fetch Data</button>
    <div>Pending: {{ pending }}</div>
    <div>Error: {{ error?.message }}</div>
    <div>Data: {{ data }}</div>
</template>

