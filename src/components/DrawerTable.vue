<template>
    <div class="center">
    <UDrawer
        title="Ethereum Transactions"
        description="10k transactions on the Ethereum blockchain network."
    >
        <UButton label="View Transactions" color="neutral" variant="subtle"/>

        <template #body>
            <TransactionTable></TransactionTable>
        </template>
    </UDrawer>
    </div>
</template>

<style lang="css">
.center {
    display: flex;
    justify-content: center;
    align-items: center;
}

</style>