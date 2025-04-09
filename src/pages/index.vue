<template>
  <div id="heading">
    <h1>
      Graph-Based Representation of Ethereum Transactions
    </h1>
  </div>
  <div>
      <EthGraph></EthGraph>
      <div id="footer">
        <DrawerTable></DrawerTable>
      </div>
  </div>

</template>

<style lang="css">
#heading {
  position: fixed;
  top: 0; 
  left: 0;
  width: 100%;
  padding-top: 5vh;
  padding-left: 5vw;
}

#footer {
    position: fixed;
    bottom: 0;
    width: 100%;
    padding-bottom:5vh;
    padding-left: 5vh;
}

h1 {
  font-size: 3em
}

</style>

<script setup>
const colorMode = useColorMode()

console.log(colorMode.preference)
</script>


<style>
body {
  background-color: #fff;
  color: rgba(0,0,0,0.8);
}
.dark-mode body {
  background-color: #1a2834;
  color: #ebf4f1;
}
.sepia-mode body {
  background-color: #f1e7d0;
  color: #433422;
}
</style>